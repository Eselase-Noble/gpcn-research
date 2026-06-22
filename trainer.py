"""
Comprehensive trainer for GPCN-ViT
FIXES APPLIED (two changes only, everything else identical to original):
  1. global_step counter — WandB step is now monotonically increasing
  2. Visualisation block — model output unpacked before .softmax() call
"""

import torch
import torch.nn as nn
import torch.optim as optim
try:
    # PyTorch >= 2.0 recommended API
    from torch.amp import GradScaler
    _AMP_NEW = True
except ImportError:
    from torch.cuda.amp import GradScaler
    _AMP_NEW = False
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import time
import logging

from utils import AverageMeter, EarlyStopping, save_checkpoint, GradientClipping, ModelEMA
from metrics import evaluate_model
from augmentation import MixUp, CutMix
from visualization import ComprehensiveLogger

logger = logging.getLogger(__name__)


class Trainer:
    """
    Comprehensive trainer with all features
    """

    def __init__(self, model: nn.Module, config, train_loader, val_loader, test_loader, class_weights: Optional[torch.Tensor] = None):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = torch.device(config.device)

        # Move model to device
        self.model = self.model.to(self.device)

        # Create loss criterion
        from losses import create_criterion
        self.criterion = create_criterion(config, class_weights)
        if class_weights is not None:
            self.criterion = self.criterion.to(self.device)

        # Create optimizer
        self.optimizer = self._create_optimizer()

        # Create scheduler
        self.scheduler = self._create_scheduler()

        # Mixed precision training
        self.use_amp = config.training.use_amp and self.device.type == 'cuda'
        if self.use_amp:
            self.scaler = GradScaler('cuda') if _AMP_NEW else GradScaler()
        else:
            self.scaler = None

        # Gradient clipping
        self.grad_clipper = GradientClipping(max_norm=config.training.gradient_clip)

        # Augmentation
        if config.data.use_advanced_augmentation:
            self.mixup = MixUp(alpha=config.data.mixup_alpha)
            self.cutmix = CutMix(alpha=config.data.cutmix_alpha)
        else:
            self.mixup = None
            self.cutmix = None

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=15,
            min_delta=0.001,
            mode='max'
        )

        # Metric used to select the best checkpoint / drive early stopping.
        # Defaults to AUC (robust on imbalanced data) instead of raw accuracy.
        self.monitor_metric = getattr(config.training, 'monitor_metric', 'accuracy')

        # Exponential Moving Average of weights
        self.use_ema = getattr(config.training, 'use_ema', False)
        self.ema = ModelEMA(self.model, decay=config.training.ema_decay) if self.use_ema else None

        # Logger
        self.logger = ComprehensiveLogger(config)

        # Best metrics tracking
        self.best_val_acc = 0.0
        self.best_val_auc = 0.0
        self.best_monitor = -float('inf')
        self.best_epoch = 0

        # Epoch tracking
        self.current_epoch = 0

        # FIX 1: global_step replaces epoch * len(loader) + batch_idx as the
        # WandB/TensorBoard step. It never resets between phases.
        # For multi-phase training, pass the previous step forward:
        #   trainer_phase2.global_step = trainer_phase1.global_step
        self.global_step = 0

        # Save directory
        self.save_dir = Path(config.data.save_dir) / config.experiment_name / 'checkpoints'
        self.save_dir.mkdir(parents=True, exist_ok=True)

        logger.info("✓ Trainer initialized")

    def _build_param_groups(self):
        """
        Build parameter groups for discriminative learning rates.

        The pretrained ViT backbone gets a low LR; the randomly-initialised
        GPCN adapters (gpcn/fusion/alpha), classification head and any
        multi-magnification heads get a higher LR. Without this the new
        modules under-train while the pretrained backbone over-trains.
        """
        cfg = self.config.training
        if not getattr(cfg, 'use_discriminative_lr', False):
            return self.model.parameters()

        new_module_keys = ('gpcn', 'fusion', '.alpha', 'head', 'mag_heads', 'mag_weights')
        backbone_params, new_params = [], []
        n_backbone = n_new = 0
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if any(k in name for k in new_module_keys):
                new_params.append(p)
                n_new += p.numel()
            else:
                backbone_params.append(p)
                n_backbone += p.numel()

        logger.info(f"Discriminative LR | backbone: {n_backbone:,} params @ lr={cfg.backbone_lr} "
                    f"| new modules: {n_new:,} params @ lr={cfg.head_lr}")

        return [
            {'params': backbone_params, 'lr': cfg.backbone_lr},
            {'params': new_params, 'lr': cfg.head_lr},
        ]

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer"""
        params = self._build_param_groups()
        if self.config.training.optimizer.lower() == 'adamw':
            optimizer = optim.AdamW(
                params,
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                betas=self.config.training.betas
            )
        elif self.config.training.optimizer.lower() == 'adam':
            optimizer = optim.Adam(
                params,
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                betas=self.config.training.betas
            )
        elif self.config.training.optimizer.lower() == 'sgd':
            optimizer = optim.SGD(
                params,
                lr=self.config.training.learning_rate,
                momentum=0.9,
                weight_decay=self.config.training.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")

        return optimizer

    def _create_scheduler(self):
        """Create learning rate scheduler"""
        warmup_epochs = int(getattr(self.config.training, 'warmup_epochs', 0) or 0)

        if self.config.training.scheduler.lower() == 'cosine':
            # FIX: warmup_epochs was configured but never applied. Now we do a
            # linear warmup for the first `warmup_epochs` epochs, then cosine
            # anneal over the remaining epochs. Stabilises pretrained-ViT fine-tuning.
            cosine_epochs = max(1, self.config.training.num_epochs - warmup_epochs)
            cosine = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=cosine_epochs,
                eta_min=self.config.training.min_lr
            )
            if warmup_epochs > 0:
                warmup = optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=0.01,
                    end_factor=1.0,
                    total_iters=warmup_epochs
                )
                scheduler = optim.lr_scheduler.SequentialLR(
                    self.optimizer,
                    schedulers=[warmup, cosine],
                    milestones=[warmup_epochs]
                )
            else:
                scheduler = cosine
        elif self.config.training.scheduler.lower() == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif self.config.training.scheduler.lower() == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
            )
        else:
            scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda epoch: 1.0
            )

        return scheduler

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()

        loss_meter = AverageMeter('Loss')
        acc_meter = AverageMeter('Accuracy')

        all_features = []
        all_labels = []

        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            # Parse batch
            if len(batch) == 2:
                images, labels = batch
            elif len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch[0], batch[1]

            images = images.to(self.device)
            labels = labels.to(self.device)
            batch_size = images.size(0)

            # Apply MixUp/CutMix. Previously only MixUp ran (CutMix was dead
            # code). Now, with probability `mix_prob`, randomly pick whichever
            # of the enabled augmentations to apply.
            mix_prob = getattr(self.config.training, 'mix_prob', 0.5)
            available = []
            if getattr(self.config.training, 'use_mixup', True) and self.mixup is not None:
                available.append(self.mixup)
            if getattr(self.config.training, 'use_cutmix', True) and self.cutmix is not None:
                available.append(self.cutmix)

            use_mixup = len(available) > 0 and np.random.rand() < mix_prob
            if use_mixup:
                mixer = available[np.random.randint(len(available))]
                images, labels_a, labels_b, lam = mixer(images, labels)

            # Forward pass
            self.optimizer.zero_grad()

            _amp_ctx = (
                torch.amp.autocast(device_type=self.device.type)
                if self.use_amp else torch.amp.autocast(device_type=self.device.type, enabled=False)
            )
            with _amp_ctx:
                if hasattr(self.model, 'base_model'):
                    features = self.model.base_model.forward_features(images)
                    logits = self.model.base_model.head(features[:, 0, :])
                    feature_vec = features[:, 0, :]
                else:
                    features = self.model.forward_features(images)
                    logits = self.model.head(features[:, 0, :])
                    feature_vec = features[:, 0, :]

                if use_mixup:
                    from losses import MixUpCrossEntropy
                    mixup_criterion = MixUpCrossEntropy()
                    loss = mixup_criterion(logits, labels_a, labels_b, lam)
                    loss_dict = {'total': loss}
                else:
                    if self.config.training.use_contrastive_loss:
                        loss, loss_dict = self.criterion(logits, labels, feature_vec)
                    else:
                        loss, loss_dict = self.criterion(logits, labels)

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm = self.grad_clipper(self.model.parameters())
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                grad_norm = self.grad_clipper(self.model.parameters())
                self.optimizer.step()

            # Update EMA shadow weights after each optimizer step
            if self.ema is not None:
                self.ema.update(self.model)

            # Compute accuracy
            _, predicted = logits.max(1)
            if not use_mixup:
                correct = predicted.eq(labels).sum().item()
                acc = 100. * correct / batch_size
            else:
                correct_a = predicted.eq(labels_a).sum().item()
                correct_b = predicted.eq(labels_b).sum().item()
                acc = 100. * (lam * correct_a + (1 - lam) * correct_b) / batch_size

            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(acc, batch_size)

            # Log batch metrics
            if batch_idx % self.config.logging.log_interval == 0:
                logger.info(
                    f'Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] '
                    f'Loss: {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                    f'Acc: {acc_meter.val:.2f}% ({acc_meter.avg:.2f}%) '
                    f'GradNorm: {grad_norm:.4f}'
                )

                # FIX 1: use self.global_step instead of
                #         epoch * len(self.train_loader) + batch_idx
                batch_metrics = {
                    'batch_loss': loss_meter.val,
                    'batch_accuracy': acc_meter.val,
                    'gradient_norm': grad_norm
                }
                batch_metrics.update({k: v.item() if isinstance(v, torch.Tensor) else v
                                       for k, v in loss_dict.items()})
                self.logger.log_metrics(batch_metrics, step=self.global_step, prefix='train/')

            # FIX 1: increment after every batch
            self.global_step += 1

        epoch_time = time.time() - start_time

        metrics = {
            'loss': loss_meter.avg,
            'accuracy': acc_meter.avg,
            'epoch_time': epoch_time
        }

        return metrics

    def validate_epoch(self, epoch: int) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
        """Validate for one epoch"""
        logger.info("Running validation...")

        # fast_validation skips the slow MC-dropout uncertainty pass (10x forward
        # passes) during training; full uncertainty is still computed at final test.
        fast_val = getattr(self.config.training, 'fast_validation', True)
        use_uncertainty = (self.config.model.use_uncertainty
                           and hasattr(self.model, 'forward')
                           and not fast_val)

        # Evaluate with EMA weights when enabled.
        if self.ema is not None:
            self.ema.apply_to(self.model)
        try:
            metrics, y_true, y_pred, y_probs = evaluate_model(
                self.model, self.val_loader, self.device, use_uncertainty=use_uncertainty
            )
        finally:
            if self.ema is not None:
                self.ema.restore(self.model)

        return metrics, y_true, y_pred, y_probs

    def _auto_resume(self):
        """Auto-resume from last_checkpoint.pth if present (crash/timeout safety).

        Restores model, optimizer, scheduler, epoch counter, global step, best
        metrics and the EMA shadow — in one load — so re-running the training
        cell after a Colab disconnect continues exactly where it stopped.
        """
        if not getattr(self.config.training, 'auto_resume', True):
            return
        if self.current_epoch > 0:  # already resumed manually by the caller
            return
        last_path = self.save_dir / 'last_checkpoint.pth'
        if not last_path.exists():
            logger.info("No checkpoint found — starting fresh.")
            return

        try:
            ckpt = torch.load(str(last_path), map_location='cpu', weights_only=False)
            target = self.model.base_model if hasattr(self.model, 'base_model') else self.model
            target.load_state_dict(ckpt['model_state_dict'])
            if 'optimizer_state_dict' in ckpt:
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scheduler_state_dict' in ckpt:
                self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            if self.ema is not None and ckpt.get('ema_state'):
                self.ema.load_state_dict(ckpt['ema_state'])

            self.current_epoch = ckpt.get('epoch', -1) + 1
            self.global_step = ckpt.get('global_step', 0)
            self.best_val_acc = ckpt.get('best_metric', 0.0)
            self.best_val_auc = ckpt.get('best_val_auc', 0.0)
            self.best_epoch = ckpt.get('best_epoch', 0)
            self.best_monitor = self.best_val_auc if self.monitor_metric == 'auc_roc' else self.best_val_acc
            logger.info(f"✓ Auto-resumed from epoch {ckpt.get('epoch', -1) + 1} "
                        f"| best {self.monitor_metric}={self.best_monitor:.4f} "
                        f"| EMA={'restored' if (self.ema is not None and ckpt.get('ema_state')) else 'n/a'}")
        except Exception as e:
            logger.warning(f"Auto-resume failed ({e}); starting fresh.")

    def train(self):
        """Full training loop"""
        logger.info("=" * 70)
        logger.info("Starting Training")
        logger.info("=" * 70)

        # Crash/timeout safety: pick up from last_checkpoint.pth if it exists.
        self._auto_resume()

        for epoch in range(self.current_epoch, self.config.training.num_epochs):
            self.current_epoch = epoch

            logger.info(f"\nEpoch {epoch + 1}/{self.config.training.num_epochs}")
            logger.info("-" * 70)

            # Unfreeze backbone after certain epochs
            if epoch == self.config.model.unfreeze_after_epoch and self.config.model.freeze_backbone:
                if hasattr(self.model, 'unfreeze_backbone'):
                    self.model.unfreeze_backbone()
                elif hasattr(self.model, 'base_model'):
                    self.model.base_model.unfreeze_backbone()
                logger.info("✓ Unfreezing backbone")

            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics, y_true, y_pred, y_probs = self.validate_epoch(epoch)

            # Get learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Log epoch metrics
            self.logger.log_epoch_metrics(epoch, train_metrics, val_metrics, current_lr)

            # Log visualizations periodically
            if (epoch + 1) % self.config.logging.image_log_frequency == 0:
                self.logger.log_confusion_matrix(
                    y_true, y_pred,
                    class_names=['Benign', 'Malignant'],
                    epoch=epoch
                )

                if val_metrics.get('auc_roc'):
                    self.logger.log_roc_curve(y_true, y_probs[:, 1], epoch)

                self.logger.log_precision_recall_curve(y_true, y_probs[:, 1], epoch)
                self.logger.log_calibration_curve(y_true, y_probs[:, 1], epoch)
                self.logger.log_learning_curves(epoch)

                self.logger.log_gradient_flow(self.model.named_parameters(), epoch)

                sample_batch = next(iter(self.val_loader))
                sample_images = sample_batch[0][:8].to(self.device)
                sample_labels = sample_batch[1][:8]

                with torch.no_grad():
                    # FIX 2: always call base_model directly so the result is
                    # a plain Tensor — not the (logits, None, None) tuple that
                    # UncertaintyGPCNViT.forward() returns.
                    if hasattr(self.model, 'base_model'):
                        sample_outputs = self.model.base_model(sample_images)
                    else:
                        sample_outputs = self.model(sample_images)
                    sample_probs = torch.softmax(sample_outputs, dim=-1)
                    _, sample_preds = sample_outputs.max(1)

                self.logger.log_sample_predictions(
                    sample_images, sample_labels, sample_preds, sample_probs, epoch
                )

            # Score used for selection / early stopping (default: AUC).
            monitor_score = val_metrics.get(self.monitor_metric, val_metrics['accuracy'])

            # Update scheduler
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(monitor_score)
            else:
                self.scheduler.step()

            # Save checkpoint (best is selected on monitor_metric, not raw accuracy)
            is_best = monitor_score > self.best_monitor
            if is_best:
                self.best_monitor = monitor_score
                self.best_val_acc = val_metrics['accuracy']
                self.best_val_auc = val_metrics.get('auc_roc', 0.0)
                self.best_epoch = epoch

                # Persist EMA weights as the "best" model so inference uses them.
                if self.ema is not None:
                    self.ema.apply_to(self.model)
                try:
                    self._save_checkpoint(epoch, val_metrics, is_best=True)
                finally:
                    if self.ema is not None:
                        self.ema.restore(self.model)
                logger.info(f"✓ New best model saved | {self.monitor_metric}: {monitor_score:.4f} "
                            f"| Acc: {self.best_val_acc:.4f} | AUC: {self.best_val_auc:.4f}")

            if (epoch + 1) % self.config.training.save_frequency == 0:
                self._save_checkpoint(epoch, val_metrics, is_best=False)

            # Always overwrite last_checkpoint.pth so crash recovery loses at most 1 epoch
            self._save_checkpoint(epoch, val_metrics, is_last=True)

            if self.early_stopping(monitor_score):
                logger.info(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break

        logger.info("\n" + "=" * 70)
        logger.info("Training Complete!")
        logger.info(f"Best Epoch: {self.best_epoch + 1}")
        logger.info(f"Best Validation Accuracy: {self.best_val_acc:.4f}")
        logger.info(f"Best Validation AUC: {self.best_val_auc:.4f}")
        logger.info("=" * 70)

        logger.info("\nEvaluating on test set...")
        # Load the best checkpoint (EMA weights if EMA was on) for the final report.
        best_path = self.save_dir / 'best_model.pth'
        if best_path.exists():
            try:
                from utils import load_checkpoint
                load_checkpoint(str(best_path), self.model)
                logger.info(f"✓ Loaded best checkpoint for final test: {best_path}")
            except Exception as e:
                logger.warning(f"Could not load best checkpoint ({e}); testing current weights.")

        use_tta = getattr(self.config.validation, 'use_tta', False)

        # Decision-threshold optimisation: tune the operating point on the
        # VALIDATION set, then freeze it for the test set. This attacks the
        # high-AUC / lower-accuracy gap and the sensitivity>>specificity skew
        # caused by class imbalance, without ever peeking at the test labels.
        test_threshold = 0.5
        if getattr(self.config.validation, 'optimize_threshold', False):
            from metrics import find_optimal_threshold
            _, val_true, _, val_probs = evaluate_model(
                self.model, self.val_loader, self.device,
                use_uncertainty=False, use_tta=use_tta
            )
            test_threshold, thr_info = find_optimal_threshold(
                val_true, val_probs[:, 1],
                mode=getattr(self.config.validation, 'threshold_mode', 'youden'),
                fn_cost=getattr(self.config.validation, 'fn_cost', 10.0),
                fp_cost=getattr(self.config.validation, 'fp_cost', 1.0),
            )
            logger.info(f"✓ Tuned decision threshold on val ({thr_info['mode']}): "
                        f"{test_threshold:.4f} "
                        f"| val sens={thr_info['sensitivity']:.3f} spec={thr_info['specificity']:.3f}")

        test_metrics, y_true, y_pred, y_probs = evaluate_model(
            self.model, self.test_loader, self.device,
            use_uncertainty=False, use_tta=use_tta, threshold=test_threshold
        )
        test_metrics['decision_threshold'] = test_threshold

        logger.info("\nTest Set Results:")
        for key, value in test_metrics.items():
            logger.info(f"  {key}: {value:.4f}")

        self.logger.close()

        return {
            'best_val_acc': self.best_val_acc,
            'best_val_auc': self.best_val_auc,
            'best_epoch': self.best_epoch,
            'test_metrics': test_metrics
        }

    def _save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False, is_last: bool = False):
        """Save checkpoint"""
        if is_best:
            checkpoint_name = 'best_model.pth'
        elif is_last:
            checkpoint_name = 'last_checkpoint.pth'
        else:
            checkpoint_name = f'checkpoint_epoch_{epoch + 1}.pth'
        checkpoint_path = self.save_dir / checkpoint_name

        # Persist EMA shadow weights too, so resume is faithful (the EMA average
        # is what we select/test on — losing it on resume would degrade results).
        ema_state = self.ema.state_dict() if self.ema is not None else None

        save_checkpoint(
            checkpoint_path=str(checkpoint_path),
            model=self.model.base_model if hasattr(self.model, 'base_model') else self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch,
            metrics=metrics,
            best_metric=self.best_val_acc,
            config=self.config.to_dict(),
            global_step=self.global_step,
            best_val_auc=self.best_val_auc,
            best_epoch=self.best_epoch,
            ema_state=ema_state,
        )