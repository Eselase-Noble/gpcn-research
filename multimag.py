"""
multimag.py — Multi-magnification fusion pipeline for GPCN-ViT (publishable).

WHY THIS IS VALID SCIENCE
-------------------------
In BreakHis the directory layout is:
    histology_slides/breast/<class>/SOB/<tumor_type>/<slide_folder>/<mag>/<img>.png
Every <slide_folder> holds the SAME tumour imaged at all four magnifications
(40X/100X/200X/400X). A multi-magnification sample is therefore one image drawn
from each magnification OF THE SAME SLIDE — genuine multi-scale views of one
tumour. We do NOT claim pixel-level registration (the fields of view differ); the
model performs multi-scale FEATURE fusion via attention. Patient-level splitting
is preserved (a patient never appears in two splits), so there is no leakage.

This drives the existing models.MultiMagnificationGPCNViT (per-magnification
heads + cross-magnification attention fusion + learnable magnification weights).

Entry points:
    from multimag import train_multimag
    result = train_multimag(config)        # returns dict with test metrics
"""

import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from augmentation import get_train_transform, get_val_transform
from dataset import create_patient_level_split
from PIL import Image

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Grouping BreakHis images by slide (same tumour, all magnifications)
# ----------------------------------------------------------------------------
def build_multimag_groups(root_dir: str,
                          magnifications: List[str],
                          patient_ids: Optional[List[str]] = None) -> List[Dict]:
    """Group images by slide folder, keeping only slides that have ALL requested
    magnifications. Optionally filter to a set of patient IDs (for the split).

    Returns a list of dicts:
        {'slide': str, 'patient_id': str, 'label': int, 'paths': {mag: [paths]}}
    """
    root = Path(root_dir)
    # slide_folder -> mag -> [paths]
    slides: Dict[Path, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    slide_meta: Dict[Path, Dict] = {}

    for mag in magnifications:
        for img_path in root.glob(f"histology_slides/breast/*/SOB/*/*/{mag}/*.png"):
            parts = img_path.parts
            try:
                breast_idx = parts.index('breast')
                class_name = parts[breast_idx + 1]
                if class_name not in ('benign', 'malignant'):
                    continue
                patient_id = '-'.join(img_path.stem.split('_')[2].split('-')[:3])
                if patient_ids is not None and patient_id not in patient_ids:
                    continue
                slide_folder = img_path.parent.parent  # .../<slide_folder>
                slides[slide_folder][mag].append(str(img_path))
                slide_meta[slide_folder] = {
                    'patient_id': patient_id,
                    'label': 0 if class_name == 'benign' else 1,
                }
            except (ValueError, IndexError):
                continue

    groups = []
    for slide_folder, mag_to_paths in slides.items():
        if all(len(mag_to_paths.get(m, [])) > 0 for m in magnifications):
            groups.append({
                'slide': str(slide_folder),
                'patient_id': slide_meta[slide_folder]['patient_id'],
                'label': slide_meta[slide_folder]['label'],
                'paths': {m: sorted(mag_to_paths[m]) for m in magnifications},
            })
    return groups


class MultiMagnificationDataset(Dataset):
    """Yields (list_of_images[one per magnification], label, patient_id).

    For each slide we create ``max_count`` samples (max over magnifications of the
    number of images), pairing image j of each magnification by ``j % len``. This
    uses every image of the densest magnification and cycles the sparser ones. In
    training mode the per-magnification image is chosen randomly each epoch
    (acts as multi-view augmentation); in eval mode it is deterministic.
    """

    def __init__(self, groups: List[Dict], magnifications: List[str],
                 transform=None, train: bool = True):
        self.groups = groups
        self.magnifications = magnifications
        self.transform = transform
        self.train = train

        # Flatten to an index list of (group_idx, sample_idx)
        self.index = []
        for gi, g in enumerate(groups):
            n = max(len(g['paths'][m]) for m in magnifications)
            for j in range(n):
                self.index.append((gi, j))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        gi, j = self.index[idx]
        g = self.groups[gi]
        imgs = []
        for m in self.magnifications:
            paths = g['paths'][m]
            if self.train:
                path = random.choice(paths)
            else:
                path = paths[j % len(paths)]
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            imgs.append(img)
        return imgs, g['label'], g['patient_id']

    def class_weights(self) -> torch.Tensor:
        labels = [self.groups[gi]['label'] for gi, _ in self.index]
        counts = np.bincount(labels, minlength=2)
        w = len(labels) / (2 * np.maximum(counts, 1))
        return torch.FloatTensor(w)


def _multimag_collate(batch):
    """Collate (list_of_M_imgs, label, pid) -> (list_of_M_batched, labels, pids)."""
    M = len(batch[0][0])
    images = [torch.stack([b[0][m] for b in batch], dim=0) for m in range(M)]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    pids = [b[2] for b in batch]
    return images, labels, pids


def create_multimag_dataloaders(config) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    mags = list(config.data.magnifications)
    train_p, val_p, test_p = create_patient_level_split(
        config.data.data_root, mags,
        config.data.test_size, config.data.val_size, config.seed
    )

    train_groups = build_multimag_groups(config.data.data_root, mags, train_p)
    val_groups = build_multimag_groups(config.data.data_root, mags, val_p)
    test_groups = build_multimag_groups(config.data.data_root, mags, test_p)

    train_ds = MultiMagnificationDataset(train_groups, mags, get_train_transform(config), train=True)
    val_ds = MultiMagnificationDataset(val_groups, mags, get_val_transform(config), train=False)
    test_ds = MultiMagnificationDataset(test_groups, mags, get_val_transform(config), train=False)

    common = dict(num_workers=config.data.num_workers, pin_memory=config.data.pin_memory,
                  collate_fn=_multimag_collate)
    train_loader = DataLoader(train_ds, batch_size=config.data.batch_size, shuffle=True,
                              drop_last=True, **common)
    val_loader = DataLoader(val_ds, batch_size=config.data.batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=config.data.batch_size, shuffle=False, **common)

    info = {
        'num_classes': config.model.num_classes,
        'class_weights': train_ds.class_weights(),
        'magnifications': mags,
        'train_slides': len(train_groups), 'val_slides': len(val_groups),
        'test_slides': len(test_groups),
        'train_samples': len(train_ds), 'val_samples': len(val_ds), 'test_samples': len(test_ds),
        'train_patients': len(train_p), 'val_patients': len(val_p), 'test_patients': len(test_p),
    }
    logger.info(f"[multimag] {info}")
    return train_loader, val_loader, test_loader, info


# ----------------------------------------------------------------------------
# Model construction
# ----------------------------------------------------------------------------
def build_multimag_model(config) -> nn.Module:
    """Build base GPCNViT then wrap in MultiMagnificationGPCNViT."""
    from model import GPCNViT, MultiMagnificationGPCNViT
    base = GPCNViT(
        num_classes=config.model.num_classes,
        pretrained=config.model.pretrained,
        num_gpcn_layers=config.model.num_gpcn_layers,
        k=config.model.gpcn_k,
        use_hybrid=config.model.use_hybrid_knn,
        alpha=config.model.alpha,
        use_multi_scale=config.model.use_multi_scale,
        pyramid_levels=config.model.pyramid_levels,
        freeze_backbone=config.model.freeze_backbone,
        hidden_dim=config.model.hidden_dim,
        dropout=config.model.dropout,
        backbone=getattr(config.model, 'backbone', 'vit_base_patch16_224'),
        backbone_weights_path=getattr(config.model, 'backbone_weights_path', ''),
        use_gpcn=getattr(config.model, 'use_gpcn', True),
    )
    return MultiMagnificationGPCNViT(base, num_magnifications=len(config.data.magnifications))


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate_multimag(model, loader, device, threshold: float = 0.5):
    from metrics import MetricsCalculator
    import torch.nn.functional as F
    model.eval()
    all_true, all_probs = [], []
    for images, labels, _ in loader:
        images = [im.to(device) for im in images]
        logits = model(images)
        probs = F.softmax(logits, dim=-1)
        all_true.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_true)
    y_probs = np.array(all_probs)
    if y_probs.shape[1] == 2 and threshold != 0.5:
        y_pred = (y_probs[:, 1] >= threshold).astype(int)
    else:
        y_pred = y_probs.argmax(axis=1)

    calc = MetricsCalculator(num_classes=y_probs.shape[1])
    metrics = calc.calculate_all_metrics(y_true, y_pred, y_probs)
    metrics.update(calc.calculate_calibration_metrics(y_true, y_probs))
    return metrics, y_true, y_pred, y_probs


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def train_multimag(config) -> Dict:
    """Full training loop for the multi-magnification fusion model."""
    from utils import set_seed, ModelEMA, GradientClipping, save_checkpoint
    from losses import FocalLoss, LabelSmoothingCrossEntropy
    from metrics import find_optimal_threshold
    import torch.optim as optim

    set_seed(config.seed, config.deterministic)
    device = torch.device(config.device)

    train_loader, val_loader, test_loader, info = create_multimag_dataloaders(config)
    model = build_multimag_model(config).to(device)
    class_weights = info['class_weights'].to(device)

    # Loss: class-weighted focal (handles imbalance) or label smoothing.
    if config.training.use_focal_loss:
        criterion = FocalLoss(alpha=config.training.focal_alpha,
                              gamma=config.training.focal_gamma,
                              class_weights=class_weights).to(device)
    else:
        criterion = LabelSmoothingCrossEntropy(config.training.label_smoothing).to(device)

    # Discriminative LR: low for pretrained backbone, high for fusion/heads.
    new_keys = ('gpcn', 'fusion', '.alpha', 'head', 'mag_heads', 'mag_weights')
    if getattr(config.training, 'use_discriminative_lr', False):
        bb = [p for n, p in model.named_parameters() if p.requires_grad and not any(k in n for k in new_keys)]
        nw = [p for n, p in model.named_parameters() if p.requires_grad and any(k in n for k in new_keys)]
        param_groups = [{'params': bb, 'lr': config.training.backbone_lr},
                        {'params': nw, 'lr': config.training.head_lr}]
    else:
        param_groups = model.parameters()
    optimizer = optim.AdamW(param_groups, lr=config.training.learning_rate,
                            weight_decay=config.training.weight_decay, betas=config.training.betas)

    warmup = int(getattr(config.training, 'warmup_epochs', 0) or 0)
    cosine_epochs = max(1, config.training.num_epochs - warmup)
    cos = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=config.training.min_lr)
    if warmup > 0:
        warm = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup)
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warm, cos], milestones=[warmup])
    else:
        scheduler = cos

    use_amp = config.training.use_amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    clipper = GradientClipping(max_norm=config.training.gradient_clip)
    ema = ModelEMA(model, decay=config.training.ema_decay) if getattr(config.training, 'use_ema', False) else None

    save_dir = Path(config.data.save_dir) / (config.experiment_name or 'gpcn_multimag') / 'checkpoints'
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / 'best_multimag.pth'
    last_path = save_dir / 'last_multimag.pth'

    best_auc, best_epoch = -1.0, 0
    start_epoch = 0
    monitor = getattr(config.training, 'monitor_metric', 'auc_roc')

    # Crash/timeout safety: auto-resume from last_multimag.pth if present.
    if getattr(config.training, 'auto_resume', True) and last_path.exists():
        try:
            ck = torch.load(str(last_path), map_location=device, weights_only=False)
            model.load_state_dict(ck['model_state_dict'])
            optimizer.load_state_dict(ck['optimizer_state_dict'])
            scheduler.load_state_dict(ck['scheduler_state_dict'])
            if ema is not None and ck.get('ema_state'):
                ema.load_state_dict(ck['ema_state'])
            start_epoch = ck.get('epoch', -1) + 1
            best_auc = ck.get('best_auc', -1.0)
            best_epoch = ck.get('best_epoch', 0)
            logger.info(f"✓ [multimag] checkpoint found — resuming from epoch "
                        f"{start_epoch + 1} / {config.training.num_epochs} "
                        f"(best {monitor} so far: {best_auc:.4f} @ epoch {best_epoch + 1})")
        except Exception as e:
            logger.warning(f"[multimag] auto-resume failed ({e}); starting fresh.")

    logger.info("=" * 70 + "\nMULTI-MAGNIFICATION FUSION TRAINING\n" + "=" * 70)
    for epoch in range(start_epoch, config.training.num_epochs):
        model.train()
        running = 0.0
        for images, labels, _ in train_loader:
            images = [im.to(device) for im in images]
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clipper(model.parameters())
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); clipper(model.parameters()); optimizer.step()
            if ema is not None:
                ema.update(model)
            running += loss.item()
        scheduler.step()

        # Validate (with EMA weights if enabled)
        if ema is not None:
            ema.apply_to(model)
        try:
            val_metrics, *_ = evaluate_multimag(model, val_loader, device)
        finally:
            if ema is not None:
                ema.restore(model)
        score = val_metrics.get(monitor, val_metrics['accuracy'])
        logger.info(f"Epoch {epoch+1}/{config.training.num_epochs} | loss={running/max(len(train_loader),1):.4f} "
                    f"| val_acc={val_metrics['accuracy']:.4f} val_auc={val_metrics.get('auc_roc',0):.4f}")

        if score > best_auc:
            best_auc, best_epoch = score, epoch
            if ema is not None:
                ema.apply_to(model)
            try:
                torch.save({'model_state_dict': model.state_dict(),
                            'epoch': epoch, 'best_auc': best_auc}, best_path)
            finally:
                if ema is not None:
                    ema.restore(model)
            logger.info(f"  ✓ new best ({monitor}={score:.4f}) saved")

        # Full training state — used for both the rolling 'last' checkpoint and
        # the periodic per-epoch checkpoints.
        full_ckpt = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'ema_state': ema.state_dict() if ema is not None else None,
            'epoch': epoch, 'best_auc': best_auc, 'best_epoch': best_epoch,
            'val_metrics': {k: float(v) for k, v in val_metrics.items()},
        }

        # Always overwrite last_multimag.pth so a crash loses at most one epoch.
        torch.save(full_ckpt, last_path)

        # Periodic per-epoch checkpoints (mirrors the per-magnification trainer's
        # checkpoint_epoch_N.pth), so you can see/resume from a specific epoch.
        save_freq = getattr(config.training, 'save_frequency', 5)
        if save_freq and (epoch + 1) % save_freq == 0:
            torch.save(full_ckpt, save_dir / f'checkpoint_epoch_{epoch + 1}.pth')
            logger.info(f"  ✓ saved checkpoint_epoch_{epoch + 1}.pth")

    # Load best, tune threshold on val, evaluate test
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)['model_state_dict'])

    test_threshold = 0.5
    if getattr(config.validation, 'optimize_threshold', False):
        _, v_true, _, v_probs = evaluate_multimag(model, val_loader, device)
        test_threshold, thr_info = find_optimal_threshold(
            v_true, v_probs[:, 1],
            mode=getattr(config.validation, 'threshold_mode', 'youden'),
            fn_cost=getattr(config.validation, 'fn_cost', 10.0),
            fp_cost=getattr(config.validation, 'fp_cost', 1.0))
        logger.info(f"✓ Tuned multimag threshold ({thr_info['mode']}): {test_threshold:.4f}")

    test_metrics, *_ = evaluate_multimag(model, test_loader, device, threshold=test_threshold)
    test_metrics['decision_threshold'] = test_threshold
    logger.info("\nMulti-magnification Test Results:")
    for k, v in test_metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    return {'best_val_auc': best_auc, 'best_epoch': best_epoch,
            'test_metrics': test_metrics, 'info': info}


if __name__ == '__main__':
    from config import get_full_config
    cfg = get_full_config()
    cfg.experiment_name = 'gpcn_multimag'
    train_multimag(cfg)
