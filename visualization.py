"""
Comprehensive visualization and logging module
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional, List
import logging

# WandB and TensorBoard
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("WandB not available")

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    logging.warning("TensorBoard not available")

from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

logger = logging.getLogger(__name__)


class ComprehensiveLogger:
    """
    Comprehensive logger with WandB and TensorBoard support
    """
    
    def __init__(self, config):
        self.config = config
        self.use_wandb = config.logging.use_wandb and WANDB_AVAILABLE
        self.use_tensorboard = config.logging.use_tensorboard and TENSORBOARD_AVAILABLE
        
        # Initialize WandB
        if self.use_wandb:
            self._init_wandb()
        
        # Initialize TensorBoard
        if self.use_tensorboard:
            self._init_tensorboard()
        
        # Storage for metrics
        self.train_metrics = []
        self.val_metrics = []
        self.epoch_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
    
    def _init_wandb(self):
        """Initialize Weights & Biases"""
        try:
            wandb.init(
                project=self.config.logging.wandb_project,
                entity=self.config.logging.wandb_entity,
                name=self.config.logging.wandb_run_name or self.config.experiment_name,
                config=self.config.to_dict(),
                reinit=True
            )
            logger.info("✓ WandB initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize WandB: {e}")
            self.use_wandb = False
    
    def _init_tensorboard(self):
        """Initialize TensorBoard"""
        try:
            log_dir = Path(self.config.logging.tensorboard_dir) / self.config.experiment_name
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(log_dir))
            logger.info(f"✓ TensorBoard initialized at {log_dir}")
        except Exception as e:
            logger.warning(f"Failed to initialize TensorBoard: {e}")
            self.use_tensorboard = False
    
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ''):
        """Log scalar metrics"""
        if self.use_wandb:
            wandb_metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
            wandb.log(wandb_metrics, step=step)
        
        if self.use_tensorboard:
            for key, value in metrics.items():
                self.writer.add_scalar(f"{prefix}{key}", value, step)
    
    def log_epoch_metrics(self, epoch: int, train_metrics: Dict, val_metrics: Dict, lr: float):
        """Log epoch-level metrics"""
        # Store history
        self.epoch_history['train_loss'].append(train_metrics.get('loss', 0))
        self.epoch_history['train_acc'].append(train_metrics.get('accuracy', 0))
        self.epoch_history['val_loss'].append(val_metrics.get('loss', 0))
        self.epoch_history['val_acc'].append(val_metrics.get('accuracy', 0))
        self.epoch_history['learning_rate'].append(lr)
        
        # Combine metrics
        combined_metrics = {
            **{f'train/{k}': v for k, v in train_metrics.items()},
            **{f'val/{k}': v for k, v in val_metrics.items()},
            'learning_rate': lr
        }
        
        self.log_metrics(combined_metrics, step=epoch)
        
        # Log to console
        logger.info(f"\nEpoch {epoch} Summary:")
        logger.info(f"  Train Loss: {train_metrics.get('loss', 0):.4f} | Train Acc: {train_metrics.get('accuracy', 0):.4f}")
        logger.info(f"  Val Loss: {val_metrics.get('loss', 0):.4f} | Val Acc: {val_metrics.get('accuracy', 0):.4f}")
        logger.info(f"  Learning Rate: {lr:.6f}")
    
    def log_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           class_names: List[str], epoch: int):
        """Log confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=ax,
                   cbar_kws={'label': 'Count'})
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_title(f'Confusion Matrix - Epoch {epoch}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Log to WandB
        if self.use_wandb:
            wandb.log({"confusion_matrix": wandb.Image(fig)}, step=epoch)
        
        # Log to TensorBoard
        if self.use_tensorboard:
            self.writer.add_figure('confusion_matrix', fig, epoch)
        
        plt.close(fig)
    
    def log_roc_curve(self, y_true: np.ndarray, y_probs: np.ndarray, epoch: int):
        """Log ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(fpr, tpr, color='darkorange', lw=3,
               label=f'ROC curve (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'ROC Curve - Epoch {epoch}', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Log to WandB
        if self.use_wandb:
            wandb.log({
                "roc_curve": wandb.Image(fig),
                "auc_roc": roc_auc
            }, step=epoch)
        
        # Log to TensorBoard
        if self.use_tensorboard:
            self.writer.add_figure('roc_curve', fig, epoch)
            self.writer.add_scalar('metrics/auc_roc', roc_auc, epoch)
        
        plt.close(fig)
        
        return roc_auc
    
    def log_precision_recall_curve(self, y_true: np.ndarray, y_probs: np.ndarray, epoch: int):
        """Log Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        pr_auc = auc(recall, precision)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(recall, precision, color='purple', lw=3,
               label=f'PR curve (AUC = {pr_auc:.4f})')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(f'Precision-Recall Curve - Epoch {epoch}', fontsize=14, fontweight='bold')
        ax.legend(loc="lower left", fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Log to WandB
        if self.use_wandb:
            wandb.log({
                "pr_curve": wandb.Image(fig),
                "auc_pr": pr_auc
            }, step=epoch)
        
        # Log to TensorBoard
        if self.use_tensorboard:
            self.writer.add_figure('pr_curve', fig, epoch)
            self.writer.add_scalar('metrics/auc_pr', pr_auc, epoch)
        
        plt.close(fig)
    
    def log_calibration_curve(self, y_true: np.ndarray, y_probs: np.ndarray, 
                             epoch: int, n_bins: int = 10):
        """Log calibration curve (reliability diagram)"""
        # Compute calibration
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        confidences = []
        accuracies = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_probs >= bin_lower) & (y_probs < bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = (y_true[in_bin] == (y_probs[in_bin] > 0.5)).mean()
                avg_confidence_in_bin = y_probs[in_bin].mean()
                confidences.append(avg_confidence_in_bin)
                accuracies.append(accuracy_in_bin)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        ax.plot(confidences, accuracies, 'o-', markersize=10, linewidth=3,
               color='red', label='Model calibration')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel('Confidence', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(f'Calibration Plot - Epoch {epoch}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Log to WandB
        if self.use_wandb:
            wandb.log({"calibration_curve": wandb.Image(fig)}, step=epoch)
        
        # Log to TensorBoard
        if self.use_tensorboard:
            self.writer.add_figure('calibration_curve', fig, epoch)
        
        plt.close(fig)
    
    def log_learning_curves(self, epoch: int):
        """Log learning curves"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        epochs = range(1, len(self.epoch_history['train_loss']) + 1)
        
        # Loss curves
        axes[0, 0].plot(epochs, self.epoch_history['train_loss'], 'b-', 
                       label='Training Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.epoch_history['val_loss'], 'r-', 
                       label='Validation Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Loss Curves', fontsize=14, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy curves
        axes[0, 1].plot(epochs, self.epoch_history['train_acc'], 'b-',
                       label='Training Accuracy', linewidth=2)
        axes[0, 1].plot(epochs, self.epoch_history['val_acc'], 'r-',
                       label='Validation Accuracy', linewidth=2)
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Accuracy', fontsize=12)
        axes[0, 1].set_title('Accuracy Curves', fontsize=14, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate
        axes[1, 0].plot(epochs, self.epoch_history['learning_rate'], 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Learning Rate', fontsize=12)
        axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Overfitting gap
        gap = [t - v for t, v in zip(self.epoch_history['train_acc'], self.epoch_history['val_acc'])]
        axes[1, 1].plot(epochs, gap, 'purple', linewidth=2)
        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Accuracy Gap (Train - Val)', fontsize=12)
        axes[1, 1].set_title('Overfitting Analysis', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Log to WandB
        if self.use_wandb:
            wandb.log({"learning_curves": wandb.Image(fig)}, step=epoch)
        
        # Log to TensorBoard
        if self.use_tensorboard:
            self.writer.add_figure('learning_curves', fig, epoch)
        
        plt.close(fig)
    
    def log_sample_predictions(self, images: torch.Tensor, labels: torch.Tensor,
                              predictions: torch.Tensor, probs: torch.Tensor,
                              epoch: int, num_samples: int = 8):
        """Log sample predictions with images"""
        if not self.use_wandb:
            return
        
        # Denormalize images
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        class_names = ['Benign', 'Malignant']
        
        # Select samples
        num_samples = min(num_samples, len(images))
        sample_images = []
        
        for i in range(num_samples):
            img = images[i].cpu() * std + mean
            img = torch.clamp(img, 0, 1)
            img_np = img.permute(1, 2, 0).numpy()
            
            true_label = class_names[labels[i].item()]
            pred_label = class_names[predictions[i].item()]
            confidence = probs[i, predictions[i]].item()
            
            # Create figure
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(img_np)
            ax.axis('off')
            
            # Color based on correctness
            color = 'green' if labels[i] == predictions[i] else 'red'
            title = f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.3f}'
            ax.set_title(title, color=color, fontsize=11, fontweight='bold')
            
            plt.tight_layout()
            sample_images.append(wandb.Image(fig))
            plt.close(fig)
        
        wandb.log({"sample_predictions": sample_images}, step=epoch)
    
    def log_uncertainty_distribution(self, epistemic: np.ndarray, 
                                    aleatoric: Optional[np.ndarray],
                                    y_true: np.ndarray, y_pred: np.ndarray,
                                    epoch: int):
        """Log uncertainty distributions"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Epistemic uncertainty distribution
        correct = (y_true == y_pred)
        axes[0, 0].hist(epistemic[correct], bins=50, alpha=0.7, label='Correct', color='green')
        axes[0, 0].hist(epistemic[~correct], bins=50, alpha=0.7, label='Incorrect', color='red')
        axes[0, 0].set_xlabel('Epistemic Uncertainty', fontsize=12)
        axes[0, 0].set_ylabel('Frequency', fontsize=12)
        axes[0, 0].set_title('Epistemic Uncertainty Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Epistemic vs Error scatter
        axes[0, 1].scatter(epistemic[correct], np.zeros(correct.sum()), 
                          alpha=0.5, label='Correct', color='green')
        axes[0, 1].scatter(epistemic[~correct], np.ones((~correct).sum()),
                          alpha=0.5, label='Incorrect', color='red')
        axes[0, 1].set_xlabel('Epistemic Uncertainty', fontsize=12)
        axes[0, 1].set_ylabel('Error (0=correct, 1=incorrect)', fontsize=12)
        axes[0, 1].set_title('Uncertainty vs Correctness', fontsize=14, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        
        if aleatoric is not None:
            # Aleatoric uncertainty distribution
            axes[1, 0].hist(aleatoric[correct], bins=50, alpha=0.7, label='Correct', color='green')
            axes[1, 0].hist(aleatoric[~correct], bins=50, alpha=0.7, label='Incorrect', color='red')
            axes[1, 0].set_xlabel('Aleatoric Uncertainty', fontsize=12)
            axes[1, 0].set_ylabel('Frequency', fontsize=12)
            axes[1, 0].set_title('Aleatoric Uncertainty Distribution', fontsize=14, fontweight='bold')
            axes[1, 0].legend(fontsize=10)
            axes[1, 0].grid(True, alpha=0.3)
            
            # Epistemic vs Aleatoric scatter
            axes[1, 1].scatter(epistemic, aleatoric, c=correct, cmap='RdYlGn', alpha=0.5)
            axes[1, 1].set_xlabel('Epistemic Uncertainty', fontsize=12)
            axes[1, 1].set_ylabel('Aleatoric Uncertainty', fontsize=12)
            axes[1, 1].set_title('Epistemic vs Aleatoric Uncertainty', fontsize=14, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Log to WandB
        if self.use_wandb:
            wandb.log({"uncertainty_analysis": wandb.Image(fig)}, step=epoch)
        
        # Log to TensorBoard
        if self.use_tensorboard:
            self.writer.add_figure('uncertainty_analysis', fig, epoch)
        
        plt.close(fig)
    
    def log_gradient_flow(self, named_parameters, epoch: int):
        """Log gradient flow through network"""
        layers = []
        avg_grads = []
        max_grads = []
        
        for name, param in named_parameters:
            if param.requires_grad and param.grad is not None:
                layers.append(name.replace('.weight', '').replace('.bias', ''))
                avg_grads.append(param.grad.abs().mean().cpu().item())
                max_grads.append(param.grad.abs().max().cpu().item())
        
        if len(avg_grads) == 0:
            return
        
        # Subsample for readability
        step = max(1, len(layers) // 30)
        layers = layers[::step]
        avg_grads = avg_grads[::step]
        max_grads = max_grads[::step]
        
        fig, ax = plt.subplots(figsize=(16, 8))
        x = np.arange(len(layers))
        ax.bar(x, avg_grads, alpha=0.7, label='Average Gradient', color='blue')
        ax.bar(x, max_grads, alpha=0.7, label='Max Gradient', color='red')
        ax.hlines(0, 0, len(avg_grads), linewidth=2, color='k')
        ax.set_xticks(x)
        ax.set_xticklabels(layers, rotation=90, fontsize=8)
        ax.set_xlabel('Layers', fontsize=12)
        ax.set_ylabel('Gradient Magnitude', fontsize=12)
        ax.set_title('Gradient Flow', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        # Log to WandB
        if self.use_wandb:
            wandb.log({"gradient_flow": wandb.Image(fig)}, step=epoch)
        
        # Log to TensorBoard
        if self.use_tensorboard:
            self.writer.add_figure('gradient_flow', fig, epoch)
        
        plt.close(fig)
    
    def close(self):
        """Close loggers"""
        if self.use_wandb:
            wandb.finish()
        if self.use_tensorboard:
            self.writer.close()
        
        logger.info("✓ Loggers closed")
