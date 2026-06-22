"""
Utility functions for the GPCN-ViT project
"""

import torch
import numpy as np
import random
import os
from pathlib import Path
from typing import Optional, Dict, Any
import logging
import sys


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO):
    """Setup logging configuration"""
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    return logging.getLogger(__name__)


def set_seed(seed: int = 42, deterministic: bool = True):
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed
        deterministic: If True, uses deterministic algorithms (may be slower)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set environment variable for deterministic behavior
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters
    
    Returns:
        Dictionary with total, trainable, and frozen parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': frozen_params
    }


def format_time(seconds: float) -> str:
    """Format time in seconds to human-readable string"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def get_device_info() -> Dict[str, Any]:
    """Get information about available devices"""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if torch.cuda.is_available():
        info['device_name'] = torch.cuda.get_device_name(0)
        info['cuda_version'] = torch.version.cuda
        info['cudnn_version'] = torch.backends.cudnn.version()
        
        # Memory info
        for i in range(torch.cuda.device_count()):
            mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
            mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
            
            info[f'device_{i}_memory'] = {
                'total_gb': f"{mem_total:.2f}",
                'reserved_gb': f"{mem_reserved:.2f}",
                'allocated_gb': f"{mem_allocated:.2f}"
            }
    
    return info


def create_experiment_directory(base_dir: str, experiment_name: str) -> Path:
    """
    Create directory structure for experiment
    
    Returns:
        Path to experiment directory
    """
    exp_dir = Path(base_dir) / experiment_name
    
    # Create subdirectories
    (exp_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (exp_dir / 'logs').mkdir(parents=True, exist_ok=True)
    (exp_dir / 'visualizations').mkdir(parents=True, exist_ok=True)
    (exp_dir / 'predictions').mkdir(parents=True, exist_ok=True)
    (exp_dir / 'metrics').mkdir(parents=True, exist_ok=True)
    
    return exp_dir


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self, name: str, fmt: str = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' depending on metric
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        self.compare = np.less if mode == 'min' else np.greater
    
    def __call__(self, score: float) -> bool:
        """
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.compare(score, self.best_score - self.min_delta):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False


def extract_patient_id(image_path: str) -> str:
    """
    Extract patient ID from BreakHis image path
    
    Format: SOB_B_A-14-22549AB_40X_0001.png
    Patient ID: 14-22549AB
    """
    from pathlib import Path
    filename = Path(image_path).stem
    
    try:
        # Split by underscore
        parts = filename.split('_')
        # Patient ID is typically the 3rd part
        patient_id = parts[2]
        return patient_id
    except:
        # Fallback: use filename
        return filename


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    scheduler: Optional[Any] = None) -> Dict[str, Any]:
    """
    Load checkpoint

    Returns:
        Dictionary with epoch, best_metric, global_step, best_val_auc, best_epoch, etc.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Load into base_model when the wrapper exposes one (consistent with save_checkpoint)
    target = model.base_model if hasattr(model, 'base_model') else model
    target.load_state_dict(checkpoint['model_state_dict'])

    # Load optimizer if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load scheduler if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return {
        'epoch': checkpoint.get('epoch', 0),
        'best_metric': checkpoint.get('best_metric', 0.0),
        'metrics': checkpoint.get('metrics', {}),
        'global_step': checkpoint.get('global_step', 0),
        'best_val_auc': checkpoint.get('best_val_auc', 0.0),
        'best_epoch': checkpoint.get('best_epoch', 0),
    }


def save_checkpoint(checkpoint_path: str, model: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   scheduler: Any,
                   epoch: int,
                   metrics: Dict[str, float],
                   best_metric: float,
                   config: Optional[Dict] = None,
                   **extra):
    """Save checkpoint.  Extra keyword args (global_step, best_val_auc, …) are stored as-is."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics,
        'best_metric': best_metric,
    }
    checkpoint.update(extra)

    if config is not None:
        checkpoint['config'] = config

    torch.save(checkpoint, checkpoint_path)


class ModelEMA:
    """
    Exponential Moving Average of model weights.

    Keeps a shadow copy of the trainable parameters that is updated after every
    optimizer step. At evaluation time call apply_to(model) to swap the EMA
    weights in (backing up the live weights), then restore(model) afterwards.
    EMA weights are typically smoother and generalise slightly better.
    """
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {n: p.detach().clone()
                       for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: torch.nn.Module):
        """Swap EMA weights into the model, backing up the live weights."""
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.backup[n] = p.detach().clone()
                p.data.copy_(self.shadow[n])

    @torch.no_grad()
    def restore(self, model: torch.nn.Module):
        """Restore the live weights saved by apply_to()."""
        for n, p in model.named_parameters():
            if n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}

    def state_dict(self):
        return {'decay': self.decay, 'shadow': self.shadow}

    def load_state_dict(self, state):
        self.decay = state.get('decay', self.decay)
        self.shadow = state.get('shadow', self.shadow)


class GradientClipping:
    """Gradient clipping with monitoring"""
    def __init__(self, max_norm: float = 1.0):
        self.max_norm = max_norm
        self.grad_norms = []
    
    def __call__(self, parameters):
        """Clip gradients and return norm"""
        total_norm = torch.nn.utils.clip_grad_norm_(parameters, self.max_norm)
        self.grad_norms.append(total_norm.item())
        return total_norm.item()
    
    def get_stats(self):
        """Get gradient norm statistics"""
        if not self.grad_norms:
            return {}
        
        return {
            'mean': np.mean(self.grad_norms),
            'std': np.std(self.grad_norms),
            'min': np.min(self.grad_norms),
            'max': np.max(self.grad_norms)
        }


def print_system_info(logger=None):
    """Print system and environment information"""
    info = get_device_info()
    
    msg = "=" * 70 + "\n"
    msg += "System Information\n"
    msg += "=" * 70 + "\n"
    msg += f"PyTorch Version: {torch.__version__}\n"
    msg += f"CUDA Available: {info['cuda_available']}\n"
    
    if info['cuda_available']:
        msg += f"CUDA Version: {info['cuda_version']}\n"
        msg += f"cuDNN Version: {info['cudnn_version']}\n"
        msg += f"Device Count: {info['device_count']}\n"
        msg += f"Device Name: {info['device_name']}\n"
        
        for key, value in info.items():
            if 'memory' in key:
                msg += f"\n{key}:\n"
                for k, v in value.items():
                    msg += f"  {k}: {v}\n"
    
    msg += "=" * 70
    
    if logger:
        logger.info(msg)
    else:
        print(msg)
