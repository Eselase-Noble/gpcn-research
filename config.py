"""
Configuration file for GPCN-ViT Breast Cancer Classification
Author: Research Team
Date: 2026

config.py
"""

import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Base architecture
    # Backbone is configurable. VERIFIED drop-in ViT-B/16/768 options (no HF auth):
    #   'vit_base_patch16_224'  -> ImageNet baseline (no download auth needed)
    #   'hf-hub:1aurent/vit_base_patch16_224.owkin_pancancer'  -> Phikon
    #        (Owkin iBOT ViT-B/16 pretrained on TCGA H&E). RECOMMENDED — biggest
    #        expected gain on BreakHis, true drop-in (embed_dim 768). [VERIFIED]
    #   'hf-hub:1aurent/vit_base_patch16_224.kaiko_ai_towards_large_pathology_fms'
    #        -> Kaiko.ai pathology FM (ViT-B/16, alternative)
    # NOTE: Lunit-DINO is only ViT-S (embed_dim 384), not ViT-B — avoid for the
    #       ViT-B/16 architecture story. UNI (MahmoodLab) is ViT-L and gated.
    backbone: str = 'hf-hub:1aurent/vit_base_patch16_224.owkin_pancancer'
    backbone_weights_path: str = ''  # optional local checkpoint to load into the ViT backbone
    num_classes: int = 2
    embed_dim: int = 768
    pretrained: bool = True

    # Ablation
    use_gpcn: bool = True  # set False to train a plain ViT baseline (ablation)

    # GPCN parameters
    num_gpcn_layers: int = 3  # Use multiple GPCN layers
    gpcn_k: int = 8  # Number of neighbors
    use_hybrid_knn: bool = True
    alpha: float = 0.3  # Spatial vs feature weight
    use_checkpoint: bool = True  # Gradient checkpointing for memory
    
    # Multi-magnification fusion rule (ablation of the attention head).
    #   'attention' -> cross-mag MHA + learned vote (ours); 'none' -> vote only;
    #   'mean' -> mean-pool CLS + shared head; 'concat' -> concat CLS + MLP.
    # build_fusion_model (fusion_baselines.py) reads this to train each arm.
    fusion_mode: str = 'attention'

    # Graph pyramid
    use_multi_scale: bool = True
    pyramid_levels: int = 3
    pooling_ratio: float = 0.5
    
    # Uncertainty
    use_uncertainty: bool = True
    mc_samples: int = 10
    
    # Classification head
    hidden_dim: int = 384
    dropout: float = 0.3
    
    # Training
    freeze_backbone: bool = False
    unfreeze_after_epoch: int = 5


@dataclass
class DataConfig:
    """Data configuration"""
    # Paths
    data_root: str = "/content/drive/MyDrive/MY RESEARCH/GPCN-ViT Breast Cancer Diagnosis/BreakHisDataset/breakhis/BreaKHis_v1/BreaKHis_v1"
    save_dir: str = "/content/drive/MyDrive/MY RESEARCH/gpcn_vit_outputs"
    
    # Dataset parameters
    magnifications: List[str] = field(default_factory=lambda: ['40X', '100X', '200X', '400X'])
    train_magnification: str = '40X'  # Primary training magnification (per-mag protocol)
    use_multi_magnification: bool = True  # Train on multiple magnifications

    # Magnification-pooled training: pool ALL four magnifications into one
    # patient-split dataset and train a single magnification-agnostic model.
    # This is the scientifically valid 'multi-magnification' setting for BreakHis
    # (the 4 magnifications are NOT registered, so per-image fusion is not valid).
    # Patient-level split is preserved across magnifications -> no leakage.
    pool_all_magnifications: bool = False
    
    # Image parameters
    image_size: int = 224
    patch_size: int = 16
    
    # Data split
    test_size: float = 0.2
    val_size: float = 0.1
    use_patient_split: bool = True  # Critical: prevent data leakage
    
    # Augmentation
    use_stain_augmentation: bool = True
    use_advanced_augmentation: bool = True
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    
    # Data loading
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Optimizer
    optimizer: str = 'adamw'
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)

    # Discriminative learning rate: low LR for the pretrained backbone,
    # higher LR for the randomly-initialised GPCN adapters + classification head.
    use_discriminative_lr: bool = True
    backbone_lr: float = 1e-5
    head_lr: float = 1e-3

    # Scheduler
    scheduler: str = 'cosine'
    num_epochs: int = 50
    warmup_epochs: int = 5  # now actually implemented (linear warmup -> cosine)
    min_lr: float = 1e-6
    
    # Loss function
    use_focal_loss: bool = True
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    
    use_contrastive_loss: bool = True
    contrastive_weight: float = 0.1
    contrastive_temperature: float = 0.07
    
    # Regularization
    label_smoothing: float = 0.1
    gradient_clip: float = 1.0

    # MixUp / CutMix (both are now applied; previously CutMix was dead code)
    use_mixup: bool = True
    use_cutmix: bool = True
    mix_prob: float = 0.5  # probability a batch gets MixUp/CutMix at all

    # Mixed precision
    use_amp: bool = True

    # Model selection / early stopping
    # Metric used to pick the best checkpoint and drive early stopping.
    # 'auc_roc' is more robust than 'accuracy' on imbalanced data.
    monitor_metric: str = 'auc_roc'

    # Exponential Moving Average of weights (cheap, reliable boost)
    use_ema: bool = True
    ema_decay: float = 0.999

    # Speed: skip the slow MC-dropout uncertainty pass during validation.
    # Uncertainty is still available at final test time.
    fast_validation: bool = True

    # Checkpointing
    save_frequency: int = 5  # Save every N epochs
    keep_best_n: int = 3

    # Crash/timeout safety: auto-resume from last_checkpoint.pth at train() start
    # (restores model/optimizer/scheduler/epoch/global_step/best metrics/EMA).
    # Just re-run the cell after a Colab disconnect and it continues seamlessly.
    auto_resume: bool = True


@dataclass
class ValidationConfig:
    """Validation and evaluation configuration"""
    # Evaluation metrics
    compute_auc: bool = True
    compute_calibration: bool = True
    compute_clinical_metrics: bool = True
    
    # Cross-validation
    use_kfold: bool = False
    n_folds: int = 5
    
    # External validation
    external_datasets: List[str] = field(default_factory=list)
    
    # Test-time augmentation (average predictions over h/v flips at test time).
    # Histology is orientation-invariant, so this is a safe, free accuracy gain
    # applied only at final evaluation.
    use_tta: bool = True

    # Decision-threshold optimisation. The operating point is tuned on the
    # validation set and then applied (frozen) to the test set — never tuned on
    # test. 'youden' balances sens/spec; 'cost' minimises clinical cost; 'f1'
    # maximises malignant-class F1.
    optimize_threshold: bool = True
    threshold_mode: str = 'youden'  # 'youden' | 'cost' | 'f1'

    # Uncertainty thresholds
    uncertainty_threshold: float = 0.5

    # Clinical cost weights
    fn_cost: float = 10.0  # False negative (miss cancer)
    fp_cost: float = 1.0   # False positive


@dataclass
class LoggingConfig:
    """Logging and visualization configuration"""
    # WandB
    use_wandb: bool = True
    wandb_project: str = 'gpcn-vit-breakhis'
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    
    # TensorBoard
    use_tensorboard: bool = True
    tensorboard_dir: str = 'runs/gpcn_vit'
    
    # Logging frequency
    log_interval: int = 10  # Log every N batches
    image_log_frequency: int = 5  # Log images every N epochs
    
    # Visualization
    visualize_attention: bool = True
    visualize_graphs: bool = True
    visualize_uncertainty: bool = True
    num_visualization_samples: int = 8
    
    # Save outputs
    save_predictions: bool = True
    save_confusion_matrix: bool = True
    save_roc_curves: bool = True


@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    # Sub-configs
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Experiment metadata
    experiment_name: str = "gpcn_vit_baseline"
    seed: int = 42
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Reproducibility
    deterministic: bool = True
    benchmark: bool = True
    
    def __post_init__(self):
        """Setup paths and create directories"""
        Path(self.data.save_dir).mkdir(parents=True, exist_ok=True)
        Path(self.logging.tensorboard_dir).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary for logging"""
        return {
            'model': self.model.__dict__,
            'data': self.data.__dict__,
            'training': self.training.__dict__,
            'validation': self.validation.__dict__,
            'logging': self.logging.__dict__,
            'experiment_name': self.experiment_name,
            'seed': self.seed,
            'device': self.device
        }


def get_default_config() -> ExperimentConfig:
    """Get default configuration"""
    return ExperimentConfig()


def get_quick_test_config() -> ExperimentConfig:
    """Get configuration for quick testing"""
    config = ExperimentConfig()
    config.experiment_name = "quick_test"
    config.training.num_epochs = 5
    config.data.batch_size = 8
    config.model.num_gpcn_layers = 3
    config.model.use_multi_scale = True
    config.model.use_uncertainty = True
    config.training.use_contrastive_loss = True
    return config


def get_full_config() -> ExperimentConfig:
    """Get configuration for full training run"""
    config = ExperimentConfig()
    config.experiment_name = "gpcn_vit_full"
    config.training.num_epochs = 100
    config.model.num_gpcn_layers = 3
    config.model.use_multi_scale = True
    config.model.use_uncertainty = True
    config.training.use_contrastive_loss = True
    config.data.use_multi_magnification = True
    return config
