"""
train.py
Main training script for GPCN-ViT Breast Cancer Classification
"""

import torch
import argparse
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import get_default_config, get_quick_test_config, get_full_config
from utils import set_seed, setup_logging, print_system_info, count_parameters, create_experiment_directory
from dataset import create_dataloaders, create_augmented_dataloaders
from model import create_model
from trainer import Trainer
import logging


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train GPCN-ViT for Breast Cancer Classification')
    
    # Experiment settings
    parser.add_argument('--experiment-name', type=str, default='gpcn_vit_baseline',
                      help='Name of experiment')
    parser.add_argument('--config-type', type=str, default='default',
                      choices=['default', 'quick', 'full'],
                      help='Configuration preset')
    
    # Data settings
    parser.add_argument('--data-root', type=str, 
                      default='/content/drive/MyDrive/MY RESEARCH/GPCN-ViT Breast Cancer Diagnosis/BreakHisDataset/breakhis/BreaKHis_v1/BreaKHis_v1',
                      help='Root directory of BreakHis dataset')
    parser.add_argument('--batch-size', type=int, default=None,
                      help='Batch size (overrides config)')
    parser.add_argument('--magnification', type=str, default='40X',
                      choices=['40X', '100X', '200X', '400X'],
                      help='Magnification level')
    
    # Model settings
    parser.add_argument('--num-gpcn-layers', type=int, default=None,
                      help='Number of GPCN layers (overrides config)')
    parser.add_argument('--use-multi-scale', action='store_true',
                      help='Use multi-scale graph pyramid')
    parser.add_argument('--use-uncertainty', action='store_true',
                      help='Use uncertainty quantification')
    
    # Training settings
    parser.add_argument('--num-epochs', type=int, default=None,
                      help='Number of epochs (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                      help='Learning rate (overrides config)')
    parser.add_argument('--no-amp', action='store_true',
                      help='Disable automatic mixed precision')
    
    # Logging
    parser.add_argument('--no-wandb', action='store_true',
                      help='Disable WandB logging')
    parser.add_argument('--no-tensorboard', action='store_true',
                      help='Disable TensorBoard logging')
    parser.add_argument('--wandb-project', type=str, default=None,
                      help='WandB project name')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                      help='Path to checkpoint to resume from')
    parser.add_argument('--test-only', action='store_true',
                      help='Only run test evaluation')
    
    return parser.parse_args()


def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    if args.config_type == 'quick':
        config = get_quick_test_config()
    elif args.config_type == 'full':
        config = get_full_config()
    else:
        config = get_default_config()
    
    # Override config with command line arguments
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    if args.data_root:
        config.data.data_root = args.data_root
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.magnification:
        config.data.train_magnification = args.magnification
    if args.num_gpcn_layers is not None:
        config.model.num_gpcn_layers = args.num_gpcn_layers
    if args.use_multi_scale:
        config.model.use_multi_scale = True
    if args.use_uncertainty:
        config.model.use_uncertainty = True
    if args.num_epochs is not None:
        config.training.num_epochs = args.num_epochs
    if args.lr is not None:
        config.training.learning_rate = args.lr
    if args.no_amp:
        config.training.use_amp = False
    if args.no_wandb:
        config.logging.use_wandb = False
    if args.no_tensorboard:
        config.logging.use_tensorboard = False
    if args.wandb_project:
        config.logging.wandb_project = args.wandb_project
    if args.seed:
        config.seed = args.seed
    
    # Create experiment directory
    exp_dir = create_experiment_directory(
        config.data.save_dir,
        config.experiment_name
    )
    
    # Setup logging
    log_file = exp_dir / 'logs' / 'training.log'
    logger = setup_logging(str(log_file))
    
    # Print system info
    print_system_info(logger)
    
    # Set random seed
    logger.info(f"\nSetting random seed: {config.seed}")
    set_seed(config.seed, config.deterministic)
    
    # Print configuration
    logger.info("\n" + "=" * 70)
    logger.info("Configuration")
    logger.info("=" * 70)
    logger.info(f"Experiment: {config.experiment_name}")
    logger.info(f"Device: {config.device}")
    logger.info(f"Seed: {config.seed}")
    logger.info(f"\nData:")
    logger.info(f"  Root: {config.data.data_root}")
    logger.info(f"  Magnification: {config.data.train_magnification}")
    logger.info(f"  Batch size: {config.data.batch_size}")
    logger.info(f"  Use patient split: {config.data.use_patient_split}")
    logger.info(f"\nModel:")
    logger.info(f"  Backbone: {config.model.backbone}")
    logger.info(f"  GPCN layers: {config.model.num_gpcn_layers}")
    logger.info(f"  K neighbors: {config.model.gpcn_k}")
    logger.info(f"  Use multi-scale: {config.model.use_multi_scale}")
    logger.info(f"  Use uncertainty: {config.model.use_uncertainty}")
    logger.info(f"\nTraining:")
    logger.info(f"  Epochs: {config.training.num_epochs}")
    logger.info(f"  Learning rate: {config.training.learning_rate}")
    logger.info(f"  Optimizer: {config.training.optimizer}")
    logger.info(f"  Scheduler: {config.training.scheduler}")
    logger.info(f"  Use AMP: {config.training.use_amp}")
    logger.info(f"  Use focal loss: {config.training.use_focal_loss}")
    logger.info(f"  Use contrastive loss: {config.training.use_contrastive_loss}")
    logger.info("=" * 70)
    
    # Check if dataset exists
    data_root = Path(config.data.data_root)
    if not data_root.exists():
        logger.error(f"Dataset not found at: {data_root}")
        logger.error("Please update the data_root path in config or pass --data-root argument")
        sys.exit(1)
    
    # Create dataloaders
    logger.info("\n" + "=" * 70)
    logger.info("Loading Dataset")
    logger.info("=" * 70)
    
    try:
        device = torch.device(config.device)
        train_loader, val_loader, test_loader, data_info = create_augmented_dataloaders(config,device)
        logger.info("✓ Dataset loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Create model
    logger.info("\n" + "=" * 70)
    logger.info("Creating Model")
    logger.info("=" * 70)
    
    try:
        model = create_model(config)
        
        # Print model info
        param_counts = count_parameters(model)
        logger.info("✓ Model created successfully")
        logger.info(f"  Total parameters: {param_counts['total']:,}")
        logger.info(f"  Trainable parameters: {param_counts['trainable']:,}")
        logger.info(f"  Frozen parameters: {param_counts['frozen']:,}")
        
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test model with sample batch
    logger.info("\nTesting model with sample batch...")
    try:
        sample_images, sample_labels = next(iter(train_loader))[:2]
        sample_images = sample_images.to(config.device)
        
        with torch.no_grad():
            if config.model.use_uncertainty:
                sample_output, _, _ = model(sample_images[:2], return_uncertainty=False)
            else:
                sample_output = model(sample_images[:2])
        
        logger.info(f"  Input shape: {sample_images.shape}")
        logger.info(f"  Output shape: {sample_output.shape}")
        logger.info("✓ Model forward pass successful")
        
    except Exception as e:
        logger.error(f"Model forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Create trainer
    logger.info("\n" + "=" * 70)
    logger.info("Initializing Trainer")
    logger.info("=" * 70)
    
    try:
        trainer = Trainer(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            class_weights=data_info['class_weights'].to(config.device)
        )
        logger.info("✓ Trainer initialized")
        
    except Exception as e:
        logger.error(f"Failed to initialize trainer: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Resume from checkpoint — explicit path takes priority, then auto-detect last_checkpoint.pth
    resume_path = args.resume
    if not resume_path:
        auto_last = Path(config.data.save_dir) / config.experiment_name / 'checkpoints' / 'last_checkpoint.pth'
        if auto_last.exists():
            resume_path = str(auto_last)
            logger.info(f"\nAuto-detected checkpoint: {resume_path}")

    if resume_path:
        logger.info(f"\nResuming from checkpoint: {resume_path}")
        from utils import load_checkpoint
        try:
            checkpoint_info = load_checkpoint(
                resume_path, trainer.model, trainer.optimizer, trainer.scheduler
            )
            trainer.current_epoch = checkpoint_info['epoch'] + 1
            trainer.best_val_acc  = checkpoint_info.get('best_metric', 0.0)
            trainer.global_step   = checkpoint_info.get('global_step', 0)
            trainer.best_val_auc  = checkpoint_info.get('best_val_auc', 0.0)
            trainer.best_epoch    = checkpoint_info.get('best_epoch', 0)
            logger.info(f"✓ Resumed from epoch {checkpoint_info['epoch']} "
                        f"(best acc: {trainer.best_val_acc:.4f})")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            sys.exit(1)
    
    # Train or test
    if args.test_only:
        logger.info("\n" + "=" * 70)
        logger.info("Running Test Evaluation Only")
        logger.info("=" * 70)
        
        from metrics import evaluate_model
        test_metrics, y_true, y_pred, y_probs = evaluate_model(
            model, test_loader, config.device, use_uncertainty=config.model.use_uncertainty
        )
        
        logger.info("\nTest Results:")
        for key, value in test_metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        
    else:
        # Train model
        try:
            results = trainer.train()
            
            logger.info("\n" + "=" * 70)
            logger.info("Training Summary")
            logger.info("=" * 70)
            logger.info(f"Best Validation Accuracy: {results['best_val_acc']:.4f}")
            logger.info(f"Best Validation AUC: {results['best_val_auc']:.4f}")
            logger.info(f"Best Epoch: {results['best_epoch'] + 1}")
            logger.info("\nTest Set Results:")
            for key, value in results['test_metrics'].items():
                logger.info(f"  {key}: {value:.4f}")
            logger.info("=" * 70)
            
            logger.info(f"\n✓ Training completed successfully!")
            logger.info(f"✓ Checkpoints saved to: {trainer.save_dir}")
            logger.info(f"✓ Logs saved to: {exp_dir / 'logs'}")
            
        except KeyboardInterrupt:
            logger.info("\n⚠ Training interrupted by user")
            sys.exit(0)
        except Exception as e:
            logger.error(f"\n✗ Training failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    main()
