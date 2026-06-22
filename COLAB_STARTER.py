"""
GPCN-ViT Training - Google Colab Starter Script

This is a simplified starter script for Google Colab.
Upload all .py files to your Colab session, then run this cell-by-cell.
"""
import nvidia

# ============================================================================
# CELL 1: Setup Environment
# ============================================================================

# Check GPU
!nvidia-smi

# Install dependencies (takes ~2 minutes)
!pip install -q torch torchvision torchaudio
!pip install -q torch-geometric
!pip install -q pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
!pip install -q timm wandb opencv-python scikit-image

print("✅ Installation complete!")


# ============================================================================
# CELL 2: Mount Google Drive
# ============================================================================

from google.colab import drive
drive.mount('/content/drive')

# Set your dataset path here
DATASET_PATH = "/content/drive/MyDrive/YOUR_PATH_TO/BreaKHis_v1"
OUTPUT_PATH = "/content/drive/MyDrive/gpcn_vit_outputs"

print(f"Dataset path: {DATASET_PATH}")
print(f"Output path: {OUTPUT_PATH}")


# ============================================================================
# CELL 3: Upload Python Files
# ============================================================================

# Option A: Upload files directly to Colab
from google.colab import files
import os

print("Please upload all .py files from the implementation")
print("Required files: config.py, model.py, dataset.py, trainer.py, etc.")
uploaded = files.upload()
print(f"✅ Uploaded {len(uploaded)} files")

# OR

# Option B: Copy from Google Drive (if you've uploaded them there)
# !cp /content/drive/MyDrive/gpcn_vit_code/*.py /content/


# ============================================================================
# CELL 4: Verify Installation
# ============================================================================

# Test imports
try:
    from config import get_default_config, get_quick_test_config
    from model import create_model
    from dataset import create_dataloaders
    from trainer import Trainer
    print("✅ All modules imported successfully!")
except Exception as e:
    print(f"❌ Import error: {e}")
    print("Make sure all .py files are uploaded")


# ============================================================================
# CELL 5: Quick Test (5 epochs, ~10 minutes)
# ============================================================================

from config import get_quick_test_config
from dataset import create_dataloaders
from model import create_model
from trainer import Trainer
from utils import set_seed

# Get configuration
config = get_quick_test_config()
config.data.data_root = DATASET_PATH
config.data.save_dir = OUTPUT_PATH
config.experiment_name = "quick_test"

# Set seed for reproducibility
set_seed(config.seed)

print("Creating dataloaders...")
train_loader, val_loader, test_loader, info = create_dataloaders(config)

print("Creating model...")
model = create_model(config)

print("Starting training...")
trainer = Trainer(
    model=model,
    config=config,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    class_weights=info['class_weights'].to(config.device)
)

results = trainer.train()

print("\n" + "="*70)
print("Quick Test Complete!")
print(f"Best Validation Accuracy: {results['best_val_acc']:.2f}%")
print("="*70)


# ============================================================================
# CELL 6: Full Training (50 epochs, ~3-4 hours)
# ============================================================================

from config import get_default_config
from dataset import create_dataloaders
from model import create_model
from trainer import Trainer
from utils import set_seed

# Get configuration
config = get_default_config()
config.data.data_root = DATASET_PATH
config.data.save_dir = OUTPUT_PATH
config.experiment_name = "full_training_40x"

# Optional: Enable WandB logging
config.logging.use_wandb = True
# import wandb
# wandb.login()  # Enter your API key

# Set seed
set_seed(config.seed)

# Create dataloaders
print("Creating dataloaders...")
train_loader, val_loader, test_loader, info = create_dataloaders(config)

# Create model
print("Creating model...")
model = create_model(config)

# Show model info
from utils import count_parameters
params = count_parameters(model)
print(f"\nModel Parameters:")
print(f"  Total: {params['total']:,}")
print(f"  Trainable: {params['trainable']:,}")

# Create trainer
print("\nInitializing trainer...")
trainer = Trainer(
    model=model,
    config=config,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    class_weights=info['class_weights'].to(config.device)
)

# Train!
print("\nStarting full training...")
print("This will take approximately 3-4 hours on T4 GPU")
results = trainer.train()

print("\n" + "="*70)
print("Training Complete!")
print(f"Best Validation Accuracy: {results['best_val_acc']:.2f}%")
print(f"Best Validation AUC: {results['best_val_auc']:.4f}")
print(f"Best Epoch: {results['best_epoch'] + 1}")
print("="*70)


# ============================================================================
# CELL 7: Evaluate on Test Set
# ============================================================================

from metrics import evaluate_model
import torch

# Load best model
best_model_path = f"{OUTPUT_PATH}/{config.experiment_name}/checkpoints/best_model.pth"
checkpoint = torch.load(best_model_path, map_location='cpu', weights_only=False)

if hasattr(model, 'base_model'):
    model.base_model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint['model_state_dict'])

# Evaluate
print("Evaluating on test set...")
test_metrics, y_true, y_pred, y_probs = evaluate_model(
    model, test_loader, config.device, use_uncertainty=True
)

print("\n" + "="*70)
print("Test Set Results:")
print("="*70)
for key, value in test_metrics.items():
    print(f"{key:30s}: {value:.4f}")
print("="*70)


# ============================================================================
# CELL 8: Visualize Results
# ============================================================================

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=['Benign', 'Malignant'],
           yticklabels=['Benign', 'Malignant'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(f'{OUTPUT_PATH}/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.savefig(f'{OUTPUT_PATH}/roc_curve.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ Visualizations saved to {OUTPUT_PATH}")


# ============================================================================
# CELL 9: Download Results
# ============================================================================

# Download checkpoint
from google.colab import files

# Download best model
try:
    files.download(best_model_path)
    print("✅ Model downloaded")
except:
    print("⚠️ Could not download model automatically")
    print(f"Manual download from: {best_model_path}")

# Download visualizations
try:
    files.download(f'{OUTPUT_PATH}/confusion_matrix.png')
    files.download(f'{OUTPUT_PATH}/roc_curve.png')
    print("✅ Visualizations downloaded")
except:
    print("⚠️ Could not download visualizations")


# ============================================================================
# CELL 10: Custom Experiment (Advanced)
# ============================================================================

# Example: Train with different settings

from config import ExperimentConfig

# Create custom config
custom_config = ExperimentConfig()
custom_config.experiment_name = "custom_experiment"
custom_config.data.data_root = DATASET_PATH
custom_config.data.save_dir = OUTPUT_PATH

# Customize settings
custom_config.model.num_gpcn_layers = 6  # More layers
custom_config.model.gpcn_k = 16  # More neighbors
custom_config.model.use_multi_scale = True
custom_config.model.use_uncertainty = True

custom_config.training.num_epochs = 80
custom_config.training.learning_rate = 5e-5
custom_config.data.batch_size = 12  # Adjust for your GPU

# Train with custom config
# (Follow same steps as Cell 6 with custom_config instead of config)


# ============================================================================
# That's it! You're ready to train GPCN-ViT on Google Colab!
# ============================================================================

"""
TIPS:
1. Always check GPU is enabled: Runtime → Change runtime type → GPU
2. For OOM errors: Reduce batch_size in config
3. To resume training: Use trainer.resume_from_checkpoint(path)
4. Monitor in WandB: wandb.login() then check wandb.ai
5. Save work to Drive frequently!

NEXT STEPS:
- Try different magnifications (40X, 100X, 200X, 400X)
- Experiment with hyperparameters
- Cross-magnification evaluation
- K-fold cross-validation

Good luck! 🚀
"""
