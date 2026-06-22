"""
start_colab.py
==============
Professional Google Colab launcher for GPCN-ViT.

HOW TO USE IN COLAB:
  1. Upload ALL .py files from this project to /content/
     (Files panel → Upload, or use Google Drive)
  2. Open a new Colab notebook
  3. In a cell, run:  %run start_colab.py
  The script handles everything from installation to training.

ALTERNATIVELY — cell-by-cell (recommended for debugging):
  Copy and paste each section marked  ── CELL N ──  into a separate Colab cell.

REQUIREMENTS:
  - Colab with GPU runtime (Runtime → Change runtime type → T4 GPU)
  - BreakHis dataset uploaded to your Google Drive
"""

# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 1 ── GPU check + dependency installation
# ══════════════════════════════════════════════════════════════════════════════

import subprocess, sys, os

def _run(cmd, desc=""):
    if desc:
        print(f"  {desc}...", end=" ", flush=True)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if desc:
        print("done" if result.returncode == 0 else f"WARN ({result.returncode})")
    return result


def setup_environment():
    """Install all required packages for GPCN-ViT."""
    print("=" * 70)
    print("  STEP 1/6 — Environment Setup")
    print("=" * 70)

    # GPU info
    gpu_result = _run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
    if gpu_result.returncode == 0 and gpu_result.stdout.strip():
        print(f"  GPU: {gpu_result.stdout.strip()}")
    else:
        print("  WARNING: No GPU detected — training will be very slow on CPU")

    # Detect torch version already installed in Colab
    try:
        import torch
        torch_ver = torch.__version__
        cuda_ver  = torch.version.cuda or "none"
        print(f"  PyTorch: {torch_ver}   CUDA: {cuda_ver}")
    except ImportError:
        torch_ver = "2.1.0"
        print("  PyTorch not found, will install 2.1.0+cu118")
        _run("pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
             "Installing PyTorch")

    # Determine PyG wheel URL from torch version
    torch_short = ".".join(torch_ver.split(".")[:2])  # e.g. "2.1"
    pyg_url = f"https://data.pyg.org/whl/torch-{torch_short}.0+cu118.html"

    packages_basic = [
        ("timm",                     "timm (ViT models)"),
        ("wandb",                    "Weights & Biases"),
        ("opencv-python-headless",   "OpenCV"),
        ("scikit-learn",             "scikit-learn"),
        ("scikit-image",             "scikit-image"),
        ("tqdm",                     "tqdm"),
        ("seaborn",                  "Seaborn"),
        ("staintools",               "StainTools (optional — stain normalisation)"),
    ]

    for pkg, desc in packages_basic:
        _run(f"pip install -q {pkg}", desc)

    # PyTorch Geometric
    print("  Installing PyTorch Geometric...", end=" ", flush=True)
    r1 = _run("pip install -q torch-geometric")
    r2 = _run(f"pip install -q pyg-lib torch-scatter torch-sparse -f {pyg_url}")
    print("done" if (r1.returncode == 0 and r2.returncode == 0) else "WARN (check manually)")

    # Verify critical imports
    failed = []
    for mod in ['torch', 'timm', 'torch_geometric', 'sklearn', 'cv2']:
        try:
            __import__(mod)
        except ImportError:
            failed.append(mod)

    if failed:
        print(f"\n  FAILED imports: {failed}")
        print("  Please install manually before continuing.")
    else:
        print("\n  All critical packages installed successfully!")

    print("=" * 70 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 2 ── Mount Google Drive
# ══════════════════════════════════════════════════════════════════════════════

def mount_drive():
    """Mount Google Drive and configure paths."""
    print("=" * 70)
    print("  STEP 2/6 — Google Drive")
    print("=" * 70)

    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        print("  Google Drive mounted at /content/drive")
    except ModuleNotFoundError:
        print("  Not running in Colab — skipping Drive mount")
        return None, None
    except Exception as exc:
        print(f"  Drive mount failed: {exc}")
        return None, None

    # ── USER CONFIGURATION ────────────────────────────────────────────────────
    # Update these two paths to match your Drive layout.
    DATASET_PATH = "/content/drive/MyDrive/MY RESEARCH/BreaKHis_v1"
    OUTPUT_PATH  = "/content/drive/MyDrive/MY RESEARCH/gpcn_vit_outputs"
    # ─────────────────────────────────────────────────────────────────────────

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    from pathlib import Path
    if Path(DATASET_PATH).exists():
        print(f"  Dataset : {DATASET_PATH}  [FOUND]")
    else:
        print(f"  Dataset : {DATASET_PATH}  [NOT FOUND]")
        print("  Please update DATASET_PATH in mount_drive() to your actual path.")

    print(f"  Outputs : {OUTPUT_PATH}")
    print("=" * 70 + "\n")
    return DATASET_PATH, OUTPUT_PATH


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 3 ── Upload / sync project files
# ══════════════════════════════════════════════════════════════════════════════

REQUIRED_FILES = [
    'config.py', 'model.py', 'gpcn_layer.py', 'knn_builder.py',
    'dataset.py', 'augmentation.py', 'gan_augmentation.py',
    'losses.py', 'metrics.py', 'calibration.py',
    'trainer.py', 'train.py', 'utils.py', 'visualization.py',
]


def upload_files(from_drive_path: str = None):
    """Upload project Python files to /content/."""
    print("=" * 70)
    print("  STEP 3/6 — Project Files")
    print("=" * 70)

    from pathlib import Path

    # Option A: copy from Drive if a code folder is provided
    if from_drive_path:
        drive_code_dir = Path(from_drive_path)
        if drive_code_dir.exists():
            print(f"  Copying from Drive: {drive_code_dir}")
            for f in REQUIRED_FILES:
                src = drive_code_dir / f
                if src.exists():
                    import shutil
                    shutil.copy(src, f'/content/{f}')
                    print(f"    {f}  [copied]")
                else:
                    print(f"    {f}  [MISSING in Drive]")
            print()
            return

    # Option B: check what's already in /content/
    missing = [f for f in REQUIRED_FILES if not (Path('/content') / f).exists()]
    present = [f for f in REQUIRED_FILES if     (Path('/content') / f).exists()]

    print(f"  Files in /content/: {len(present)}/{len(REQUIRED_FILES)}")
    if missing:
        print(f"\n  Missing files: {missing}")
        print("\n  Choose one:")
        print("    A) Upload via browser dialog")
        print("    B) Copy from Google Drive (edit from_drive_path)")
        print("    C) Already in /content/ — continue anyway\n")

        try:
            from google.colab import files
            print("  Launching file picker (select all .py files at once)...")
            uploaded = files.upload()
            print(f"  Uploaded {len(uploaded)} files.")
        except Exception:
            print("  File picker unavailable — please upload manually to /content/")
    else:
        print("  All required files present in /content/")

    print("=" * 70 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 4 ── Verify imports
# ══════════════════════════════════════════════════════════════════════════════

def verify_imports():
    """Verify all project modules import cleanly."""
    print("=" * 70)
    print("  STEP 4/6 — Import Verification")
    print("=" * 70)

    sys.path.insert(0, '/content')
    modules = {
        'config':        'Configuration',
        'model':         'Model (GPCNViT)',
        'gpcn_layer':    'GPCN Layer',
        'knn_builder':   'KNN Builder',
        'dataset':       'Dataset',
        'augmentation':  'Augmentation',
        'losses':        'Loss Functions',
        'metrics':       'Metrics',
        'trainer':       'Trainer',
        'visualization': 'Logger',
        'utils':         'Utilities',
    }

    all_ok = True
    for mod, desc in modules.items():
        try:
            __import__(mod)
            print(f"  {desc:<30} [OK]")
        except Exception as exc:
            print(f"  {desc:<30} [FAIL] — {exc}")
            all_ok = False

    if all_ok:
        print("\n  All modules imported successfully!")
    else:
        print("\n  Fix the failing imports before proceeding.")

    print("=" * 70 + "\n")
    return all_ok


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 5 ── Quick smoke test (5 epochs)
# ══════════════════════════════════════════════════════════════════════════════

def run_quick_test(dataset_path: str, output_path: str, seed: int = 42):
    """5-epoch smoke test — completes in ~10 min on T4."""
    print("=" * 70)
    print("  QUICK TEST — 5 Epochs (smoke test, ~10 min on T4)")
    print("=" * 70)

    sys.path.insert(0, '/content')
    import torch
    from config import get_quick_test_config
    from dataset import create_dataloaders
    from model import create_model
    from trainer import Trainer
    from utils import set_seed, count_parameters

    config = get_quick_test_config()
    config.data.data_root       = dataset_path
    config.data.save_dir        = output_path
    config.experiment_name      = "quick_test"
    config.logging.use_wandb    = False
    config.logging.use_tensorboard = False
    config.device               = 'cuda' if torch.cuda.is_available() else 'cpu'

    set_seed(seed, deterministic=False)

    print(f"  Device     : {config.device.upper()}")
    print(f"  Magnification: {config.data.train_magnification}")
    print(f"  Batch size : {config.data.batch_size}")

    print("\n  Loading dataset...")
    train_loader, val_loader, test_loader, info = create_dataloaders(config)
    print(f"  Train: {info['train_samples']}  Val: {info['val_samples']}  Test: {info['test_samples']}")

    print("\n  Building model...")
    model = create_model(config)
    pc    = count_parameters(model)
    print(f"  Parameters: {pc['total']:,}  Trainable: {pc['trainable']:,}")

    print("\n  Training...")
    device  = torch.device(config.device)
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_weights=info['class_weights'].to(device),
    )

    results = trainer.train()

    print("\n" + "=" * 70)
    print("  QUICK TEST COMPLETE")
    print(f"  Best Val Accuracy : {results['best_val_acc']:.4f}")
    print(f"  Best Val AUC      : {results['best_val_auc']:.4f}")
    print("=" * 70 + "\n")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 6 ── Full training (50 / 100 epochs)
# ══════════════════════════════════════════════════════════════════════════════

def run_full_training(
    dataset_path: str,
    output_path: str,
    magnification: str = '40X',
    num_epochs: int = 50,
    batch_size: int = 16,
    use_wandb: bool = False,
    wandb_project: str = 'gpcn-vit-breakhis',
    seed: int = 42,
    resume_checkpoint: str = None,
):
    """
    Full training run with all features enabled.

    Args:
        dataset_path     : Path to BreaKHis_v1 root
        output_path      : Directory for checkpoints / logs
        magnification    : '40X' | '100X' | '200X' | '400X'
        num_epochs       : Number of epochs (50 default, 100 for full)
        batch_size       : Batch size (16 for ≥16 GB VRAM, 8 for T4)
        use_wandb        : Enable WandB logging
        wandb_project    : WandB project name
        seed             : Random seed
        resume_checkpoint: Path to checkpoint to resume from
    """
    print("=" * 70)
    print(f"  FULL TRAINING — {num_epochs} Epochs @ {magnification}")
    print("=" * 70)

    sys.path.insert(0, '/content')
    import torch
    from config import get_default_config
    from dataset import create_augmented_dataloaders
    from model import create_model
    from trainer import Trainer
    from utils import set_seed, count_parameters, load_checkpoint

    config = get_default_config()
    config.data.data_root            = dataset_path
    config.data.save_dir             = output_path
    config.data.train_magnification  = magnification
    config.data.batch_size           = batch_size
    config.training.num_epochs       = num_epochs
    config.model.use_multi_scale     = True
    config.model.use_uncertainty     = True
    config.training.use_contrastive_loss = True
    config.training.use_focal_loss   = True
    config.logging.use_wandb         = use_wandb
    config.logging.wandb_project     = wandb_project
    config.logging.use_tensorboard   = True
    config.experiment_name           = f"gpcn_vit_{magnification}_{num_epochs}ep"
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.seed   = seed

    if use_wandb:
        try:
            import wandb
            wandb.login()
            print("  WandB login OK")
        except Exception as exc:
            print(f"  WandB login failed: {exc} — disabling WandB")
            config.logging.use_wandb = False

    set_seed(seed, deterministic=False)

    device = torch.device(config.device)
    print(f"\n  Device   : {config.device.upper()}")
    print(f"  Mag      : {magnification}")
    print(f"  Epochs   : {num_epochs}")
    print(f"  Batch    : {batch_size}")
    print(f"  AMP      : {config.training.use_amp}")

    print("\n  Loading dataset (patient-level split)...")
    train_loader, val_loader, test_loader, info = create_augmented_dataloaders(config, device)
    print(f"  Train: {info['train_samples']}  Val: {info['val_samples']}  Test: {info['test_samples']}")

    print("\n  Building model...")
    model = create_model(config).to(device)
    pc    = count_parameters(model)
    print(f"  Parameters: {pc['total']:,}  Trainable: {pc['trainable']:,}")

    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_weights=info['class_weights'].to(device),
    )

    if resume_checkpoint:
        info_ckpt = load_checkpoint(resume_checkpoint, trainer.model, trainer.optimizer, trainer.scheduler)
        trainer.current_epoch = info_ckpt['epoch'] + 1
        trainer.best_val_acc  = info_ckpt.get('best_metric', 0.0)
        print(f"  Resumed from epoch {info_ckpt['epoch']}")

    print("\n  Starting training — check WandB / TensorBoard for live curves")
    print("─" * 70)

    results = trainer.train()

    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print(f"  Best Val Accuracy : {results['best_val_acc']:.4f}")
    print(f"  Best Val AUC      : {results['best_val_auc']:.4f}")
    print(f"  Best Epoch        : {results['best_epoch'] + 1}")
    print("  Test Set:")
    for k, v in sorted(results['test_metrics'].items()):
        print(f"    {k:<35} {v:.4f}")
    print("=" * 70 + "\n")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 7 ── Evaluation and visualisation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_and_plot(
    model,
    test_loader,
    config,
    output_path: str,
    checkpoint_path: str = None,
):
    """Load best checkpoint, evaluate, and save all diagnostic plots."""
    print("=" * 70)
    print("  EVALUATION & VISUALISATION")
    print("=" * 70)

    import torch
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
    from metrics import evaluate_model

    device = torch.device(config.device)

    if checkpoint_path:
        ckpt  = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        base  = model.base_model if hasattr(model, 'base_model') else model
        base.load_state_dict(ckpt['model_state_dict'])
        print(f"  Loaded checkpoint: {checkpoint_path}")

    test_metrics, y_true, y_pred, y_probs = evaluate_model(
        model, test_loader, device,
        use_uncertainty=config.model.use_uncertainty
    )

    print(f"\n{'Metric':<35} {'Value':>10}")
    print("─" * 47)
    for k, v in sorted(test_metrics.items()):
        print(f"  {k:<33} {v:>10.4f}")

    os.makedirs(output_path, exist_ok=True)

    # ── Confusion Matrix ──────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'], ax=ax,
                linewidths=0.5, linecolor='gray')
    ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
    ax.set_title('GPCN-ViT — Confusion Matrix', fontsize=15, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{output_path}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # ── ROC Curve ─────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color='#E84040', lw=3, label=f'GPCN-ViT (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title('ROC Curve', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f'{output_path}/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # ── Precision-Recall Curve ────────────────────────────────────────────
    precision, recall, _ = precision_recall_curve(y_true, y_probs[:, 1])
    pr_auc = auc(recall, precision)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color='#6A5ACD', lw=3, label=f'GPCN-ViT (AUC = {pr_auc:.4f})')
    ax.set_xlabel('Recall', fontsize=13)
    ax.set_ylabel('Precision', fontsize=13)
    ax.set_title('Precision-Recall Curve', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f'{output_path}/pr_curve.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # ── Reliability Diagram ───────────────────────────────────────────────
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    conf, acc = [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_probs[:, 1] >= lo) & (y_probs[:, 1] < hi)
        if mask.sum() > 0:
            conf.append(y_probs[:, 1][mask].mean())
            acc.append((y_true[mask] == 1).mean())
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax.plot(conf, acc, 'o-', color='#E84040', ms=8, lw=2, label='GPCN-ViT')
    ax.set_xlabel('Confidence', fontsize=13)
    ax.set_ylabel('Accuracy', fontsize=13)
    ax.set_title('Reliability Diagram (Calibration)', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f'{output_path}/calibration.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"\n  Plots saved to: {output_path}")
    print(f"    confusion_matrix.png  |  roc_curve.png  |  pr_curve.png  |  calibration.png")
    print("=" * 70 + "\n")

    return test_metrics


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 8 ── Download results from Colab to local machine
# ══════════════════════════════════════════════════════════════════════════════

def download_results(output_path: str, experiment_name: str):
    """Download the best checkpoint and all plots to your local machine."""
    print("=" * 70)
    print("  DOWNLOADING RESULTS")
    print("=" * 70)

    try:
        from google.colab import files
    except ImportError:
        print("  Not in Colab — files are already on your filesystem.")
        return

    from pathlib import Path

    to_download = [
        f"{output_path}/{experiment_name}/checkpoints/best_model.pth",
        f"{output_path}/confusion_matrix.png",
        f"{output_path}/roc_curve.png",
        f"{output_path}/pr_curve.png",
        f"{output_path}/calibration.png",
    ]

    for path in to_download:
        if Path(path).exists():
            try:
                files.download(path)
                print(f"  Downloaded: {Path(path).name}")
            except Exception as exc:
                print(f"  Could not auto-download {Path(path).name}: {exc}")
        else:
            print(f"  Not found (skipped): {path}")

    print("\n  Tip: If auto-download fails, use the Files panel in Colab.")
    print("=" * 70 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 9 ── Multi-magnification experiment
# ══════════════════════════════════════════════════════════════════════════════

def run_all_magnifications(dataset_path: str, output_path: str, num_epochs: int = 50):
    """
    Train and evaluate on all four BreakHis magnifications sequentially.
    Results are aggregated for paper-ready comparison.
    """
    import torch
    from config import get_default_config
    from dataset import create_dataloaders
    from model import create_model
    from trainer import Trainer
    from utils import set_seed, count_parameters
    from metrics import evaluate_model

    magnifications = ['40X', '100X', '200X', '400X']
    all_results    = {}

    for mag in magnifications:
        print(f"\n{'='*70}")
        print(f"  MAGNIFICATION: {mag}")
        print(f"{'='*70}")

        config = get_default_config()
        config.data.data_root           = dataset_path
        config.data.save_dir            = output_path
        config.data.train_magnification = mag
        config.training.num_epochs      = num_epochs
        config.model.use_multi_scale    = True
        config.model.use_uncertainty    = True
        config.experiment_name          = f"gpcn_vit_{mag}"
        config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        config.logging.use_wandb        = False

        set_seed(42, deterministic=False)
        device = torch.device(config.device)

        train_loader, val_loader, test_loader, info = create_dataloaders(config)
        model = create_model(config).to(device)

        trainer = Trainer(
            model=model, config=config,
            train_loader=train_loader, val_loader=val_loader,
            test_loader=test_loader,
            class_weights=info['class_weights'].to(device),
        )
        run_result = trainer.train()

        test_metrics, _, _, _ = evaluate_model(
            model, test_loader, device, use_uncertainty=False
        )
        all_results[mag] = {**run_result, 'test_metrics': test_metrics}

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  MULTI-MAGNIFICATION SUMMARY")
    print(f"{'='*70}")
    header = f"  {'Metric':<25}" + "".join(f"  {m:>8}" for m in magnifications)
    print(header)
    print("  " + "─" * (25 + 12 * len(magnifications)))

    key_metrics = ['accuracy', 'auc_roc', 'sensitivity', 'specificity', 'ece', 'f1_score']
    for metric in key_metrics:
        row = f"  {metric:<25}"
        for mag in magnifications:
            val = all_results[mag]['test_metrics'].get(metric, float('nan'))
            row += f"  {val:>8.4f}"
        print(row)

    print(f"{'='*70}\n")
    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 10 ── Main entry point (run everything end-to-end)
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """
    End-to-end pipeline: setup → files → verify → train → evaluate → download.
    Designed to run as %run start_colab.py in a Colab notebook.
    """
    print()
    print("=" * 70)
    print("  GPCN-ViT  |  Breast Cancer Histopathology Classification")
    print("  Graph Patch Correlation Network + Vision Transformer")
    print("=" * 70)
    print()

    # ── Step 1: Environment ───────────────────────────────────────────────
    setup_environment()

    # ── Step 2: Drive ─────────────────────────────────────────────────────
    dataset_path, output_path = mount_drive()

    # Fallback paths for non-Colab execution
    if dataset_path is None:
        dataset_path = os.environ.get('BREAKHIS_PATH', '/data/BreaKHis_v1')
        output_path  = os.environ.get('OUTPUT_PATH',   '/tmp/gpcn_vit_outputs')
        os.makedirs(output_path, exist_ok=True)

    # ── Step 3: Files ─────────────────────────────────────────────────────
    # To copy from Drive, set this to your Drive code folder:
    DRIVE_CODE_PATH = None  # e.g. "/content/drive/MyDrive/MY RESEARCH/gpcn_code"
    upload_files(from_drive_path=DRIVE_CODE_PATH)

    # ── Step 4: Imports ───────────────────────────────────────────────────
    ok = verify_imports()
    if not ok:
        print("Fix imports before running training.")
        return

    # ── Step 5: Quick test (uncomment to run) ─────────────────────────────
    # run_quick_test(dataset_path, output_path)

    # ── Step 6: Full training ─────────────────────────────────────────────
    results = run_full_training(
        dataset_path   = dataset_path,
        output_path    = output_path,
        magnification  = '40X',
        num_epochs     = 50,
        batch_size     = 16,
        use_wandb      = False,   # set True + wandb.login() for live curves
        seed           = 42,
    )

    # ── Step 7: Download ──────────────────────────────────────────────────
    download_results(output_path, experiment_name='gpcn_vit_40X_50ep')

    print("Pipeline complete. Check your Drive for saved checkpoints and plots.")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    main()
