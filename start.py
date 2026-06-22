"""
start.py
========
Professional local launcher for GPCN-ViT Breast Cancer Classification.

Performs full pre-flight environment checks, then launches training with the
configuration you select. Run with --help for all options.

Usage:
    python start.py                                      # interactive guided setup
    python start.py --data-root /path/to/BreaKHis_v1    # specify dataset
    python start.py --mode quick                         # 5-epoch smoke test
    python start.py --mode full --magnification 400X    # full run on 400X
    python start.py --check-only                         # pre-flight only, no training
"""

import sys
import os
import platform
import subprocess
import argparse
import time
import logging
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Terminal colours (graceful degradation on Windows)
# ─────────────────────────────────────────────────────────────────────────────

_USE_COLOR = sys.stdout.isatty() and platform.system() != 'Windows'

def _c(text, code):
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text

def green(t):  return _c(t, '32')
def red(t):    return _c(t, '31')
def yellow(t): return _c(t, '33')
def cyan(t):   return _c(t, '36')
def bold(t):   return _c(t, '1')


# ─────────────────────────────────────────────────────────────────────────────
# Banner
# ─────────────────────────────────────────────────────────────────────────────

BANNER = r"""
  ██████╗ ██████╗  ██████╗███╗   ██╗      ██╗   ██╗██╗████████╗
 ██╔════╝ ██╔══██╗██╔════╝████╗  ██║      ██║   ██║██║╚══██╔══╝
 ██║  ███╗██████╔╝██║     ██╔██╗ ██║█████╗██║   ██║██║   ██║
 ██║   ██║██╔═══╝ ██║     ██║╚██╗██║╚════╝╚██╗ ██╔╝██║   ██║
 ╚██████╔╝██║     ╚██████╗██║ ╚████║       ╚████╔╝ ██║   ██║
  ╚═════╝ ╚═╝      ╚═════╝╚═╝  ╚═══╝        ╚═══╝  ╚═╝   ╚═╝

  Graph Patch Correlation Network + Vision Transformer
  Breast Cancer Histopathology Classification (BreakHis)
"""


# ─────────────────────────────────────────────────────────────────────────────
# Pre-flight checks
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_PACKAGES = [
    ('torch',             'PyTorch'),
    ('torchvision',       'TorchVision'),
    ('timm',              'timm (ViT models)'),
    ('torch_geometric',   'PyTorch Geometric'),
    ('sklearn',           'scikit-learn'),
    ('PIL',               'Pillow'),
    ('cv2',               'OpenCV'),
    ('numpy',             'NumPy'),
    ('pandas',            'pandas'),
    ('matplotlib',        'Matplotlib'),
    ('seaborn',           'Seaborn'),
    ('scipy',             'SciPy'),
    ('tqdm',              'tqdm'),
]

OPTIONAL_PACKAGES = [
    ('wandb',             'Weights & Biases'),
    ('tensorboard',       'TensorBoard'),
]


def _check_python():
    major, minor = sys.version_info[:2]
    ok = (major == 3 and minor >= 9)
    status = green('OK') if ok else red('FAIL')
    print(f"  Python {major}.{minor}   [{status}]")
    if not ok:
        print(red(f"    Python >= 3.9 required, found {major}.{minor}"))
    return ok


def _check_torch_cuda():
    try:
        import torch
        print(f"  PyTorch {torch.__version__}   [{green('OK')}]")

        if torch.cuda.is_available():
            dev_name = torch.cuda.get_device_name(0)
            vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  CUDA {torch.version.cuda} — {dev_name} ({vram_gb:.1f} GB)   [{green('OK')}]")
            if vram_gb < 8:
                print(yellow(f"    Warning: {vram_gb:.1f} GB VRAM may be tight — consider batch_size=8"))
            return True, True
        else:
            print(f"  CUDA   [{yellow('NOT AVAILABLE')}] — CPU training will be very slow")
            return True, False
    except ImportError:
        print(f"  PyTorch   [{red('NOT FOUND')}]")
        print(red("    Install: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"))
        return False, False


def _check_packages():
    all_ok = True
    for mod, name in REQUIRED_PACKAGES:
        try:
            __import__(mod)
            print(f"  {name:<30} [{green('OK')}]")
        except ImportError:
            print(f"  {name:<30} [{red('MISSING')}]")
            all_ok = False

    for mod, name in OPTIONAL_PACKAGES:
        try:
            __import__(mod)
            print(f"  {name:<30} [{green('OK')} (optional)]")
        except ImportError:
            print(f"  {name:<30} [{yellow('not installed')} (optional)]")

    return all_ok


def _check_dataset(data_root: str):
    root = Path(data_root)
    if not root.exists():
        print(f"  Dataset [{red('NOT FOUND')}]: {data_root}")
        return False

    pattern = "histology_slides/breast/*/SOB/*/*/40X/*.png"
    images  = list(root.glob(pattern))
    if len(images) == 0:
        print(f"  Dataset [{red('EMPTY')}]: no images matched pattern inside {data_root}")
        print(f"    Expected: {root / pattern}")
        return False

    print(f"  Dataset [{green('OK')}]: {len(images):,} images @ 40X found in {data_root}")
    return True


def _check_project_files():
    required = [
        'config.py', 'model.py', 'gpcn_layer.py', 'knn_builder.py',
        'dataset.py', 'augmentation.py', 'losses.py', 'metrics.py',
        'trainer.py', 'train.py', 'utils.py', 'visualization.py',
    ]
    root = Path(__file__).parent
    all_ok = True
    for f in required:
        path = root / f
        if path.exists():
            print(f"  {f:<30} [{green('OK')}]")
        else:
            print(f"  {f:<30} [{red('MISSING')}]")
            all_ok = False
    return all_ok


def run_preflight(data_root: str, verbose: bool = True):
    """Run all pre-flight checks. Returns True if safe to proceed."""
    sep = "─" * 60

    print(f"\n{bold('PRE-FLIGHT CHECKS')}")
    print(sep)

    print(f"\n{cyan('[ Python & CUDA ]')}")
    py_ok = _check_python()
    torch_ok, cuda_ok = _check_torch_cuda()

    print(f"\n{cyan('[ Project Files ]')}")
    files_ok = _check_project_files()

    print(f"\n{cyan('[ Python Packages ]')}")
    pkgs_ok = _check_packages()

    print(f"\n{cyan('[ Dataset ]')}")
    data_ok = _check_dataset(data_root)

    all_ok = py_ok and torch_ok and files_ok and pkgs_ok and data_ok

    print(f"\n{sep}")
    if all_ok:
        print(green("  All checks passed — ready to train!"))
    else:
        print(red("  One or more checks failed — see above."))
        if not data_ok:
            print(yellow("  Tip: pass --data-root /path/to/BreaKHis_v1"))
        if not pkgs_ok:
            print(yellow("  Tip: pip install -r requirements.txt"))
    print(sep)

    return all_ok, cuda_ok


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='GPCN-ViT local training launcher',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start.py --data-root /data/BreaKHis_v1
  python start.py --mode quick --data-root /data/BreaKHis_v1 --no-wandb
  python start.py --mode full  --magnification 400X --epochs 100
  python start.py --check-only --data-root /data/BreaKHis_v1
  python start.py --test-only  --checkpoint runs/best_model.pth
        """
    )

    # Mode
    parser.add_argument('--mode', choices=['quick', 'default', 'full'], default='default',
                        help='Training preset: quick (5 ep) | default (50 ep) | full (100 ep)')
    parser.add_argument('--check-only', action='store_true',
                        help='Run pre-flight checks only, then exit')
    parser.add_argument('--test-only',  action='store_true',
                        help='Evaluate a checkpoint on the test set (requires --checkpoint)')

    # Data
    parser.add_argument('--data-root', type=str, default=None,
                        help='Path to BreaKHis_v1 root directory')
    parser.add_argument('--magnification', type=str, default='40X',
                        choices=['40X', '100X', '200X', '400X'],
                        help='Magnification level to train on (default: 40X)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size (default depends on mode)')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory to save outputs (default: ./outputs)')

    # Model
    parser.add_argument('--gpcn-layers', type=int, default=None,
                        help='Number of GPCN adapter layers (default: 3)')
    parser.add_argument('--no-uncertainty', action='store_true',
                        help='Disable Monte Carlo Dropout uncertainty estimation')
    parser.add_argument('--no-multiscale',  action='store_true',
                        help='Disable graph pyramid (single GPCN layer)')

    # Training
    parser.add_argument('--epochs',  type=int,   default=None, help='Override training epochs')
    parser.add_argument('--lr',      type=float, default=None, help='Override learning rate')
    parser.add_argument('--no-amp',  action='store_true',      help='Disable automatic mixed precision')
    parser.add_argument('--seed',    type=int,   default=42,   help='Random seed (default: 42)')

    # Logging
    parser.add_argument('--wandb-project', type=str, default=None,
                        help='WandB project name (enables WandB logging)')
    parser.add_argument('--no-wandb',      action='store_true', help='Disable WandB')
    parser.add_argument('--experiment',    type=str, default=None,
                        help='Experiment name (default: gpcn_vit_{mode}_{magnification})')

    # Misc
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint path for --test-only or to resume training')
    parser.add_argument('--skip-checks', action='store_true',
                        help='Skip pre-flight checks (not recommended)')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    parser.add_argument('--all-magnifications', action='store_true',
                        help='Train on all magnifications (40X→100X→200X→400X) sequentially, '
                             'skipping any already completed')

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Interactive dataset discovery
# ─────────────────────────────────────────────────────────────────────────────

COMMON_DATASET_LOCATIONS = [
    Path.home() / 'Downloads' / 'BreaKHis_v1',
    Path.home() / 'data'     / 'BreaKHis_v1',
    Path.home() / 'datasets' / 'BreaKHis_v1',
    Path('/data')            / 'BreaKHis_v1',
    Path('/content')         / 'BreaKHis_v1',
    Path('/mnt/data')        / 'BreaKHis_v1',
]


def discover_dataset():
    """Try to auto-detect the BreakHis dataset location."""
    for loc in COMMON_DATASET_LOCATIONS:
        if loc.exists():
            pattern = "histology_slides/breast/*/SOB/*/*/40X/*.png"
            if list(loc.glob(pattern)):
                print(yellow(f"  Auto-detected dataset at: {loc}"))
                return str(loc)
    return None


def prompt_dataset():
    """Interactively ask the user for the dataset path."""
    detected = discover_dataset()
    if detected:
        answer = input(f"\n  Use auto-detected path? [{detected}] (y/n): ").strip().lower()
        if answer in ('', 'y', 'yes'):
            return detected

    while True:
        path = input("\n  Enter path to BreaKHis_v1: ").strip()
        if not path:
            continue
        path = os.path.expanduser(path)
        if Path(path).exists():
            return path
        print(red(f"  Path not found: {path}"))


# ─────────────────────────────────────────────────────────────────────────────
# Build config from args
# ─────────────────────────────────────────────────────────────────────────────

def build_config(args, cuda_ok: bool):
    sys.path.insert(0, str(Path(__file__).parent))
    from config import get_default_config, get_quick_test_config, get_full_config

    if args.mode == 'quick':
        config = get_quick_test_config()
    elif args.mode == 'full':
        config = get_full_config()
    else:
        config = get_default_config()

    # Data
    config.data.data_root       = args.data_root
    config.data.train_magnification = args.magnification
    config.data.save_dir        = args.save_dir or str(Path.cwd() / 'outputs')
    if args.batch_size:
        config.data.batch_size  = args.batch_size

    # Model
    if args.gpcn_layers:
        config.model.num_gpcn_layers  = args.gpcn_layers
    if args.no_uncertainty:
        config.model.use_uncertainty  = False
    if args.no_multiscale:
        config.model.use_multi_scale  = False

    # Training
    if args.epochs:
        config.training.num_epochs    = args.epochs
    if args.lr:
        config.training.learning_rate = args.lr
    if args.no_amp or not cuda_ok:
        config.training.use_amp       = False

    # Logging
    if args.no_wandb:
        config.logging.use_wandb      = False
    if args.wandb_project:
        config.logging.use_wandb      = True
        config.logging.wandb_project  = args.wandb_project

    # Experiment name
    exp_name = args.experiment or f"gpcn_vit_{args.mode}_{args.magnification}"
    config.experiment_name = exp_name

    # Seed / device
    config.seed   = args.seed
    config.device = 'cuda' if cuda_ok else 'cpu'

    return config


# ─────────────────────────────────────────────────────────────────────────────
# Per-magnification training helper
# ─────────────────────────────────────────────────────────────────────────────

def _run_one_magnification(args, config, cuda_ok):
    """
    Train for one magnification level described by `config`.

    Auto-resumes from `last_checkpoint.pth` if present (no need to pass
    --checkpoint when restarting after a crash).  An explicit --checkpoint arg
    still takes priority for the single-magnification flow.

    Returns the results dict from Trainer.train().
    """
    import torch
    from utils import (set_seed, setup_logging, create_experiment_directory,
                       count_parameters, load_checkpoint)
    from dataset import create_augmented_dataloaders
    from model import create_model
    from trainer import Trainer

    device = torch.device(config.device)

    exp_dir  = create_experiment_directory(config.data.save_dir, config.experiment_name)
    log_file = exp_dir / 'logs' / 'training.log'
    setup_logging(str(log_file))

    set_seed(config.seed, config.deterministic)

    # ── Dataset ───────────────────────────────────────────────────────────────
    print(f"\n{bold('LOADING DATASET')}  [{config.data.train_magnification}]  "
          f"(patient-level split — no leakage)")
    train_loader, val_loader, test_loader, data_info = create_augmented_dataloaders(config, device)
    n_train, n_val, n_test = data_info['train_samples'], data_info['val_samples'], data_info['test_samples']
    print(green(f"  Train: {n_train}  Val: {n_val}  Test: {n_test}"))

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\n{bold('BUILDING MODEL')}")
    model = create_model(config)
    model = model.to(device)
    pc = count_parameters(model)
    print(green(f"  Parameters — Total: {pc['total']:,}  Trainable: {pc['trainable']:,}"))

    # Smoke-test forward pass
    with torch.no_grad():
        batch  = next(iter(train_loader))
        images = batch[0][:2].to(device)
        out    = model.base_model(images) if hasattr(model, 'base_model') else model(images)
    print(green(f"  Forward pass OK — output shape: {tuple(out.shape)}"))

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_weights=data_info['class_weights'].to(device),
    )

    # ── Checkpoint resume ─────────────────────────────────────────────────────
    # Priority: explicit --checkpoint arg  >  auto-detected last_checkpoint.pth
    ckpt_path = getattr(args, 'checkpoint', None)
    if not ckpt_path:
        auto_last = (Path(config.data.save_dir) / config.experiment_name
                     / 'checkpoints' / 'last_checkpoint.pth')
        if auto_last.exists():
            ckpt_path = str(auto_last)
            print(yellow(f"  Auto-resuming from: {auto_last}"))

    if ckpt_path and Path(ckpt_path).exists():
        info = load_checkpoint(ckpt_path, trainer.model, trainer.optimizer, trainer.scheduler)
        trainer.current_epoch = info['epoch'] + 1
        trainer.best_val_acc  = info.get('best_metric', 0.0)
        trainer.global_step   = info.get('global_step', 0)
        trainer.best_val_auc  = info.get('best_val_auc', 0.0)
        trainer.best_epoch    = info.get('best_epoch', 0)
        print(green(f"  Resumed from epoch {info['epoch']} "
                    f"(best acc: {trainer.best_val_acc:.4f})"))

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\n{bold('STARTING TRAINING')}  → logs in {log_file}")
    print("─" * 60)
    results = trainer.train()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(cyan(BANNER))

    args = parse_args()

    # Resolve dataset path
    if args.data_root is None and not args.check_only:
        if args.test_only and args.checkpoint:
            args.data_root = prompt_dataset()
        else:
            args.data_root = prompt_dataset()
    elif args.data_root:
        args.data_root = os.path.expanduser(args.data_root)

    # Pre-flight
    if not args.skip_checks:
        data_root_for_check = args.data_root or '.'
        all_ok, cuda_ok = run_preflight(data_root_for_check, verbose=args.verbose)
        if args.check_only:
            sys.exit(0 if all_ok else 1)
        if not all_ok:
            answer = input(yellow("\n  Checks failed. Proceed anyway? (y/N): ")).strip().lower()
            if answer not in ('y', 'yes'):
                sys.exit(1)
    else:
        import torch
        cuda_ok = torch.cuda.is_available()

    # Build config
    config = build_config(args, cuda_ok)

    print(f"\n{bold('EXPERIMENT CONFIGURATION')}")
    print("─" * 60)
    print(f"  Name          : {config.experiment_name}")
    print(f"  Mode          : {args.mode}")
    print(f"  Device        : {config.device.upper()}")
    print(f"  Magnification : {config.data.train_magnification}")
    print(f"  Epochs        : {config.training.num_epochs}")
    print(f"  Batch size    : {config.data.batch_size}")
    print(f"  LR            : {config.training.learning_rate}")
    print(f"  GPCN layers   : {config.model.num_gpcn_layers}")
    print(f"  Multi-scale   : {config.model.use_multi_scale}")
    print(f"  Uncertainty   : {config.model.use_uncertainty}")
    print(f"  AMP           : {config.training.use_amp}")
    print(f"  WandB         : {config.logging.use_wandb}")
    print(f"  Seed          : {config.seed}")
    print(f"  Save dir      : {config.data.save_dir}")
    print("─" * 60)

    # Inject path so all modules are importable
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import setup_logging, set_seed, create_experiment_directory
    from dataset import create_dataloaders
    from model import create_model
    from metrics import evaluate_model
    import torch

    # ── Test-only mode ─────────────────────────────────────────────────────
    if args.test_only:
        if not args.checkpoint:
            print(red("  --test-only requires --checkpoint <path>"))
            sys.exit(1)

        exp_dir  = create_experiment_directory(config.data.save_dir, config.experiment_name)
        log_file = exp_dir / 'logs' / 'training.log'
        setup_logging(str(log_file))
        set_seed(config.seed, config.deterministic)

        print(f"\n{bold('TEST EVALUATION')}")
        device = torch.device(config.device)
        model  = create_model(config)
        model  = model.to(device)

        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        base = model.base_model if hasattr(model, 'base_model') else model
        base.load_state_dict(checkpoint['model_state_dict'])
        print(green(f"  Checkpoint loaded from: {args.checkpoint}"))

        _, val_loader, test_loader, _ = create_dataloaders(config)
        test_metrics, y_true, y_pred, y_probs = evaluate_model(
            model, test_loader, device, use_uncertainty=config.model.use_uncertainty
        )

        print(f"\n{'Metric':<35} {'Value':>10}")
        print("─" * 47)
        for k, v in sorted(test_metrics.items()):
            print(f"  {k:<33} {v:>10.4f}")
        sys.exit(0)

    # ── All-magnifications mode ────────────────────────────────────────────
    if args.all_magnifications:
        import json
        import argparse as _ap
        ALL_MAGS = ['40X', '100X', '200X', '400X']

        base_exp  = args.experiment or f"gpcn_vit_{args.mode}_all"
        save_root = Path(args.save_dir or 'outputs')
        prog_dir  = save_root / base_exp
        prog_dir.mkdir(parents=True, exist_ok=True)
        prog_file = prog_dir / 'training_progress.json'

        if prog_file.exists():
            with open(prog_file) as _f:
                progress = json.load(_f)
            done = list(progress.get('completed', {}).keys())
            print(yellow(f"\n  Progress file found — already completed: {done}"))
        else:
            progress = {'completed': {}}

        all_results = {}
        t0_all = time.time()

        for mag in ALL_MAGS:
            if mag in progress['completed']:
                print(yellow(f"\n  [{mag}] Already completed — skipping"))
                all_results[mag] = progress['completed'][mag]
                continue

            print(f"\n{bold('=' * 60)}")
            print(f"{bold(f'  MAGNIFICATION: {mag}')}")
            print(f"{bold('=' * 60)}")

            mag_config = build_config(args, cuda_ok)
            mag_config.data.train_magnification = mag
            mag_config.experiment_name = f"{base_exp}_{mag}"

            # Each magnification auto-detects its own last_checkpoint.pth;
            # the explicit --checkpoint flag is intentionally cleared here.
            mag_args = _ap.Namespace(**vars(args))
            mag_args.checkpoint = None

            t0_mag = time.time()
            try:
                results = _run_one_magnification(mag_args, mag_config, cuda_ok)
            except KeyboardInterrupt:
                print(yellow(f"\n  [{mag}] Interrupted — progress saved; re-run to continue."))
                with open(prog_file, 'w') as _f:
                    json.dump(progress, _f, indent=2, default=str)
                sys.exit(0)

            elapsed_mag = time.time() - t0_mag
            h, r = divmod(int(elapsed_mag), 3600)
            m, s = divmod(r, 60)

            # Serialise results (numpy floats → plain float)
            stored = {k: (float(v) if isinstance(v, (float, int)) else v)
                      for k, v in results.items() if k != 'test_metrics'}
            stored['test_metrics'] = {k: float(v) for k, v in results.get('test_metrics', {}).items()}
            progress['completed'][mag] = stored

            with open(prog_file, 'w') as _f:
                json.dump(progress, _f, indent=2, default=str)

            all_results[mag] = results
            print(green(f"\n  [{mag}] Done in {h:02d}h {m:02d}m {s:02d}s — "
                        f"best acc: {results['best_val_acc']:.4f}  "
                        f"best AUC: {results['best_val_auc']:.4f}"))

        elapsed_all = time.time() - t0_all
        h, r = divmod(int(elapsed_all), 3600)
        m, s = divmod(r, 60)

        print(f"\n{bold('ALL MAGNIFICATIONS COMPLETE')}  (total {h:02d}h {m:02d}m {s:02d}s)")
        print("─" * 60)
        for mag in ALL_MAGS:
            if mag in all_results:
                r = all_results[mag]
                print(f"  {mag}  best_acc={r['best_val_acc']:.4f}  "
                      f"best_auc={r['best_val_auc']:.4f}  "
                      f"best_epoch={r['best_epoch'] + 1}")
        print("─" * 60)
        print(green(f"  Progress file: {prog_file}"))
        return

    # ── Single-magnification training mode ────────────────────────────────
    t0 = time.time()
    try:
        results = _run_one_magnification(args, config, cuda_ok)
    except KeyboardInterrupt:
        print(yellow("\n  Interrupted by user — partial results saved."))
        sys.exit(0)

    elapsed = time.time() - t0
    hrs, rem = divmod(int(elapsed), 3600)
    mins, secs = divmod(rem, 60)

    save_dir_path = Path(config.data.save_dir) / config.experiment_name / 'checkpoints'
    exp_dir       = Path(config.data.save_dir) / config.experiment_name

    print(f"\n{bold('RESULTS')}")
    print("─" * 60)
    print(f"  Best val accuracy : {results['best_val_acc']:.4f}")
    print(f"  Best val AUC      : {results['best_val_auc']:.4f}")
    print(f"  Best epoch        : {results['best_epoch'] + 1}")
    print(f"  Total time        : {hrs:02d}h {mins:02d}m {secs:02d}s")
    print(f"\n  Test Set:")
    for k, v in sorted(results['test_metrics'].items()):
        print(f"    {k:<33} {v:.4f}")
    print("─" * 60)
    print(green(f"\n  Checkpoints saved to: {save_dir_path}"))
    print(green(f"  Logs saved to:        {exp_dir / 'logs'}"))


if __name__ == '__main__':
    main()
