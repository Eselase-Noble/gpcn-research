"""
bach.py — BACH (ICIAR 2018) MULTI-CLASS support for GPCN-ViT.

WHAT BACH IS
------------
The ICIAR-2018 "BreAst Cancer Histology" (BACH) challenge, Part A, is 400 H&E
microscopy images (~2048x1536 px), 100 per class, in FOUR classes:
    Normal, Benign, InSitu (in-situ carcinoma), Invasive (invasive carcinoma).
Unlike BreaKHis, BACH images are captured at a SINGLE magnification (no
40X/100X/200X/400X levels), so the multi-magnification fusion model does not
apply. This module trains a single-image GPCN-ViT for the native MULTI-CLASS
task (4 classes, clinically ordered Normal < Benign < InSitu < Invasive).

LABELS (clinically ordered, severity increasing):
    Normal -> 0,  Benign -> 1,  InSitu -> 2,  Invasive -> 3
The class set is detected from the folder names, so a dataset variant with a 5th
class would extend automatically; num_classes is set from what is found.

SPLITTING NOTE
--------------
BACH Part A ships without a clean patient->image mapping (the 400 images are
treated as independent cases), so a label-STRATIFIED image-level split is the
standard, accepted protocol for BACH (no patient grouping to leak). This differs
from the BreaKHis pipeline (which is patient-level) and is stated for transparency.

Usage
-----
    from config import get_default_config
    from bach import train_bach
    cfg = get_default_config(); cfg.data.save_dir = OUTPUT_PATH
    res = train_bach(cfg, bach_root=BACH_PATH)     # 4-class training + metrics
"""

import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from augmentation import get_train_transform, get_val_transform

logger = logging.getLogger(__name__)

# Canonical, clinically-ordered class names (severity increasing). Detected
# folders are mapped to this order; anything unrecognised is appended after.
_CANONICAL = ['normal', 'benign', 'insitu', 'invasive']
_IMG_EXTS = ('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp')


def _norm(name: str) -> str:
    return name.lower().replace(' ', '').replace('_', '').replace('-', '').strip()


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
def build_bach_index(bach_root: str) -> Tuple[List[Tuple[str, int]], List[str]]:
    """Scan BACH Part A and return ([(image_path, class_idx), ...], class_names).

    Looks for class sub-directories anywhere under bach_root (handles both
    '<root>/Photos/<Class>/*.tif' and '<root>/<Class>/*.tif' layouts). Class
    indices follow the clinical order Normal<Benign<InSitu<Invasive.
    """
    root = Path(bach_root)
    if not root.exists():
        raise FileNotFoundError(f"BACH root not found: {bach_root}")

    # Collect images per (normalised) class folder name.
    by_class: Dict[str, List[str]] = defaultdict(list)
    for sub in sorted(root.rglob('*')):
        if not sub.is_dir():
            continue
        key = _norm(sub.name)
        if key not in _CANONICAL and not any(c in key for c in _CANONICAL):
            continue
        # Normalise partial matches (e.g. 'insitucarcinoma' -> 'insitu')
        for c in _CANONICAL:
            if c in key:
                key = c
                break
        imgs = [str(p) for p in sorted(sub.iterdir())
                if p.suffix.lower() in _IMG_EXTS]
        by_class[key].extend(imgs)

    if not by_class:
        raise RuntimeError(
            f"No BACH class folders found under {bach_root}. Expected folders "
            f"named like Normal/Benign/InSitu/Invasive containing images.")

    # Order classes: canonical first, then any extras alphabetically.
    found = list(by_class.keys())
    ordered = [c for c in _CANONICAL if c in found] + sorted(c for c in found if c not in _CANONICAL)
    class_to_idx = {c: i for i, c in enumerate(ordered)}

    index: List[Tuple[str, int]] = []
    for c in ordered:
        for p in by_class[c]:
            index.append((p, class_to_idx[c]))

    counts = {c: len(by_class[c]) for c in ordered}
    logger.info(f"[bach] {len(index)} images, {len(ordered)} classes {counts}")
    return index, ordered


class BACHDataset(Dataset):
    """Single-image BACH dataset: yields (image_tensor, class_idx)."""

    def __init__(self, items: List[Tuple[str, int]], transform=None):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

    def labels(self) -> List[int]:
        return [l for _, l in self.items]

    def class_weights(self, num_classes: int) -> torch.Tensor:
        counts = np.bincount(self.labels(), minlength=num_classes)
        w = len(self.items) / (num_classes * np.maximum(counts, 1))
        return torch.FloatTensor(w)


def _stratified_split(items, num_classes, test_size, val_size, seed):
    """Label-stratified image-level split (BACH has no patient grouping)."""
    rng = random.Random(seed)
    by_label: Dict[int, List] = defaultdict(list)
    for it in items:
        by_label[it[1]].append(it)
    train, val, test = [], [], []
    for lab in range(num_classes):
        group = by_label[lab]
        rng.shuffle(group)
        n = len(group)
        n_test = int(round(n * test_size))
        n_val = int(round(n * val_size))
        test += group[:n_test]
        val += group[n_test:n_test + n_val]
        train += group[n_test + n_val:]
    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    return train, val, test


def create_bach_dataloaders(config, bach_root: str
                            ) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    index, class_names = build_bach_index(bach_root)
    num_classes = len(class_names)
    train_items, val_items, test_items = _stratified_split(
        index, num_classes, config.data.test_size, config.data.val_size, config.seed)

    train_ds = BACHDataset(train_items, get_train_transform(config))
    val_ds = BACHDataset(val_items, get_val_transform(config))
    test_ds = BACHDataset(test_items, get_val_transform(config))

    common = dict(num_workers=config.data.num_workers, pin_memory=config.data.pin_memory)
    train_loader = DataLoader(train_ds, batch_size=config.data.batch_size,
                              shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_ds, batch_size=config.data.batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=config.data.batch_size, shuffle=False, **common)

    info = {
        'num_classes': num_classes,
        'class_names': class_names,
        'class_weights': train_ds.class_weights(num_classes),
        'train': len(train_ds), 'val': len(val_ds), 'test': len(test_ds),
    }
    logger.info(f"[bach] split {info}")
    return train_loader, val_loader, test_loader, info


# ----------------------------------------------------------------------------
# Model construction (single-image GPCN-ViT with K-class head)
# ----------------------------------------------------------------------------
def build_single_gpcnvit(config, num_classes: int) -> nn.Module:
    from model import GPCNViT
    return GPCNViT(
        num_classes=num_classes,
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


# ----------------------------------------------------------------------------
# Multi-class evaluation
# ----------------------------------------------------------------------------
def _multiclass_ece(y_true, y_probs, n_bins=15):
    """Top-1 confidence ECE/MCE for multi-class predictions."""
    conf = y_probs.max(axis=1)
    pred = y_probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece, mce = 0.0, 0.0
    for i in range(n_bins):
        m = (conf >= bins[i]) & (conf < bins[i + 1])
        if m.mean() > 0:
            gap = abs(conf[m].mean() - correct[m].mean())
            ece += gap * m.mean()
            mce = max(mce, gap)
    return float(ece), float(mce)


@torch.no_grad()
def evaluate_bach(model, loader, device, num_classes: int,
                  class_names: Optional[List[str]] = None) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """Multi-class metrics: accuracy, balanced accuracy, weighted/macro P/R/F1,
    MCC, macro & weighted one-vs-rest AUC, top-1 ECE/MCE, and per-class F1."""
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                  precision_recall_fscore_support, matthews_corrcoef,
                                  roc_auc_score, confusion_matrix, f1_score)
    model.eval()
    all_true, all_probs = [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        if isinstance(logits, tuple):
            logits = logits[0]
        probs = F.softmax(logits, dim=-1)
        all_true.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_true)
    y_probs = np.array(all_probs)
    y_pred = y_probs.argmax(axis=1)

    metrics: Dict[str, float] = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    pw, rw, fw, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    metrics.update({'precision_weighted': pw, 'recall_weighted': rw, 'f1_weighted': fw,
                    'precision_macro': pm, 'recall_macro': rm, 'f1_macro': fm})
    try:
        labels_present = list(range(num_classes))
        metrics['auc_macro_ovr'] = roc_auc_score(y_true, y_probs, multi_class='ovr',
                                                  average='macro', labels=labels_present)
        metrics['auc_weighted_ovr'] = roc_auc_score(y_true, y_probs, multi_class='ovr',
                                                     average='weighted', labels=labels_present)
    except Exception:
        metrics['auc_macro_ovr'] = metrics['auc_weighted_ovr'] = 0.0
    ece, mce = _multiclass_ece(y_true, y_probs)
    metrics['ece'], metrics['mce'] = ece, mce

    # Per-class F1 for the table.
    per_f1 = f1_score(y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0)
    names = class_names or [str(i) for i in range(num_classes)]
    for name, f in zip(names, per_f1):
        metrics[f'f1_{name}'] = float(f)
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred, labels=list(range(num_classes))).tolist()
    return metrics, y_true, y_probs


# ----------------------------------------------------------------------------
# Train on BACH (multi-class)
# ----------------------------------------------------------------------------
def train_bach(config, bach_root: Optional[str] = None, loaders=None) -> Dict:
    """Train a single-magnification GPCN-ViT for the BACH multi-class task.

    Class-weighted focal loss (works for K classes), discriminative LR,
    warmup->cosine, EMA, AMP, accuracy/AUC model selection, auto-resume.
    Returns {'best_val_score', 'best_epoch', 'test_metrics', 'calibration', 'info'}.

    `loaders` lets a caller (e.g. k-fold CV) pass a pre-built
    (train_loader, val_loader, test_loader, info) tuple instead of the standard
    stratified split built from `bach_root`. Exactly one of the two is required.
    """
    from utils import ModelEMA, GradientClipping, set_seed
    from losses import FocalLoss, LabelSmoothingCrossEntropy
    import torch.optim as optim

    set_seed(config.seed, config.deterministic)
    device = torch.device(config.device)

    if loaders is not None:
        train_loader, val_loader, test_loader, info = loaders
    elif bach_root is not None:
        train_loader, val_loader, test_loader, info = create_bach_dataloaders(config, bach_root)
    else:
        raise ValueError("train_bach requires either bach_root or pre-built loaders.")
    num_classes = info['num_classes']
    class_names = info['class_names']
    model = build_single_gpcnvit(config, num_classes).to(device)
    class_weights = info['class_weights'].to(device)

    if config.training.use_focal_loss:
        criterion = FocalLoss(alpha=config.training.focal_alpha,
                              gamma=config.training.focal_gamma,
                              class_weights=class_weights).to(device)
    else:
        criterion = LabelSmoothingCrossEntropy(config.training.label_smoothing).to(device)

    new_keys = ('gpcn', 'fusion', '.alpha', 'head')
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

    save_dir = Path(config.data.save_dir) / (config.experiment_name or 'gpcn_bach') / 'checkpoints'
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / 'best_bach.pth'
    last_path = save_dir / 'last_bach.pth'

    best_score, best_epoch, start_epoch = -1.0, 0, 0
    # AUC is binary-style under MetricsCalculator; for multi-class select on accuracy.
    monitor = 'accuracy'

    if getattr(config.training, 'auto_resume', True) and last_path.exists():
        try:
            ck = torch.load(str(last_path), map_location=device, weights_only=False)
            model.load_state_dict(ck['model_state_dict'])
            optimizer.load_state_dict(ck['optimizer_state_dict'])
            scheduler.load_state_dict(ck['scheduler_state_dict'])
            if ema is not None and ck.get('ema_state'):
                ema.load_state_dict(ck['ema_state'])
            start_epoch = ck.get('epoch', -1) + 1
            best_score = ck.get('best_score', -1.0)
            best_epoch = ck.get('best_epoch', 0)
            logger.info(f"✓ [bach] resuming from epoch {start_epoch + 1}/{config.training.num_epochs}")
        except Exception as e:
            logger.warning(f"[bach] auto-resume failed ({e}); starting fresh.")

    logger.info("=" * 70 + f"\nBACH MULTI-CLASS TRAINING ({num_classes} classes: {class_names})\n" + "=" * 70)
    for epoch in range(start_epoch, config.training.num_epochs):
        model.train()
        running = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                if isinstance(logits, tuple):
                    logits = logits[0]
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

        if ema is not None:
            ema.apply_to(model)
        try:
            val_metrics, *_ = evaluate_bach(model, val_loader, device, num_classes, class_names)
        finally:
            if ema is not None:
                ema.restore(model)
        score = val_metrics.get(monitor, val_metrics['accuracy'])
        logger.info(f"Epoch {epoch+1}/{config.training.num_epochs} | loss={running/max(len(train_loader),1):.4f} "
                    f"| val_acc={val_metrics['accuracy']:.4f} val_f1m={val_metrics['f1_macro']:.4f} "
                    f"val_auc={val_metrics['auc_macro_ovr']:.4f}")

        if score > best_score:
            best_score, best_epoch = score, epoch
            if ema is not None:
                ema.apply_to(model)
            try:
                torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch,
                            'best_score': best_score, 'class_names': class_names}, best_path)
            finally:
                if ema is not None:
                    ema.restore(model)
            logger.info(f"  ✓ new best ({monitor}={score:.4f}) saved")

        last_ckpt = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'ema_state': ema.state_dict() if ema is not None else None,
            'epoch': epoch, 'best_score': best_score, 'best_epoch': best_epoch,
            'class_names': class_names,
        }
        torch.save(last_ckpt, last_path)

        # Per-epoch checkpoint (mirrors BreakHis) so the reached epoch is known
        # after a break/crash. Saved every save_frequency epochs.
        save_freq = getattr(config.training, 'save_frequency', 5) or 1
        if (epoch + 1) % save_freq == 0:
            torch.save(last_ckpt, save_dir / f'checkpoint_bach_epoch_{epoch + 1}.pth')

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)['model_state_dict'])

    test_metrics, *_ = evaluate_bach(model, test_loader, device, num_classes, class_names)
    logger.info("\nBACH Multi-class Test Results:")
    for k, v in sorted(test_metrics.items()):
        if k == 'confusion_matrix':
            logger.info(f"  {k}: {v}")
        else:
            logger.info(f"  {k:<22} {v:.4f}")

    # Post-hoc calibration: fit a single temperature on the VALIDATION set and
    # report ECE/MCE before vs after on the TEST set (top-label reliability for
    # the K>2 BACH task). This does not change accuracy/AUC — it only rescales
    # the probabilities — but the raw model is typically over-confident, so the
    # calibrated ECE/MCE are the numbers to report in the manuscript.
    calibration = None
    if getattr(config.training, 'run_calibration', True):
        try:
            from calibration_report import temperature_calibration_report
            diagram_path = str(save_dir / 'bach_reliability.png')
            calibration = temperature_calibration_report(
                model, val_loader, test_loader, device,
                save_path=diagram_path, class_names=class_names,
                mode='auto', show=False)
            # Surface the calibrated numbers alongside the raw test metrics so a
            # single results record carries both (raw ECE stays in test_metrics).
            test_metrics['ece_calibrated'] = calibration['ece_after']
            test_metrics['mce_calibrated'] = calibration['mce_after']
            test_metrics['temperature'] = calibration['temperature']
            logger.info(
                f"\nBACH Calibration (temperature scaling): T={calibration['temperature']:.4f}\n"
                f"  ECE {calibration['ece_before']:.4f} -> {calibration['ece_after']:.4f}"
                f"  |  MCE {calibration['mce_before']:.4f} -> {calibration['mce_after']:.4f}\n"
                f"  reliability diagram: {diagram_path}")
        except Exception as e:
            logger.warning(f"[bach] calibration report skipped ({e}).")

    return {'best_val_score': best_score, 'best_epoch': best_epoch,
            'test_metrics': test_metrics, 'calibration': calibration, 'info': info}


# ----------------------------------------------------------------------------
# Loaders from an explicit item partition (for k-fold CV)
# ----------------------------------------------------------------------------
def _loaders_from_items(config, train_items, val_items, test_items,
                        num_classes, class_names) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    train_ds = BACHDataset(train_items, get_train_transform(config))
    val_ds = BACHDataset(val_items, get_val_transform(config))
    test_ds = BACHDataset(test_items, get_val_transform(config))
    common = dict(num_workers=config.data.num_workers, pin_memory=config.data.pin_memory)
    train_loader = DataLoader(train_ds, batch_size=config.data.batch_size,
                              shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_ds, batch_size=config.data.batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=config.data.batch_size, shuffle=False, **common)
    info = {'num_classes': num_classes, 'class_names': class_names,
            'class_weights': train_ds.class_weights(num_classes),
            'train': len(train_ds), 'val': len(val_ds), 'test': len(test_ds)}
    return train_loader, val_loader, test_loader, info


def make_bach_cv_folds(index, num_classes, n_folds, seed, val_frac=0.15):
    """Stratified, image-level K-fold partition for BACH (no patient grouping
    exists, so image-level stratification is the accepted protocol).

    For each fold returns (train_items, val_items, test_items):
      * test_items  = the held-out fold,
      * val_items   = a stratified `val_frac` slice carved from the remainder
                      (for model selection / temperature fitting),
      * train_items = the rest.
    Every image is in the test fold exactly once; the three sets are disjoint.
    """
    from sklearn.model_selection import StratifiedKFold, train_test_split
    items = list(index)
    labels = [it[1] for it in items]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for trainval_idx, test_idx in skf.split(items, labels):
        test_it = [items[i] for i in test_idx]
        tv_it = [items[i] for i in trainval_idx]
        tv_y = [labels[i] for i in trainval_idx]
        train_it, val_it = train_test_split(
            tv_it, test_size=val_frac, random_state=seed, stratify=tv_y)
        folds.append((train_it, val_it, test_it))
    return folds


def run_bach_kfold_cv(config, bach_root: str, n_folds: int = 5,
                      results_dir: Optional[str] = None, val_frac: float = 0.15) -> Dict:
    """Patient-free, image-level stratified K-fold CV for BACH using the same
    training recipe as train_bach. Reports each metric as mean ± std and a 95%
    CI across folds. Crash-safe: each fold caches `cv_bach_fold<k>.json` and is
    skipped on re-run (so a Colab disconnect costs at most the in-progress fold).
    """
    import json
    from utils import set_seed
    from kfold_cv import aggregate_folds, build_cv_table, _write_cv_csv

    set_seed(config.seed, config.deterministic)
    index, class_names = build_bach_index(bach_root)
    num_classes = len(class_names)

    base_name = config.experiment_name or 'gpcn_bach'
    rdir = Path(results_dir) if results_dir else Path(config.data.save_dir) / base_name / 'cv'
    rdir.mkdir(parents=True, exist_ok=True)

    folds = make_bach_cv_folds(index, num_classes, n_folds, config.seed, val_frac)
    logger.info("=" * 70 + f"\nBACH IMAGE-LEVEL {n_folds}-FOLD CV | {num_classes} classes "
                f"{class_names} | backbone={config.model.backbone}\n" + "=" * 70)

    per_fold: List[Dict[str, float]] = []
    for k, (train_it, val_it, test_it) in enumerate(folds):
        fold_json = rdir / f'cv_bach_fold{k}.json'
        if fold_json.exists():
            with open(fold_json) as f:
                rec = json.load(f)
            per_fold.append(rec['metrics'])
            logger.info(f"✓ Fold {k + 1}/{n_folds} cached — skipping "
                        f"(acc={rec['metrics'].get('accuracy', float('nan')):.4f})")
            continue

        logger.info(f"\n{'#' * 70}\n# BACH FOLD {k + 1}/{n_folds} | "
                    f"train={len(train_it)} val={len(val_it)} test={len(test_it)} images\n{'#' * 70}")
        # Per-fold experiment_name so checkpoints/auto-resume don't collide.
        config.experiment_name = f"{base_name}_cv{n_folds}_fold{k}"
        loaders = _loaders_from_items(config, train_it, val_it, test_it, num_classes, class_names)
        out = train_bach(config, loaders=loaders)
        metrics = {kk: float(vv) for kk, vv in out['test_metrics'].items()
                   if isinstance(vv, (int, float)) and not isinstance(vv, bool)}
        rec = {'fold': k, 'backbone': config.model.backbone,
               'use_gpcn': getattr(config.model, 'use_gpcn', True),
               'n_train': len(train_it), 'n_val': len(val_it), 'n_test': len(test_it),
               'best_val_score': out.get('best_val_score'), 'best_epoch': out.get('best_epoch'),
               'metrics': metrics}
        with open(fold_json, 'w') as f:
            json.dump(rec, f, indent=2)
        logger.info(f"✓ Saved {fold_json} (acc={metrics.get('accuracy', float('nan')):.4f})")
        per_fold.append(metrics)

    config.experiment_name = base_name  # restore
    aggregate = aggregate_folds(per_fold)
    summary = {'dataset': 'BACH', 'folds': n_folds, 'num_classes': num_classes,
               'class_names': class_names, 'backbone': config.model.backbone,
               'use_gpcn': getattr(config.model, 'use_gpcn', True),
               'per_fold': per_fold, 'aggregate': aggregate}
    with open(rdir / 'cv_bach_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    table = build_cv_table(aggregate)
    logger.info("\n" + "=" * 70 + f"\nBACH {n_folds}-FOLD CV RESULTS\n" + "=" * 70 + "\n" + table)
    _write_cv_csv(aggregate, rdir / 'cv_bach_summary.csv')
    logger.info(f"✓ BACH CV artifacts in {rdir}")
    return summary


# ----------------------------------------------------------------------------
# Ablation: GPCN ON vs OFF on BACH, repeated over seeds, with significance
# ----------------------------------------------------------------------------
def run_bach_ablation(config, bach_root: str, seeds=(0, 1, 2),
                      results_dir: str = 'results') -> Dict:
    """GPCN ON vs OFF (plain ViT) on BACH under identical settings, repeated
    over `seeds`. Saves a per-(arm,seed) result JSON (crash-safe: existing ones
    are skipped), then runs a PAIRED t-test per metric across seeds so the
    GPCN contribution comes with a p-value and a 95% CI — not a single number.
    """
    import json
    from stats_tests import compare_runs, format_comparison_table

    rdir = Path(results_dir); rdir.mkdir(parents=True, exist_ok=True)
    base_name = config.experiment_name or 'gpcn_bach'
    # Metrics to test for significance (present in evaluate_bach output).
    metrics = ['accuracy', 'balanced_accuracy', 'f1_macro', 'auc_macro_ovr', 'mcc', 'ece']

    records = {True: [], False: []}  # use_gpcn -> [record per seed], seed-aligned
    for seed in seeds:
        for use_gpcn in (True, False):
            tag = f"bach_ablation_seed{seed}_gpcn{'ON' if use_gpcn else 'OFF'}"
            run_json = rdir / f"{tag}.json"
            if run_json.exists():
                with open(run_json) as f:
                    rec = json.load(f)
                records[use_gpcn].append(rec)
                logger.info(f"✓ {tag} cached — skipping (acc={rec['metrics'].get('accuracy', float('nan')):.4f})")
                continue
            config.seed = seed
            config.model.use_gpcn = use_gpcn
            config.experiment_name = tag
            logger.info(f"\n{'#' * 70}\n# BACH ABLATION: seed={seed} GPCN={'ON' if use_gpcn else 'OFF'}\n{'#' * 70}")
            out = train_bach(config, bach_root=bach_root)
            rec = {'tag': tag, 'seed': seed, 'use_gpcn': use_gpcn,
                   'backbone': config.model.backbone,
                   'best_epoch': out.get('best_epoch'),
                   'metrics': {k: float(v) for k, v in out['test_metrics'].items()
                               if isinstance(v, (int, float)) and not isinstance(v, bool)}}
            with open(run_json, 'w') as f:
                json.dump(rec, f, indent=2)
            records[use_gpcn].append(rec)

    config.experiment_name = base_name  # restore
    reports = compare_runs(records[True], records[False], metrics,
                           name_a='GPCN ON', name_b='GPCN OFF (plain ViT)')
    table = format_comparison_table(reports)
    logger.info("\n" + "=" * 70 + "\nBACH ABLATION — GPCN ON vs OFF (paired over seeds)\n"
                + "=" * 70 + "\n" + table)
    summary = {'dataset': 'BACH', 'seeds': list(seeds), 'metrics_tested': metrics,
               'comparisons': reports,
               'records_on': records[True], 'records_off': records[False]}
    with open(rdir / 'bach_ablation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"✓ BACH ablation summary in {rdir / 'bach_ablation_summary.json'}")
    return summary


if __name__ == '__main__':
    import argparse
    from config import get_default_config
    ap = argparse.ArgumentParser(description='BACH multi-class GPCN-ViT')
    ap.add_argument('--bach_root', required=True)
    ap.add_argument('--mode', default='train',
                    choices=['train', 'kfold', 'ablation'],
                    help='train: single split; kfold: K-fold CV; ablation: GPCN on/off over seeds')
    ap.add_argument('--n_folds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--seeds', type=int, nargs='*', default=[0, 1, 2])
    ap.add_argument('--results_dir', default='results')
    args = ap.parse_args()

    cfg = get_default_config()
    cfg.experiment_name = 'gpcn_bach'
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.epochs is not None:
        cfg.training.num_epochs = args.epochs

    if args.mode == 'train':
        train_bach(cfg, args.bach_root)
    elif args.mode == 'kfold':
        run_bach_kfold_cv(cfg, args.bach_root, n_folds=args.n_folds, results_dir=args.results_dir)
    else:
        run_bach_ablation(cfg, args.bach_root, seeds=tuple(args.seeds), results_dir=args.results_dir)
