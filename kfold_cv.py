"""
kfold_cv.py — Patient-level K-fold cross-validation REPORTING for GPCN-ViT.

WHY THIS EXISTS
---------------
The headline BreaKHis results come from ONE patient-level split. Reviewers
(rightly) ask: is that number stable, or did you get a lucky split? This module
answers that with **stratified, patient-level K-fold cross-validation** and
reports each metric as **mean ± std and a 95% confidence interval** across folds.

RIGOUR (the things that make this leakage-free and honest)
----------------------------------------------------------
1. **Patient-level folds.** Folds are built over PATIENTS, not images. The same
   patient's slides never appear in two folds — no leakage (the central BreaKHis
   pitfall). Folds are label-STRATIFIED on the patient consensus label.
2. **Held-out test fold + carved validation.** For fold k, fold k is the TEST
   set; a small validation set is carved (stratified) out of the REMAINING
   patients for model selection and decision-threshold tuning. The test fold is
   therefore never used to pick the model or the threshold — no test peeking.
3. **Identical recipe per fold.** Same config (backbone, EMA, warmup→cosine,
   discriminative LR, class-weighted focal, TTA, threshold tuning) the single
   run uses; only the patient partition changes.
4. **Crash-safe.** Each fold writes its own result JSON and uses its own
   experiment_name (so the Trainer's auto-resume works per fold). Re-running
   skips folds whose result JSON already exists, so a Colab disconnect costs at
   most the in-progress fold.

Usage (Colab / notebook):
    from config import get_full_config
    from kfold_cv import run_kfold_cv, build_cv_table
    cfg = get_full_config()
    cfg.data.data_root = DATASET_PATH; cfg.data.save_dir = OUTPUT_PATH
    cfg.data.train_magnification = '200X'      # or set pool_all_magnifications=True
    cfg.training.num_epochs = 50
    out = run_kfold_cv(cfg, n_folds=5)
    print(build_cv_table(out['aggregate']))

Usage (CLI):
    python kfold_cv.py --data_root <BreaKHis> --save_dir <out> --mag 200X --n_folds 5 --epochs 50
    python kfold_cv.py --pooled --n_folds 5 --epochs 50   # magnification-agnostic
"""

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Metrics reported in the CV table, in column order, with display names. Only
# those actually present in every fold's test_metrics are shown.
REPORT_KEYS = [
    ('accuracy', 'Accuracy', '%'),
    ('auc_roc', 'AUC-ROC', 'f'),
    ('sensitivity', 'Sensitivity', '%'),
    ('specificity', 'Specificity', '%'),
    ('f1_class1', 'F1 (malignant)', 'f'),
    ('mcc', 'MCC', 'f'),
    ('ece', 'ECE', 'f'),
    ('brier', 'Brier', 'f'),
    ('decision_threshold', 'Threshold', 'f'),
]

# Two-tailed t critical values at 95% for small samples (df = n_folds - 1).
# Falls back to the normal-approximation 1.96 for df not listed / large df.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
        13: 2.160, 14: 2.145, 19: 2.093, 24: 2.064, 29: 2.045}


def _t_crit(df: int) -> float:
    if df <= 0:
        return float('nan')
    if df in _T95:
        return _T95[df]
    return 1.96  # large-sample normal approximation


# ----------------------------------------------------------------------------
# Patient-level, stratified fold construction (with a carved validation set)
# ----------------------------------------------------------------------------
def _collect_patient_labels(root_dir: str, mags: List[str]) -> Tuple[List[str], List[int]]:
    """Return (patients, labels) with one consensus (majority-vote) label per
    patient, scanning across all requested magnifications."""
    from collections import defaultdict
    root = Path(root_dir)
    patient_to_label: Dict[str, List[int]] = defaultdict(list)
    for mag in mags:
        pattern = f"histology_slides/breast/*/SOB/*/*/{mag}/*.png"
        for img_path in root.glob(pattern):
            parts = img_path.parts
            try:
                breast_idx = parts.index('breast')
                class_name = parts[breast_idx + 1]
                if class_name not in ('benign', 'malignant'):
                    continue
                patient_id = img_path.stem.split('_')[2]
                patient_to_label[patient_id].append(0 if class_name == 'benign' else 1)
            except (ValueError, IndexError):
                continue
    patients = list(patient_to_label.keys())
    labels = [1 if sum(patient_to_label[p]) > len(patient_to_label[p]) / 2 else 0
              for p in patients]
    if not patients:
        raise RuntimeError(f"No BreaKHis patients found under {root_dir} for mags {mags}.")
    return patients, labels


def make_cv_folds(root_dir: str, mags: List[str], n_folds: int, seed: int,
                  val_frac: float = 0.15
                  ) -> List[Tuple[List[str], List[str], List[str]]]:
    """Build `n_folds` patient-level, label-stratified folds.

    For each fold returns (train_patients, val_patients, test_patients):
      * test_patients  = the held-out fold,
      * val_patients   = a stratified `val_frac` slice of the remaining patients
                         (for model selection / threshold tuning),
      * train_patients = the rest.
    Guarantees disjoint patient sets (no leakage) and that every patient is the
    test fold exactly once.
    """
    from sklearn.model_selection import StratifiedKFold, train_test_split
    patients, labels = _collect_patient_labels(root_dir, mags)
    patients = list(patients)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds: List[Tuple[List[str], List[str], List[str]]] = []
    for trainval_idx, test_idx in skf.split(patients, labels):
        test_p = [patients[i] for i in test_idx]
        tv_p = [patients[i] for i in trainval_idx]
        tv_y = [labels[i] for i in trainval_idx]
        # Stratified val carve-out from the train+val remainder.
        train_p, val_p = train_test_split(
            tv_p, test_size=val_frac, random_state=seed, stratify=tv_y)
        folds.append((train_p, val_p, test_p))
    return folds


# ----------------------------------------------------------------------------
# Dataloaders for an explicit patient partition (mirrors dataset.create_dataloaders)
# ----------------------------------------------------------------------------
def _build_loaders(config, mag_spec, train_p, val_p, test_p):
    from torch.utils.data import DataLoader
    from dataset import BREAKHISDataset
    from augmentation import get_train_transform, get_val_transform

    train_ds = BREAKHISDataset(config.data.data_root, 'train', mag_spec,
                               get_train_transform(config), train_p, True)
    val_ds = BREAKHISDataset(config.data.data_root, 'val', mag_spec,
                             get_val_transform(config), val_p, True)
    test_ds = BREAKHISDataset(config.data.data_root, 'test', mag_spec,
                              get_val_transform(config), test_p, True)

    common = dict(num_workers=config.data.num_workers, pin_memory=config.data.pin_memory)
    train_loader = DataLoader(train_ds, batch_size=config.data.batch_size,
                              shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_ds, batch_size=config.data.batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=config.data.batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader, train_ds.get_class_weights()


# ----------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------
def aggregate_folds(per_fold: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Aggregate a list of per-fold test-metric dicts into
    {metric: {mean, std, ci95_low, ci95_high, n, values}} for every scalar
    metric present in ALL folds."""
    if not per_fold:
        return {}
    common_keys = set(per_fold[0].keys())
    for m in per_fold[1:]:
        common_keys &= set(m.keys())
    n = len(per_fold)
    df = n - 1
    tcrit = _t_crit(df)
    agg: Dict[str, Dict[str, float]] = {}
    for k in sorted(common_keys):
        vals = []
        for m in per_fold:
            v = m.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                vals.append(float(v))
        if len(vals) != n:  # skip non-scalar / inconsistent keys (e.g. confusion_matrix)
            continue
        mean = sum(vals) / n
        if n > 1:
            var = sum((v - mean) ** 2 for v in vals) / (n - 1)  # sample variance
            std = math.sqrt(var)
            half = tcrit * std / math.sqrt(n)
        else:
            std = 0.0
            half = float('nan')
        agg[k] = {'mean': mean, 'std': std, 'ci95_low': mean - half,
                  'ci95_high': mean + half, 'n': n, 'values': vals}
    return agg


def build_cv_table(aggregate: Dict[str, Dict[str, float]]) -> str:
    """Markdown table: Metric | Mean ± Std | 95% CI | per-fold values."""
    lines = ['| Metric | Mean ± Std | 95% CI | Per-fold |',
             '|---|---|---|---|']

    def fmt(v, kind):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return '—'
        if kind == '%':
            return f'{v * 100:.2f}'
        return f'{v:.4f}'

    shown = set()
    ordered = [(k, name, kind) for k, name, kind in REPORT_KEYS if k in aggregate]
    # Append any other aggregated scalar metrics after the curated ones.
    ordered += [(k, k, 'f') for k in aggregate if k not in {kk for kk, _, _ in REPORT_KEYS}]

    for k, name, kind in ordered:
        if k in shown:
            continue
        shown.add(k)
        a = aggregate[k]
        suffix = '%' if kind == '%' else ''
        mean_std = f'{fmt(a["mean"], kind)}{suffix} ± {fmt(a["std"], kind)}{suffix}'
        ci = f'[{fmt(a["ci95_low"], kind)}, {fmt(a["ci95_high"], kind)}]{suffix}'
        folds = ', '.join(fmt(v, kind) for v in a['values'])
        lines.append(f'| {name} | {mean_std} | {ci} | {folds} |')
    return '\n'.join(lines)


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def run_kfold_cv(config, n_folds: int = 5, results_dir: Optional[str] = None,
                 val_frac: float = 0.15) -> Dict:
    """Run patient-level K-fold CV with the standard training recipe and return
    {'per_fold': [...], 'aggregate': {...}, 'folds': n, 'mag': ...}.

    Crash-safe: each fold's metrics are cached to `<results_dir>/cv_fold<k>.json`
    and skipped on re-run. Set a distinct base experiment_name per CV study.
    """
    import torch
    from utils import set_seed, setup_logging
    from model import create_model
    from trainer import Trainer

    setup_logging()
    set_seed(config.seed, config.deterministic)
    if not getattr(config, 'device', None):
        config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pooled = getattr(config.data, 'pool_all_magnifications', False)
    mag_spec = list(config.data.magnifications) if pooled else config.data.train_magnification
    mag_tag = 'ALL' if pooled else str(mag_spec)
    mags = mag_spec if isinstance(mag_spec, list) else [mag_spec]

    base_name = config.experiment_name or 'gpcn'
    rdir = Path(results_dir) if results_dir else Path(config.data.save_dir) / base_name / 'cv'
    rdir.mkdir(parents=True, exist_ok=True)

    folds = make_cv_folds(config.data.data_root, mags, n_folds, config.seed, val_frac)
    logger.info("=" * 70 + f"\nPATIENT-LEVEL {n_folds}-FOLD CV | mag={mag_tag} | "
                f"backbone={config.model.backbone}\n" + "=" * 70)

    per_fold: List[Dict[str, float]] = []
    for k, (train_p, val_p, test_p) in enumerate(folds):
        fold_json = rdir / f'cv_fold{k}.json'
        if fold_json.exists():  # resume at fold granularity
            with open(fold_json) as f:
                rec = json.load(f)
            per_fold.append(rec['metrics'])
            logger.info(f"✓ Fold {k + 1}/{n_folds} cached — skipping "
                        f"(acc={rec['metrics'].get('accuracy', float('nan')):.4f})")
            continue

        logger.info(f"\n{'#' * 70}\n# FOLD {k + 1}/{n_folds} | "
                    f"train={len(train_p)} val={len(val_p)} test={len(test_p)} patients\n{'#' * 70}")

        # Per-fold experiment_name so Trainer checkpoints/auto-resume don't collide.
        config.experiment_name = f"{base_name}_cv{n_folds}_{mag_tag}_fold{k}"
        train_loader, val_loader, test_loader, class_weights = _build_loaders(
            config, mag_spec, train_p, val_p, test_p)
        model = create_model(config)
        trainer = Trainer(model=model, config=config,
                          train_loader=train_loader, val_loader=val_loader,
                          test_loader=test_loader,
                          class_weights=class_weights.to(config.device))
        out = trainer.train()
        metrics = {k2: float(v) for k2, v in out['test_metrics'].items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool)}

        rec = {'fold': k, 'magnification': mag_tag,
               'backbone': config.model.backbone, 'use_gpcn': config.model.use_gpcn,
               'n_train': len(train_p), 'n_val': len(val_p), 'n_test': len(test_p),
               'best_val_auc': out.get('best_val_auc'), 'best_epoch': out.get('best_epoch'),
               'metrics': metrics}
        with open(fold_json, 'w') as f:
            json.dump(rec, f, indent=2)
        logger.info(f"✓ Saved {fold_json} (acc={metrics.get('accuracy', float('nan')):.4f})")
        per_fold.append(metrics)

    config.experiment_name = base_name  # restore
    aggregate = aggregate_folds(per_fold)

    summary = {'folds': n_folds, 'magnification': mag_tag,
               'backbone': config.model.backbone, 'use_gpcn': config.model.use_gpcn,
               'per_fold': per_fold, 'aggregate': aggregate}
    with open(rdir / 'cv_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    table = build_cv_table(aggregate)
    logger.info("\n" + "=" * 70 + f"\n{n_folds}-FOLD CV RESULTS (mag={mag_tag})\n" + "=" * 70 + "\n" + table)
    _write_cv_csv(aggregate, rdir / 'cv_summary.csv')
    logger.info(f"✓ CV artifacts in {rdir}")
    return summary


def _write_cv_csv(aggregate: Dict[str, Dict[str, float]], path: Path):
    import csv
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'mean', 'std', 'ci95_low', 'ci95_high', 'n'])
        for k, a in aggregate.items():
            w.writerow([k, a['mean'], a['std'], a['ci95_low'], a['ci95_high'], a['n']])


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    import torch
    from config import get_full_config
    ap = argparse.ArgumentParser(description='Patient-level K-fold CV for GPCN-ViT')
    ap.add_argument('--data_root', default=None)
    ap.add_argument('--save_dir', default=None)
    ap.add_argument('--mag', default='200X', help='single magnification (ignored if --pooled)')
    ap.add_argument('--pooled', action='store_true', help='magnification-agnostic (pool all 4)')
    ap.add_argument('--n_folds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--backbone', default=None)
    ap.add_argument('--val_frac', type=float, default=0.15)
    args = ap.parse_args()

    cfg = get_full_config()
    if args.data_root:
        cfg.data.data_root = args.data_root
    if args.save_dir:
        cfg.data.save_dir = args.save_dir
    if args.backbone:
        cfg.model.backbone = args.backbone
    if args.epochs is not None:
        cfg.training.num_epochs = args.epochs
    cfg.data.pool_all_magnifications = bool(args.pooled)
    if not args.pooled:
        cfg.data.train_magnification = args.mag
    cfg.experiment_name = 'gpcn_cv'
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    out = run_kfold_cv(cfg, n_folds=args.n_folds, val_frac=args.val_frac)
    print('\n\n' + '=' * 70 + '\nK-FOLD CV SUMMARY\n' + '=' * 70)
    print(build_cv_table(out['aggregate']))


if __name__ == '__main__':
    main()
