"""
experiments.py — Publication-grade experiment driver for GPCN-ViT on BreakHis.

What this gives you (the things reviewers ask for):
  1. Per-magnification runs (40X/100X/200X/400X) — the standard BreakHis protocol.
  2. Magnification-pooled run — one magnification-agnostic model on all data.
  3. Ablation — GPCN ON vs OFF (plain ViT) under identical settings, to isolate
     the contribution of the Graph Patch Correlation Network.
  4. Backbone comparison — ImageNet-ViT vs Phikon (pathology FM).
  5. Full metric collection (acc, AUC, sensitivity, specificity, F1, MCC, ECE,
     Brier, tuned decision threshold) saved to JSON + CSV.
  6. A publication-ready comparison table vs published BreakHis SOTA.

Usage (CLI):
    python experiments.py --mode per_mag            # 4 magnifications, default backbone
    python experiments.py --mode pooled             # magnification-agnostic model
    python experiments.py --mode ablation           # GPCN on/off at 40X
    python experiments.py --mode backbone           # ImageNet vs Phikon at 40X
    python experiments.py --mode table --results results/*.json   # build tables only

Usage (Colab / notebook):
    from experiments import run_all_magnifications, build_comparison_table
    results = run_all_magnifications(backbone='hf-hub:1aurent/vit_base_patch16_224.owkin_pancancer')
    print(build_comparison_table(results))

NOTE: the published-SOTA numbers in PUBLISHED_SOTA are pre-filled from commonly
cited works but every value is tagged for verification — CONFIRM each against the
original paper (protocol differs: image- vs patient-level, binary vs multiclass)
before putting the table in a manuscript.
"""

import argparse
import json
import glob
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

MAGNIFICATIONS = ['40X', '100X', '200X', '400X']

# Metrics we report, in column order, with display names.
REPORT_KEYS = [
    ('accuracy', 'Accuracy'),
    ('auc_roc', 'AUC-ROC'),
    ('sensitivity', 'Sensitivity'),
    ('specificity', 'Specificity'),
    ('f1_class1', 'F1 (malignant)'),
    ('mcc', 'MCC'),
    ('ece', 'ECE'),
    ('decision_threshold', 'Threshold'),
]


# ----------------------------------------------------------------------------
# Published BreakHis SOTA (image-level binary benign/malignant accuracy, %).
# EVERY NUMBER IS [VERIFY] — confirm against the cited source before publishing.
# Protocols differ; we note them so the comparison is apples-to-apples.
# ----------------------------------------------------------------------------
PUBLISHED_SOTA: List[Dict] = [
    {
        'method': 'Spanhol et al. (AlexNet)', 'year': 2016,
        'protocol': 'image-level binary', 'source': 'Spanhol et al., IJCNN 2016 [VERIFY]',
        '40X': 90.0, '100X': 88.4, '200X': 84.6, '400X': 86.1,
    },
    {
        'method': 'Bayramoglu et al. (magnification-independent CNN)', 'year': 2016,
        'protocol': 'image-level binary', 'source': 'Bayramoglu et al., ICPR 2016 [VERIFY]',
        '40X': 83.0, '100X': 83.1, '200X': 84.6, '400X': 82.1,
    },
    {
        'method': 'Gour et al. (ResHist)', 'year': 2020,
        'protocol': 'image-level binary', 'source': 'Gour et al., 2020 [VERIFY]',
        '40X': 87.0, '100X': 88.0, '200X': 90.0, '400X': 85.0,
    },
    {
        'method': 'Boumaraf et al. (fine-tuned ResNet-18)', 'year': 2021,
        'protocol': 'image-level binary', 'source': 'Boumaraf et al., 2021 [VERIFY]',
        '40X': 98.4, '100X': 97.9, '200X': 98.5, '400X': 97.0,
    },
]


# ----------------------------------------------------------------------------
# Running experiments
# ----------------------------------------------------------------------------
def _run_one(config, tag: str, results_dir: Path) -> Dict:
    """Train + evaluate a single configured run; return its test metrics."""
    # Imports are local so `--mode table` works without torch installed.
    import torch
    from utils import set_seed, setup_logging
    from model import create_model
    from dataset import create_dataloaders
    from trainer import Trainer

    setup_logging()
    set_seed(config.seed, config.deterministic)

    logger.info(f"\n{'#'*70}\n# RUN: {tag}\n{'#'*70}")
    train_loader, val_loader, test_loader, info = create_dataloaders(config)
    model = create_model(config)
    trainer = Trainer(
        model=model, config=config,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        class_weights=info['class_weights'].to(config.device),
    )
    out = trainer.train()
    test_metrics = out['test_metrics']

    record = {
        'tag': tag,
        'backbone': config.model.backbone,
        'use_gpcn': config.model.use_gpcn,
        'magnification': ('ALL' if getattr(config.data, 'pool_all_magnifications', False)
                          else config.data.train_magnification),
        'best_val_auc': out['best_val_auc'],
        'best_epoch': out['best_epoch'],
        'metrics': {k: float(v) for k, v in test_metrics.items()},
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{tag}.json"
    with open(out_path, 'w') as f:
        json.dump(record, f, indent=2)
    logger.info(f"✓ Saved {out_path}")
    return record


def _base_config(backbone: Optional[str], use_gpcn: bool, epochs: Optional[int],
                 data_root: Optional[str] = None, save_dir: Optional[str] = None):
    from config import get_full_config
    config = get_full_config()
    if backbone is not None:
        config.model.backbone = backbone
    config.model.use_gpcn = use_gpcn
    if epochs is not None:
        config.training.num_epochs = epochs
    # Colab path overrides (so callers don't depend on config defaults).
    if data_root is not None:
        config.data.data_root = data_root
    if save_dir is not None:
        config.data.save_dir = save_dir
    return config


def run_all_magnifications(backbone: Optional[str] = None, use_gpcn: bool = True,
                           epochs: Optional[int] = None, results_dir: str = 'results',
                           data_root: Optional[str] = None, save_dir: Optional[str] = None) -> List[Dict]:
    """Standard BreakHis protocol: one model per magnification."""
    rdir = Path(results_dir)
    records = []
    for mag in MAGNIFICATIONS:
        config = _base_config(backbone, use_gpcn, epochs, data_root, save_dir)
        config.data.pool_all_magnifications = False
        config.data.train_magnification = mag
        config.experiment_name = f"gpcn_{mag}_{'phikon' if backbone and 'owkin' in backbone else 'vit'}"
        records.append(_run_one(config, config.experiment_name, rdir))
    return records


def run_pooled(backbone: Optional[str] = None, use_gpcn: bool = True,
               epochs: Optional[int] = None, results_dir: str = 'results',
               data_root: Optional[str] = None, save_dir: Optional[str] = None) -> List[Dict]:
    """Magnification-agnostic model trained on all four magnifications pooled."""
    config = _base_config(backbone, use_gpcn, epochs, data_root, save_dir)
    config.data.pool_all_magnifications = True
    config.experiment_name = "gpcn_pooled"
    return [_run_one(config, config.experiment_name, Path(results_dir))]


def run_ablation(backbone: Optional[str] = None, mag: str = '40X',
                 epochs: Optional[int] = None, results_dir: str = 'results',
                 data_root: Optional[str] = None, save_dir: Optional[str] = None,
                 seeds=(0, 1, 2)) -> List[Dict]:
    """GPCN ON vs OFF (plain ViT) at one magnification, identical settings,
    repeated over `seeds`. Runs are crash-safe (existing per-(seed,arm) JSONs are
    skipped). After all runs, a PAIRED t-test per metric across seeds quantifies
    the GPCN contribution with a p-value and 95% CI, written to
    `ablation_<mag>_significance.json`. Returns the flat list of run records."""
    from stats_tests import compare_runs, format_comparison_table
    rdir = Path(results_dir)
    metrics = ['accuracy', 'auc_roc', 'sensitivity', 'specificity', 'f1_class1', 'mcc', 'ece']
    by_arm = {True: [], False: []}  # seed-aligned per arm
    records: List[Dict] = []
    for seed in seeds:
        for use_gpcn in (True, False):
            tag = f"ablation_{mag}_gpcn{'ON' if use_gpcn else 'OFF'}_seed{seed}"
            run_json = rdir / f"{tag}.json"
            if run_json.exists():  # resume at run granularity
                with open(run_json) as f:
                    rec = json.load(f)
            else:
                config = _base_config(backbone, use_gpcn, epochs, data_root, save_dir)
                config.seed = seed
                config.data.pool_all_magnifications = False
                config.data.train_magnification = mag
                config.experiment_name = tag
                rec = _run_one(config, tag, rdir)
            by_arm[use_gpcn].append(rec)
            records.append(rec)

    try:
        reports = compare_runs(by_arm[True], by_arm[False], metrics,
                               name_a='GPCN ON', name_b='GPCN OFF (plain ViT)')
        table = format_comparison_table(reports)
        logger.info("\n" + "=" * 70 + f"\nBREAKHIS ABLATION @ {mag} — GPCN ON vs OFF "
                    f"(paired over {len(seeds)} seeds)\n" + "=" * 70 + "\n" + table)
        with open(rdir / f'ablation_{mag}_significance.json', 'w') as f:
            json.dump({'dataset': 'BreakHis', 'magnification': mag, 'seeds': list(seeds),
                       'metrics_tested': metrics, 'comparisons': reports}, f, indent=2)
        logger.info(f"✓ Significance summary in {rdir / f'ablation_{mag}_significance.json'}")
    except Exception as e:
        logger.warning(f"Ablation significance summary skipped ({e}).")
    return records


def run_multimag(backbone: Optional[str] = None, use_gpcn: bool = True,
                 epochs: Optional[int] = None, results_dir: str = 'results',
                 data_root: Optional[str] = None, save_dir: Optional[str] = None) -> List[Dict]:
    """Multi-magnification fusion model (MultiMagnificationGPCNViT) over all 4
    magnifications of the same slide. Saved with magnification tag 'MULTI'."""
    from multimag import train_multimag
    config = _base_config(backbone, use_gpcn, epochs, data_root, save_dir)
    config.experiment_name = "gpcn_multimag"
    out = train_multimag(config)
    record = {
        'tag': 'gpcn_multimag',
        'backbone': config.model.backbone,
        'use_gpcn': config.model.use_gpcn,
        'magnification': 'MULTI',
        'best_val_auc': out['best_val_auc'],
        'best_epoch': out['best_epoch'],
        'metrics': {k: float(v) for k, v in out['test_metrics'].items()},
    }
    rdir = Path(results_dir); rdir.mkdir(parents=True, exist_ok=True)
    with open(rdir / 'gpcn_multimag.json', 'w') as f:
        json.dump(record, f, indent=2)
    return [record]


def run_backbone_comparison(mag: str = '40X', epochs: Optional[int] = None,
                            results_dir: str = 'results',
                            data_root: Optional[str] = None, save_dir: Optional[str] = None) -> List[Dict]:
    """ImageNet-ViT vs Phikon at one magnification, GPCN on."""
    rdir = Path(results_dir)
    backbones = {
        'imagenet': 'vit_base_patch16_224',
        'phikon': 'hf-hub:1aurent/vit_base_patch16_224.owkin_pancancer',
    }
    records = []
    for name, bb in backbones.items():
        config = _base_config(bb, True, epochs, data_root, save_dir)
        config.data.pool_all_magnifications = False
        config.data.train_magnification = mag
        config.experiment_name = f"backbone_{mag}_{name}"
        records.append(_run_one(config, config.experiment_name, rdir))
    return records


# ----------------------------------------------------------------------------
# Aggregation + tables
# ----------------------------------------------------------------------------
def load_results(paths: List[str]) -> List[Dict]:
    records = []
    for p in paths:
        for f in sorted(glob.glob(p)):
            with open(f) as fh:
                records.append(json.load(fh))
    return records


def aggregate_by_magnification(records: List[Dict]) -> Dict[str, Dict]:
    """Map magnification -> metrics dict (uses the latest record per magnification)."""
    by_mag = {}
    for r in records:
        by_mag[r['magnification']] = r['metrics']
    return by_mag


def build_results_table(records: List[Dict]) -> str:
    """Markdown table of OUR results per magnification + average row."""
    by_mag = aggregate_by_magnification(records)
    mags = [m for m in MAGNIFICATIONS if m in by_mag]
    if not mags:
        mags = list(by_mag.keys())

    header = '| Metric | ' + ' | '.join(mags) + ' | Average |'
    sep = '|' + '---|' * (len(mags) + 2)
    lines = [header, sep]
    for key, name in REPORT_KEYS:
        vals = []
        for m in mags:
            v = by_mag[m].get(key)
            vals.append(v)
        present = [v for v in vals if v is not None]
        avg = sum(present) / len(present) if present else None

        def fmt(v):
            if v is None:
                return '—'
            # accuracy/sens/spec are 0-1 here -> show as %
            if key in ('accuracy', 'sensitivity', 'specificity'):
                return f'{v*100:.2f}%'
            if key == 'decision_threshold':
                return f'{v:.3f}'
            return f'{v:.4f}'

        row = f'| {name} | ' + ' | '.join(fmt(v) for v in vals) + f' | {fmt(avg)} |'
        lines.append(row)
    return '\n'.join(lines)


def build_comparison_table(records: List[Dict], our_name: str = 'GPCN-ViT (ours)') -> str:
    """Markdown table comparing OUR per-magnification accuracy vs published SOTA."""
    by_mag = aggregate_by_magnification(records)
    cols = MAGNIFICATIONS

    def acc(m):
        d = by_mag.get(m)
        if not d or d.get('accuracy') is None:
            return None
        return d['accuracy'] * 100.0

    header = '| Method | Year | Protocol | ' + ' | '.join(cols) + ' | Avg |'
    sep = '|' + '---|' * (len(cols) + 4)
    lines = [header, sep]

    for row in PUBLISHED_SOTA:
        vals = [row.get(c) for c in cols]
        present = [v for v in vals if v is not None]
        avg = sum(present) / len(present) if present else None
        cells = ' | '.join(f'{v:.2f}' if v is not None else '—' for v in vals)
        avg_str = f'{avg:.2f}' if avg is not None else '—'
        lines.append(f"| {row['method']} | {row['year']} | {row['protocol']} | "
                     f"{cells} | {avg_str} |")

    ours = [acc(c) for c in cols]
    present = [v for v in ours if v is not None]
    avg = sum(present) / len(present) if present else None
    cells = ' | '.join(f'**{v:.2f}**' if v is not None else '—' for v in ours)
    lines.append(f"| **{our_name}** | 2026 | image-level binary | {cells} | "
                 f"{('**%.2f**' % avg) if avg is not None else '—'} |")

    note = ("\n\n> ⚠️ Published numbers are pre-filled from commonly cited works and "
            "tagged [VERIFY]; confirm each against the original paper (protocols differ) "
            "before manuscript submission. Sources:\n")
    for row in PUBLISHED_SOTA:
        note += f">  - {row['method']}: {row['source']}\n"
    return '\n'.join(lines) + note


def write_csv(records: List[Dict], path: str = 'results/summary.csv'):
    import csv
    by_mag = aggregate_by_magnification(records)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    keys = [k for k, _ in REPORT_KEYS]
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['magnification'] + keys)
        for m, metrics in by_mag.items():
            w.writerow([m] + [metrics.get(k, '') for k in keys])
    logger.info(f"✓ Wrote {path}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='GPCN-ViT experiment driver')
    ap.add_argument('--mode', required=True,
                    choices=['per_mag', 'pooled', 'multimag', 'ablation', 'backbone', 'table'])
    ap.add_argument('--backbone', default=None,
                    help="timm backbone or hf-hub id (default: config default = Phikon)")
    ap.add_argument('--mag', default='40X', help='magnification for ablation/backbone modes')
    ap.add_argument('--seeds', type=int, nargs='*', default=[0, 1, 2],
                    help='seeds for --mode ablation (paired significance across seeds)')
    ap.add_argument('--epochs', type=int, default=None, help='override num_epochs')
    ap.add_argument('--results', nargs='*', default=['results/*.json'],
                    help='glob(s) of result JSONs for --mode table')
    ap.add_argument('--results_dir', default='results')
    args = ap.parse_args()

    if args.mode == 'per_mag':
        records = run_all_magnifications(args.backbone, epochs=args.epochs, results_dir=args.results_dir)
    elif args.mode == 'pooled':
        records = run_pooled(args.backbone, epochs=args.epochs, results_dir=args.results_dir)
    elif args.mode == 'multimag':
        records = run_multimag(args.backbone, epochs=args.epochs, results_dir=args.results_dir)
    elif args.mode == 'ablation':
        records = run_ablation(args.backbone, args.mag, epochs=args.epochs,
                               results_dir=args.results_dir, seeds=tuple(args.seeds))
    elif args.mode == 'backbone':
        records = run_backbone_comparison(args.mag, epochs=args.epochs, results_dir=args.results_dir)
    else:  # table
        records = load_results(args.results)

    print('\n\n' + '=' * 70 + '\nRESULTS TABLE\n' + '=' * 70)
    print(build_results_table(records))
    print('\n\n' + '=' * 70 + '\nSOTA COMPARISON\n' + '=' * 70)
    print(build_comparison_table(records))
    try:
        write_csv(records, str(Path(args.results_dir) / 'summary.csv'))
    except Exception as e:
        logger.warning(f"CSV write skipped: {e}")


if __name__ == '__main__':
    main()
