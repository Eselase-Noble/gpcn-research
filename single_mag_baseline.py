"""
single_mag_baseline.py — Fair fused-vs-single-magnification comparison for GPCN-ViT.

WHY THIS EXISTS
---------------
The per-magnification table in the paper is evaluated PER IMAGE (thousands of
images, one model per scale), while the multi-magnification result is evaluated
PER SLIDE-SAMPLE (one image drawn from each of the four magnifications of the
same slide). Comparing 92.55% (per-image avg) against 98.54% (per-slide fused)
mixes two different sampling units and overstates the gain.

This module fixes that by reporting a SINGLE-MAGNIFICATION baseline on the
IDENTICAL slide-level test set used by the fused model. It is a single-stream
ABLATION of the *already trained* fusion model:

    for one magnification m:
        feat   = base_model.forward_features(img_m)[:, 0]   # shared backbone, CLS
        fused  = fusion(feat, feat, feat)                   # self-attn over 1 token
        logits = mag_heads[m](fused)                        # that scale's own head

Every weight is the trained one; the fusion attention sees a single token, so it
can only attend to itself — the ONLY thing removed versus the full model is the
other magnifications' information. Run on the same test loader (same N, same
slides), fused-vs-single is now apples-to-apples and isolates exactly what the
cross-magnification fusion contributes.

It loads best_multimag.pth, so you do NOT retrain — it just re-evaluates.

Usage
-----
    from config import get_full_config
    from single_mag_baseline import run_single_mag_baseline
    cfg = get_full_config(); cfg.experiment_name = 'gpcn_multimag'
    table = run_single_mag_baseline(cfg)      # prints comparison, returns dict
"""

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from multimag import create_multimag_dataloaders, build_multimag_model, evaluate_multimag

logger = logging.getLogger(__name__)

# Metrics shown in the comparison table (higher is better except ECE).
_TABLE_KEYS = ['accuracy', 'auc_roc', 'sensitivity', 'specificity',
               'f1_class1', 'mcc', 'ece']


@torch.no_grad()
def evaluate_single_mag(model, loader, device, mag_index: int, threshold: float = 0.5):
    """Single-stream ablation for one magnification on the slide-level set."""
    from metrics import MetricsCalculator
    mm = model.module if hasattr(model, 'module') else model
    mm.eval()
    all_true, all_probs = [], []
    for images, labels, _ in loader:
        img = images[mag_index].to(device)
        feat = mm.base_model.forward_features(img)[:, 0, :]      # (B, D) CLS token
        tok = feat.unsqueeze(1)                                  # (B, 1, D)
        fused, _ = mm.fusion(tok, tok, tok)                      # self-attn over 1 token
        logits = mm.mag_heads[mag_index](fused[:, 0, :])
        probs = F.softmax(logits, dim=-1)
        all_true.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_true)
    y_probs = np.array(all_probs)
    if y_probs.shape[1] == 2 and threshold != 0.5:
        y_pred = (y_probs[:, 1] >= threshold).astype(int)
    else:
        y_pred = y_probs.argmax(axis=1)

    calc = MetricsCalculator(num_classes=y_probs.shape[1])
    metrics = calc.calculate_all_metrics(y_true, y_pred, y_probs)
    metrics.update(calc.calculate_calibration_metrics(y_true, y_probs))
    return metrics, y_true, y_probs


def _tuned_threshold(model, val_loader, device, config, mag_index: int) -> float:
    """Tune a single-mag decision threshold on val, mirroring the fused model so
    the comparison is fair on the decision rule too."""
    if not getattr(config.validation, 'optimize_threshold', False):
        return 0.5
    from metrics import find_optimal_threshold
    _, v_true, v_probs = evaluate_single_mag(model, val_loader, device, mag_index)
    thr, _ = find_optimal_threshold(
        v_true, v_probs[:, 1],
        mode=getattr(config.validation, 'threshold_mode', 'youden'),
        fn_cost=getattr(config.validation, 'fn_cost', 10.0),
        fp_cost=getattr(config.validation, 'fp_cost', 1.0))
    return thr


def _print_table(fused: Dict, single: Dict[str, Dict]):
    mags = list(single.keys())
    width = 14 + 10 * (len(mags) + 1)
    print("\n" + "=" * width)
    print("FUSED vs SINGLE-MAGNIFICATION — identical slide-level test set")
    print("=" * width)
    print(f"{'Metric':<14}" + "".join(f"{m:>10}" for m in mags) + f"{'Fused':>10}")
    print("-" * width)
    for k in _TABLE_KEYS:
        row = f"{k:<14}"
        for m in mags:
            row += f"{single[m].get(k, float('nan')):>10.4f}"
        row += f"{fused.get(k, float('nan')):>10.4f}"
        print(row)
    # Single-mag average vs fused on accuracy/mcc, the two headline numbers.
    avg_acc = np.mean([single[m]['accuracy'] for m in mags])
    avg_mcc = np.mean([single[m]['mcc'] for m in mags])
    print("-" * width)
    print(f"single-mag average:   accuracy {avg_acc:.4f}   mcc {avg_mcc:.4f}")
    print(f"fusion gain:          accuracy {fused['accuracy'] - avg_acc:+.4f}   "
          f"mcc {fused['mcc'] - avg_mcc:+.4f}")
    print("=" * width + "\n")


def run_single_mag_baseline(config, checkpoint: str = None) -> Dict:
    """Load the trained fusion model and report fused vs per-magnification on the
    identical slide-level test set. Returns
        {'fused': {...}, 'single': {mag: {...}}, 'single_avg': {...}}.
    """
    device = torch.device(getattr(config, 'device',
                                  'cuda' if torch.cuda.is_available() else 'cpu'))
    mags = list(config.data.magnifications)

    _, val_loader, test_loader, info = create_multimag_dataloaders(config)
    logger.info(f"[baseline] test slides={info['test_slides']} "
                f"samples={info['test_samples']}")

    model = build_multimag_model(config).to(device)
    if checkpoint is None:
        checkpoint = str(Path(config.data.save_dir)
                         / (config.experiment_name or 'gpcn_multimag')
                         / 'checkpoints' / 'best_multimag.pth')
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    logger.info(f"[baseline] loaded {checkpoint} (epoch {ckpt.get('epoch', '?')})")

    # Fused result, threshold-tuned on val (same recipe as training).
    fused_thr = 0.5
    if getattr(config.validation, 'optimize_threshold', False):
        from metrics import find_optimal_threshold
        _, v_true, _, v_probs = evaluate_multimag(model, val_loader, device)
        fused_thr, _ = find_optimal_threshold(
            v_true, v_probs[:, 1],
            mode=getattr(config.validation, 'threshold_mode', 'youden'),
            fn_cost=getattr(config.validation, 'fn_cost', 10.0),
            fp_cost=getattr(config.validation, 'fp_cost', 1.0))
    fused, *_ = evaluate_multimag(model, test_loader, device, threshold=fused_thr)
    fused['decision_threshold'] = fused_thr

    # Single-magnification baselines, each tuned on val the same way.
    single: Dict[str, Dict] = {}
    for i, mag in enumerate(mags):
        thr = _tuned_threshold(model, val_loader, device, config, i)
        m, *_ = evaluate_single_mag(model, test_loader, device, i, threshold=thr)
        m['decision_threshold'] = thr
        single[mag] = m

    single_avg = {k: float(np.mean([single[m][k] for m in mags])) for k in _TABLE_KEYS}

    # Learned magnification weights — softmax(mag_weights) is the trust the fusion
    # model places on each scale in the final weighted vote (model.py:399-400).
    # This corroborates which streams the ablation finds strongest.
    mm = model.module if hasattr(model, 'module') else model
    mag_w = F.softmax(mm.mag_weights.detach().float().cpu(), dim=0).numpy()
    learned_weights = {mag: float(w) for mag, w in zip(mags, mag_w)}
    print("Learned magnification weights  softmax(mag_weights) — fusion trust per scale:")
    for mag, w in learned_weights.items():
        bar = '#' * int(round(w * 50))
        print(f"  {mag:>6}: {w:.4f}  {bar}")
    print()

    _print_table(fused, single)
    return {'fused': fused, 'single': single, 'single_avg': single_avg,
            'learned_weights': learned_weights}


if __name__ == '__main__':
    from config import get_full_config
    cfg = get_full_config()
    cfg.experiment_name = 'gpcn_multimag'
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_single_mag_baseline(cfg)
