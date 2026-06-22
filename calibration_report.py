"""
calibration_report.py — Reliability diagrams + temperature scaling for GPCN-ViT.

WHY THIS EXISTS
---------------
A clinical model must not just be accurate, its probabilities must MEAN
something: when it says 0.9 it should be right ~90% of the time. Reviewers ask
for (a) a reliability diagram and (b) post-hoc calibration (temperature scaling)
with before/after ECE & MCE. This module produces exactly that, reusing the
project's existing `TemperatureScaler` (LBFGS-on-val, in metrics.py) and the
SAME ECE/MCE definition used elsewhere so the numbers are consistent.

WHAT IT DOES (no test peeking)
------------------------------
1. Collect TEST-set logits from a trained model.
2. Fit a single scalar temperature T on the VALIDATION set only (LBFGS on NLL).
3. Recompute TEST probabilities as softmax(logits / T).
4. Compute ECE & MCE before and after, and plot side-by-side reliability
   diagrams (with a confidence histogram), annotated with T, ECE, MCE.

Two reliability conventions, picked automatically by class count:
  * binary (num_classes == 2): positive-class reliability — x = P(malignant),
    y = empirical malignant rate. Matches metrics.calculate_calibration_metrics.
  * multi-class (BACH, K>2): top-label reliability — x = max softmax prob,
    y = top-1 accuracy. Matches bach._multiclass_ece.

Usage (notebook):
    from calibration_report import temperature_calibration_report
    rep = temperature_calibration_report(model, val_loader, test_loader, device,
                                         save_path=f'{OUTPUT_PATH}/reliability.png',
                                         class_names=['Benign','Malignant'])
    print(rep)   # {'temperature', 'ece_before','mce_before','ece_after','mce_after', ...}
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Reliability statistics (single source of truth for the curve AND the ECE/MCE)
# ----------------------------------------------------------------------------
def reliability_curve(y_true: np.ndarray, y_probs: np.ndarray,
                      mode: str = 'auto', n_bins: int = 15
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Return (bin_centers, bin_acc, bin_conf, bin_count, ece, mce).

    mode:
      'positive'  — binary, confidence = P(class 1); accuracy = fraction class 1.
      'toplabel'  — multi-class, confidence = max prob; accuracy = top-1 correct.
      'auto'      — 'positive' if y_probs is 1-D or has 2 columns, else 'toplabel'.
    Empty bins get NaN for acc/conf (so they are skipped when plotting).
    """
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    if mode == 'auto':
        mode = 'positive' if (y_probs.ndim == 1 or y_probs.shape[1] == 2) else 'toplabel'

    if mode == 'positive':
        conf = y_probs if y_probs.ndim == 1 else y_probs[:, 1]
        correct = (y_true == 1).astype(float)  # "accuracy" = empirical positive rate
    elif mode == 'toplabel':
        conf = y_probs.max(axis=1)
        correct = (y_probs.argmax(axis=1) == y_true).astype(float)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    acc = np.full(n_bins, np.nan)
    avg_conf = np.full(n_bins, np.nan)
    count = np.zeros(n_bins, dtype=int)
    ece = 0.0
    mce = 0.0
    n = len(conf)
    for i in range(n_bins):
        in_bin = (conf >= edges[i]) & (conf < edges[i + 1])
        # include the right endpoint 1.0 in the last bin
        if i == n_bins - 1:
            in_bin = in_bin | (conf == 1.0)
        c = int(in_bin.sum())
        count[i] = c
        if c > 0:
            a = float(correct[in_bin].mean())
            cf = float(conf[in_bin].mean())
            acc[i] = a
            avg_conf[i] = cf
            gap = abs(cf - a)
            ece += gap * (c / n)
            mce = max(mce, gap)
    return centers, acc, avg_conf, count, float(ece), float(mce)


# ----------------------------------------------------------------------------
# Logit collection + temperature fitting
# ----------------------------------------------------------------------------
@torch.no_grad()
def _collect_logits(model, loader, device) -> Tuple[np.ndarray, torch.Tensor]:
    """Run the model over a loader and return (y_true [N], logits [N,K] on CPU)."""
    model.eval()
    ys, logits = [], []
    for batch in loader:
        images = batch[0].to(device)
        labels = batch[1]
        out = model(images)
        if isinstance(out, tuple):
            out = out[0]
        logits.append(out.detach().float().cpu())
        ys.append(labels.detach().cpu() if torch.is_tensor(labels) else torch.as_tensor(labels))
    return torch.cat(ys).numpy(), torch.cat(logits)


def fit_temperature(model, val_loader, device) -> float:
    """Fit a single temperature on the validation set (LBFGS on NLL). Reuses the
    project's TemperatureScaler so behaviour matches the rest of the codebase."""
    from metrics import TemperatureScaler
    scaler = TemperatureScaler().to(device)
    scaler.set_temperature(model, val_loader, device)
    return float(scaler.temperature.detach().cpu().item())


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def plot_reliability(ax, centers, acc, count, ece, mce, title: str, n_bins: int = 15):
    """Draw a reliability diagram on `ax`: observed accuracy vs confidence bars,
    the y=x identity line, and gap shading. Annotates ECE/MCE."""
    width = 1.0 / n_bins
    valid = ~np.isnan(acc)
    # Gap (red, behind) then accuracy (blue, front) — the classic guo-et-al style.
    ax.bar(centers[valid], acc[valid], width=width, edgecolor='black', linewidth=0.5,
           color='#4C72B0', label='Observed', zorder=2)
    gap = np.where(valid, np.maximum(centers - acc, 0), 0.0)
    ax.bar(centers[valid], gap[valid], width=width, bottom=acc[valid],
           color='#DD8452', alpha=0.55, edgecolor='#A0522D', linewidth=0.5,
           hatch='//', label='Gap', zorder=1)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect calibration', zorder=3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel('Confidence'); ax.set_ylabel('Accuracy / observed rate')
    ax.set_title(title)
    ax.legend(loc='upper left', fontsize=8)
    ax.text(0.97, 0.05, f'ECE = {ece:.4f}\nMCE = {mce:.4f}',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.85))


def temperature_calibration_report(model, val_loader, test_loader, device,
                                   save_path: Optional[str] = None,
                                   n_bins: int = 15,
                                   class_names: Optional[List[str]] = None,
                                   mode: str = 'auto',
                                   show: bool = True) -> Dict[str, float]:
    """Full before/after temperature-scaling calibration report on the TEST set.

    Fits T on `val_loader` only, applies it to `test_loader`, computes ECE/MCE
    before and after, and renders side-by-side reliability diagrams (+ a
    confidence histogram). Returns the numbers and saves the figure if asked.
    """
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    device = torch.device(device) if not isinstance(device, torch.device) else device

    # 1) test logits, 2) T on val, 3) post-scaling probs.
    y_true, logits = _collect_logits(model, test_loader, device)
    T = fit_temperature(model, val_loader, device)
    probs_before = F.softmax(logits, dim=-1).numpy()
    probs_after = F.softmax(logits / T, dim=-1).numpy()

    if mode == 'auto':
        mode = 'positive' if logits.shape[1] == 2 else 'toplabel'

    cb, ab, conf_b, cnt_b, ece_b, mce_b = reliability_curve(y_true, probs_before, mode, n_bins)
    ca, aa, conf_a, cnt_a, ece_a, mce_a = reliability_curve(y_true, probs_after, mode, n_bins)

    conv = 'positive-class P(malignant)' if mode == 'positive' else 'top-label'
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    plot_reliability(axes[0], cb, ab, cnt_b, ece_b, mce_b,
                     f'Before temperature scaling\n({conv})', n_bins)
    plot_reliability(axes[1], ca, aa, cnt_a, ece_a, mce_a,
                     f'After temperature scaling (T = {T:.3f})\n({conv})', n_bins)
    fig.suptitle('Reliability diagram — calibration before vs after', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"✓ Saved reliability diagram to {save_path}")
    if show:
        try:
            plt.show()
        except Exception:
            pass
    else:
        plt.close(fig)

    report = {
        'temperature': T,
        'ece_before': ece_b, 'mce_before': mce_b,
        'ece_after': ece_a, 'mce_after': mce_a,
        'ece_reduction': ece_b - ece_a, 'mce_reduction': mce_b - mce_a,
        'mode': mode, 'n_test': int(len(y_true)), 'save_path': save_path,
    }
    logger.info("Calibration report: T=%.4f | ECE %.4f->%.4f | MCE %.4f->%.4f",
                T, ece_b, ece_a, mce_b, mce_a)
    return report


# ----------------------------------------------------------------------------
# Convenience: build loaders + load a trained checkpoint, then report.
# ----------------------------------------------------------------------------
def report_from_checkpoint(config, checkpoint_path: str,
                           save_path: Optional[str] = None,
                           show: bool = True) -> Dict[str, float]:
    """Load a single-image GPCN-ViT checkpoint and run the calibration report on
    the standard patient-level split for the configured magnification/protocol."""
    from model import create_model
    from dataset import create_dataloaders
    from utils import load_checkpoint

    device = torch.device(config.device)
    _, val_loader, test_loader, _ = create_dataloaders(config)
    model = create_model(config).to(device)
    try:
        load_checkpoint(checkpoint_path, model)
    except Exception:
        ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state = ck.get('model_state_dict', ck)
        (model.base_model if hasattr(model, 'base_model') else model).load_state_dict(state)
    logger.info(f"✓ Loaded checkpoint {checkpoint_path}")
    return temperature_calibration_report(model, val_loader, test_loader, device,
                                          save_path=save_path, show=show)
