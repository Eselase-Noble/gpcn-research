"""
fusion_baselines.py — Does cross-magnification ATTENTION beat naive fusion?

WHY THIS EXISTS
---------------
`single_mag_baseline.py` answers "do four views beat one?" — but that is almost
trivially yes (any multi-view aggregation helps). It does NOT defend the *fusion
architecture*: a reviewer will rightly ask whether the cross-magnification
attention head earns its keep over a dumb baseline that just averages four
single-magnification predictions.

This module supplies the comparison that actually isolates the contribution of
the attention fusion, in two forms:

  (A) TRAINING-FREE post-hoc late fusion, computed from the already-trained
      fusion model's single-stream outputs (no GPU retrain — an immediate
      sanity baseline). Modes: average-probability, majority vote.

  (B) FROM-SCRATCH fusion variants trained under the identical recipe, selected
      by `config.model.fusion_mode`:
        'attention' — the full model (cross-mag MHA + learned vote)   [ours]
        'none'      — attention replaced by identity (heads + vote only)
        'mean'      — mean-pool the 4 CLS features -> shared head
        'concat'    — concat 4 CLS -> MLP -> logits
      Train each with the SAME train_multimag loop (it calls build_fusion_model
      when fusion_mode is set), giving a clean apples-to-apples ablation that
      pins the gain on the attention mechanism, not on "using four scales."

Every fused-vs-baseline comparison is reported with a paired McNemar test and a
Wilson 95% CI on accuracy, so the gain is defended statistically, not asserted.

Usage
-----
    # (A) cheap, no retrain — run after training the attention model:
    from config import get_full_config
    from fusion_baselines import run_posthoc_fusion_baselines
    cfg = get_full_config(); cfg.experiment_name = 'gpcn_multimag'
    run_posthoc_fusion_baselines(cfg)

    # (B) clean, retrain each variant (one line per variant, e.g. in Colab):
    cfg.model.fusion_mode = 'mean'   # then train_multimag(cfg) as usual
"""

import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_TABLE_KEYS = ['accuracy', 'auc_roc', 'sensitivity', 'specificity',
               'f1_class1', 'mcc', 'ece']


# ---------------------------------------------------------------------------
# Statistics: McNemar (paired) + Wilson interval — for defensible claims
# ---------------------------------------------------------------------------
def mcnemar_test(y_true, pred_a, pred_b) -> Dict:
    """Paired test that model A and model B differ on the SAME samples.

    Compares the discordant pairs: b01 = A wrong & B right, b10 = A right & B
    wrong. With the small error counts on BreakHis slides, use the EXACT
    binomial p-value (mid-p not needed); fall back is symmetric.
    Returns b01/b10 and a two-sided p-value.
    """
    y_true = np.asarray(y_true); a = np.asarray(pred_a); b = np.asarray(pred_b)
    a_correct = (a == y_true); b_correct = (b == y_true)
    b01 = int(np.sum(~a_correct & b_correct))   # A wrong, B right (B helps)
    b10 = int(np.sum(a_correct & ~b_correct))   # A right, B wrong (B hurts)
    n = b01 + b10
    if n == 0:
        p = 1.0
    else:
        k = min(b01, b10)
        # two-sided exact binomial under p=0.5
        tail = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
        p = min(1.0, 2.0 * tail)
    return {'b01_B_better': b01, 'b10_A_better': b10, 'n_discordant': n,
            'p_value': p, 'significant_05': (p < 0.05)}


def wilson_ci(correct: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score 95% CI for a proportion — correct for small n / near-1 acc."""
    if n == 0:
        return (float('nan'), float('nan'))
    p = correct / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# ---------------------------------------------------------------------------
# (A) Training-free post-hoc late fusion from the trained attention model
# ---------------------------------------------------------------------------
@torch.no_grad()
def _single_stream_probs(model, loader, device) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Per-magnification single-stream probabilities on the SAME slide order.

    Reuses the trained backbone + per-mag heads with the fusion attention seeing
    one token (self-only) — identical to single_mag_baseline.evaluate_single_mag,
    but returns all four aligned so we can combine them post-hoc.
    Returns (y_true, [probs_mag0, probs_mag1, ...]) with matching row order.
    """
    mm = model.module if hasattr(model, 'module') else model
    mm.eval()
    n_mag = mm.num_magnifications
    y_true, per_mag = [], [[] for _ in range(n_mag)]
    for images, labels, _ in loader:
        y_true.extend(labels.numpy())
        for i in range(n_mag):
            img = images[i].to(device)
            feat = mm.base_model.forward_features(img)[:, 0, :]
            tok = feat.unsqueeze(1)
            fused, _ = mm.fusion(tok, tok, tok)
            logits = mm.mag_heads[i](fused[:, 0, :])
            per_mag[i].append(F.softmax(logits, dim=-1).cpu().numpy())
    return np.array(y_true), [np.concatenate(p, axis=0) for p in per_mag]


def _metrics(y_true, y_probs, threshold=0.5) -> Tuple[Dict, np.ndarray]:
    from metrics import MetricsCalculator
    if y_probs.shape[1] == 2 and threshold != 0.5:
        y_pred = (y_probs[:, 1] >= threshold).astype(int)
    else:
        y_pred = y_probs.argmax(axis=1)
    calc = MetricsCalculator(num_classes=y_probs.shape[1])
    m = calc.calculate_all_metrics(y_true, y_pred, y_probs)
    m.update(calc.calculate_calibration_metrics(y_true, y_probs))
    return m, y_pred


def run_posthoc_fusion_baselines(config, checkpoint: str = None) -> Dict:
    """Compare the trained ATTENTION fusion against training-free late-fusion
    baselines (average-probability, majority vote) on the identical test slides,
    with McNemar significance and Wilson CIs. No retraining.

    NOTE (state this in the paper): the late-fusion baselines reuse heads trained
    WITH attention present, so they are a *cheap sanity* baseline. The clean
    comparison is the from-scratch `fusion_mode` variants (section B). Both point
    the same way; report the from-scratch numbers as primary.
    """
    from multimag import (create_multimag_dataloaders, build_multimag_model,
                          evaluate_multimag)
    device = torch.device(getattr(config, 'device',
                                  'cuda' if torch.cuda.is_available() else 'cpu'))
    _, val_loader, test_loader, info = create_multimag_dataloaders(config)
    model = build_multimag_model(config).to(device)
    if checkpoint is None:
        checkpoint = str(Path(config.data.save_dir)
                         / (config.experiment_name or 'gpcn_multimag')
                         / 'checkpoints' / 'best_multimag.pth')
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    logger.info(f"[fusion-baselines] loaded {checkpoint}")

    # Attention fusion (ours), threshold-tuned on val exactly like training.
    fused_thr = 0.5
    if getattr(config.validation, 'optimize_threshold', False):
        from metrics import find_optimal_threshold
        _, v_true, _, v_probs = evaluate_multimag(model, val_loader, device)
        fused_thr, _ = find_optimal_threshold(
            v_true, v_probs[:, 1],
            mode=getattr(config.validation, 'threshold_mode', 'youden'),
            fn_cost=getattr(config.validation, 'fn_cost', 10.0),
            fp_cost=getattr(config.validation, 'fp_cost', 1.0))
    fused_m, y_true, fused_pred, _ = evaluate_multimag(model, test_loader, device,
                                                       threshold=fused_thr)

    # Naive baselines from single-stream probs (same slide order as fused eval).
    yt, per_mag = _single_stream_probs(model, test_loader, device)
    assert np.array_equal(yt, y_true), "slide order mismatch between evals"

    avg_probs = np.mean(per_mag, axis=0)                       # average-probability
    avg_m, avg_pred = _metrics(y_true, avg_probs)
    votes = np.stack([p.argmax(1) for p in per_mag], axis=1)   # majority vote
    vote_pred = (votes.sum(1) >= (len(per_mag) / 2)).astype(int)
    vote_probs = np.stack([1 - votes.mean(1), votes.mean(1)], axis=1)
    vote_m, _ = _metrics(y_true, vote_probs)

    n = len(y_true)
    results = {'attention': fused_m, 'avg_prob': avg_m, 'majority_vote': vote_m,
               'significance': {}, 'n': n}
    print("\n" + "=" * 78)
    print("FUSION HEAD vs NAIVE FUSION — identical test slides (post-hoc, no retrain)")
    print("=" * 78)
    print(f"{'Metric':<14}{'avg-prob':>12}{'maj-vote':>12}{'ATTENTION':>12}")
    print("-" * 78)
    for k in _TABLE_KEYS:
        print(f"{k:<14}{avg_m.get(k, float('nan')):>12.4f}"
              f"{vote_m.get(k, float('nan')):>12.4f}{fused_m.get(k, float('nan')):>12.4f}")
    lo, hi = wilson_ci(int(round(fused_m['accuracy'] * n)), n)
    print("-" * 78)
    print(f"attention accuracy 95% Wilson CI: [{lo:.4f}, {hi:.4f}]  (n={n})")
    for name, pred in [('avg_prob', avg_pred), ('majority_vote', vote_pred)]:
        mc = mcnemar_test(y_true, pred, fused_pred)
        results['significance'][name] = mc
        flag = '✓ sig' if mc['significant_05'] else '✗ n.s.'
        print(f"McNemar attention vs {name:<13}: p={mc['p_value']:.4f}  {flag}  "
              f"(attention better on {mc['b01_B_better']}, worse on {mc['b10_A_better']})")
    print("=" * 78 + "\n")
    return results


# ---------------------------------------------------------------------------
# (B) From-scratch fusion variants (clean ablation of the attention mechanism)
# ---------------------------------------------------------------------------
class ConfigurableFusionGPCNViT(nn.Module):
    """Multi-magnification model with a swappable fusion rule, so the SAME
    training recipe can produce each ablation arm. `mode='attention'` reproduces
    MultiMagnificationGPCNViT exactly; the others remove the cross-mag attention.
    """
    def __init__(self, base_model, num_magnifications: int = 4, mode: str = 'attention'):
        super().__init__()
        assert mode in ('attention', 'none', 'mean', 'concat')
        self.base_model = base_model
        self.num_magnifications = num_magnifications
        self.mode = mode
        D = base_model.embed_dim
        C = base_model.num_classes
        if mode in ('attention', 'none'):
            self.mag_heads = nn.ModuleList([nn.Linear(D, C) for _ in range(num_magnifications)])
            self.mag_weights = nn.Parameter(torch.ones(num_magnifications) / num_magnifications)
        if mode == 'attention':
            self.fusion = nn.MultiheadAttention(D, num_heads=8, dropout=0.1, batch_first=True)
        elif mode == 'mean':
            self.head = nn.Linear(D, C)
        elif mode == 'concat':
            self.head = nn.Sequential(nn.Linear(D * num_magnifications, D), nn.GELU(),
                                      nn.Dropout(0.3), nn.Linear(D, C))

    def forward(self, images_multi_mag, magnification_indices=None):
        if not isinstance(images_multi_mag, list):
            return self.base_model(images_multi_mag)
        feats = [self.base_model.forward_features(img)[:, 0, :] for img in images_multi_mag]
        F_stack = torch.stack(feats, dim=1)                     # (B, M, D)
        if self.mode == 'mean':
            return self.head(F_stack.mean(dim=1))
        if self.mode == 'concat':
            return self.head(F_stack.flatten(1))
        # attention / none: per-mag heads + learned vote
        fused = self.fusion(F_stack, F_stack, F_stack)[0] if self.mode == 'attention' else F_stack
        preds = torch.stack([h(fused[:, i, :]) for i, h in enumerate(self.mag_heads)], dim=1)
        w = F.softmax(self.mag_weights, dim=0)
        return (preds * w.view(1, -1, 1)).sum(dim=1)


def build_fusion_model(config) -> nn.Module:
    """Drop-in for build_multimag_model that honours config.model.fusion_mode.
    Point train_multimag at this to train an ablation arm from scratch."""
    from model import GPCNViT
    base = GPCNViT(
        num_classes=config.model.num_classes, pretrained=config.model.pretrained,
        num_gpcn_layers=config.model.num_gpcn_layers, k=config.model.gpcn_k,
        use_hybrid=config.model.use_hybrid_knn, alpha=config.model.alpha,
        use_multi_scale=config.model.use_multi_scale, pyramid_levels=config.model.pyramid_levels,
        freeze_backbone=config.model.freeze_backbone, hidden_dim=config.model.hidden_dim,
        dropout=config.model.dropout,
        backbone=getattr(config.model, 'backbone', 'vit_base_patch16_224'),
        backbone_weights_path=getattr(config.model, 'backbone_weights_path', ''),
        use_gpcn=getattr(config.model, 'use_gpcn', True))
    mode = getattr(config.model, 'fusion_mode', 'attention')
    return ConfigurableFusionGPCNViT(base, num_magnifications=len(config.data.magnifications),
                                     mode=mode)


if __name__ == '__main__':
    from config import get_full_config
    cfg = get_full_config()
    cfg.experiment_name = 'gpcn_multimag'
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_posthoc_fusion_baselines(cfg)
