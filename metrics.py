"""
Full Metrics and Evaluation Module with Calibration and Uncertainty Support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, average_precision_score,
    matthews_corrcoef, brier_score_loss
)
from typing import Dict, Tuple, Optional


# ============================================================
# 1️⃣ Temperature Scaling for Post-hoc Calibration
# ============================================================
class TemperatureScaler(nn.Module):
    """Temperature scaling for post-hoc calibration"""

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temperature

    def set_temperature(self, model, val_loader, device):
        """Tune temperature on validation set"""
        model.eval()
        logits_list, labels_list = [], []

        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 2:
                    images, labels = batch
                else:
                    images, labels = batch[0], batch[1]

                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                logits_list.append(outputs)
                labels_list.append(labels)

        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)

        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = F.cross_entropy(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval)
        print(f"[TemperatureScaler] Optimal temperature: {self.temperature.item():.4f}")
        return self


# ============================================================
# 2️⃣ Metrics Calculator
# ============================================================
class MetricsCalculator:
    """Compute classification, calibration, and clinical metrics"""

    def __init__(self, num_classes: int = 2):
        self.num_classes = num_classes

    def calculate_all_metrics(self, y_true, y_pred, y_probs) -> Dict[str, float]:
        metrics = {}
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)

        if self.num_classes == 2:
            metrics['precision_class1'] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
            metrics['recall_class1'] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
            metrics['f1_class1'] = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

            metrics['sensitivity'] = metrics['recall_class1']
            metrics['specificity'] = recall_score(y_true, y_pred, pos_label=0, zero_division=0)

            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_probs[:, 1])
            except:
                metrics['auc_roc'] = 0.0

            try:
                metrics['auc_pr'] = average_precision_score(y_true, y_probs[:, 1])
            except:
                metrics['auc_pr'] = 0.0

            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            metrics['true_positives'] = int(tp)

            if fp > 0 and fn > 0:
                metrics['diagnostic_odds_ratio'] = (tp * tn) / (fp * fn)
            else:
                metrics['diagnostic_odds_ratio'] = float('inf')

            metrics['clinical_cost'] = 10.0 * fn + 1.0 * fp
            metrics['brier_score'] = brier_score_loss(y_true, y_probs[:, 1])

        return metrics

    # --------------------------
    # Calibration Metrics
    # --------------------------
    def calculate_calibration_metrics(self, y_true, y_probs, n_bins=15):
        if y_probs.ndim == 2:
            y_probs = y_probs[:, 1]

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        mce = 0.0

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            in_bin = (y_probs >= bin_lower) & (y_probs < bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence = y_probs[in_bin].mean()
                error = abs(avg_confidence - accuracy_in_bin)
                ece += error * prop_in_bin
                mce = max(mce, error)

        return {'ece': float(ece), 'mce': float(mce)}


# ============================================================
# 3️⃣ Uncertainty Analysis
# ============================================================
class UncertaintyAnalyzer:
    """Analyze epistemic & aleatoric uncertainty"""

    def analyze_uncertainty(self,
                            y_true,
                            y_pred,
                            epistemic_uncertainty,
                            aleatoric_uncertainty: Optional[np.ndarray] = None) -> Dict[str, float]:

        correct = (y_true == y_pred).astype(float)
        from scipy.stats import spearmanr

        uncertainty_error_correlation, _ = spearmanr(epistemic_uncertainty, 1 - correct)

        correct_mask = correct == 1
        incorrect_mask = correct == 0

        avg_uncertainty_correct = epistemic_uncertainty[correct_mask].mean() if correct_mask.sum() > 0 else 0
        avg_uncertainty_incorrect = epistemic_uncertainty[incorrect_mask].mean() if incorrect_mask.sum() > 0 else 0

        metrics = {
            'uncertainty_error_correlation': float(uncertainty_error_correlation),
            'avg_uncertainty_correct': float(avg_uncertainty_correct),
            'avg_uncertainty_incorrect': float(avg_uncertainty_incorrect),
            'uncertainty_separation': float(avg_uncertainty_incorrect - avg_uncertainty_correct)
        }

        if aleatoric_uncertainty is not None:
            metrics['avg_aleatoric_uncertainty'] = float(aleatoric_uncertainty.mean())
            metrics['avg_epistemic_uncertainty'] = float(epistemic_uncertainty.mean())

        return metrics

    def get_uncertain_samples(self, uncertainty: np.ndarray, threshold: float = 0.5, top_k: Optional[int] = None):
        if top_k is not None:
            return np.argsort(uncertainty)[-top_k:]
        return np.where(uncertainty > threshold)[0]


# ============================================================
# 4️⃣ Evaluate Model
# ============================================================
def _logits_to_probs(model, images, scaler):
    """Run a plain (non-uncertainty) forward pass and return softmax probs."""
    outputs = model(images)
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    if scaler is not None:
        outputs = scaler(outputs)
    return F.softmax(outputs, dim=-1)


def _tta_probs(model, images, scaler):
    """Average softmax probs over the image plus horizontal/vertical flips.

    Histology is largely orientation-invariant, so flip-TTA is a safe,
    cheap way to reduce variance at inference time.
    """
    views = [images, torch.flip(images, dims=[3]), torch.flip(images, dims=[2])]
    probs = sum(_logits_to_probs(model, v, scaler) for v in views) / len(views)
    return probs


def find_optimal_threshold(y_true: np.ndarray,
                           y_probs_pos: np.ndarray,
                           mode: str = 'youden',
                           fn_cost: float = 10.0,
                           fp_cost: float = 1.0) -> Tuple[float, Dict[str, float]]:
    """Pick a decision threshold on a (validation) set instead of fixed 0.5.

    BreakHis is class-imbalanced and the model's high AUC but lower accuracy is a
    classic sign that 0.5 is not the best operating point. We sweep candidate
    thresholds and select one by:

      - 'youden'  : maximise Youden's J = sensitivity + specificity - 1
                    (balances sensitivity/specificity, threshold-free of priors)
      - 'cost'    : minimise clinical cost = fn_cost * FN + fp_cost * FP
                    (encodes that missing a cancer is worse than a false alarm)
      - 'f1'      : maximise F1 of the positive (malignant) class

    Args:
        y_true: ground-truth labels [N]
        y_probs_pos: predicted P(malignant) [N]
        mode: 'youden' | 'cost' | 'f1'
        fn_cost, fp_cost: costs used when mode == 'cost'

    Returns:
        best_threshold, info dict (objective value + sens/spec at that threshold)
    """
    y_true = np.asarray(y_true)
    y_probs_pos = np.asarray(y_probs_pos)

    # Candidate thresholds: unique predicted scores (plus 0/1 guards).
    cand = np.unique(np.concatenate([[0.0], y_probs_pos, [1.0]]))
    pos = y_true == 1
    neg = ~pos
    n_pos = max(int(pos.sum()), 1)
    n_neg = max(int(neg.sum()), 1)

    best_t, best_obj, best_info = 0.5, -np.inf, {}
    for t in cand:
        pred = y_probs_pos >= t
        tp = int(np.sum(pred & pos))
        fp = int(np.sum(pred & neg))
        fn = int(np.sum(~pred & pos))
        tn = int(np.sum(~pred & neg))
        sens = tp / n_pos
        spec = tn / n_neg

        if mode == 'cost':
            obj = -(fn_cost * fn + fp_cost * fp)  # maximise negative cost
        elif mode == 'f1':
            denom = (2 * tp + fp + fn)
            obj = (2 * tp) / denom if denom > 0 else 0.0
        else:  # youden
            obj = sens + spec - 1.0

        if obj > best_obj:
            best_obj = obj
            best_t = float(t)
            best_info = {'sensitivity': sens, 'specificity': spec,
                         'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}

    best_info['threshold'] = best_t
    best_info['mode'] = mode
    return best_t, best_info


def evaluate_model(model: torch.nn.Module,
                   dataloader: torch.utils.data.DataLoader,
                   device: torch.device,
                   use_uncertainty: bool = False,
                   val_loader: Optional[torch.utils.data.DataLoader] = None,
                   use_temperature_scaling: bool = False,
                   use_tta: bool = False,
                   threshold: float = 0.5
                   ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:

    model.eval()

    scaler = None
    if use_temperature_scaling and val_loader is not None:
        scaler = TemperatureScaler().to(device)
        scaler.set_temperature(model, val_loader, device)

    all_labels, all_preds, all_probs = [], [], []
    all_epistemic, all_aleatoric = [], []

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 2:
                images, labels = batch
            else:
                images, labels = batch[0], batch[1]

            images, labels = images.to(device), labels.to(device)

            # Forward
            if use_uncertainty and hasattr(model, "forward"):
                try:
                    outputs = model(images, return_uncertainty=True)
                    if isinstance(outputs, tuple):
                        probs, epistemic, aleatoric = outputs
                    else:
                        raise ValueError
                    # NOTE: `probs` here is already a softmax probability
                    # (UncertaintyGPCNViT returns the mean MC-dropout prediction).
                    # FIX: do NOT softmax again — the previous double-softmax
                    # corrupted calibration/Brier metrics.

                    all_epistemic.extend(epistemic.cpu().numpy())
                    if aleatoric is not None:
                        all_aleatoric.extend(aleatoric.cpu().numpy())

                except (TypeError, ValueError):
                    probs = (_tta_probs(model, images, scaler) if use_tta
                             else _logits_to_probs(model, images, scaler))
            else:
                probs = (_tta_probs(model, images, scaler) if use_tta
                         else _logits_to_probs(model, images, scaler))

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_probs = np.array(all_probs)

    # Predictions. For binary problems honour a tunable decision threshold on the
    # positive (malignant) class; otherwise fall back to argmax. threshold=0.5 is
    # exactly equivalent to argmax, so default behaviour is unchanged.
    if y_probs.shape[1] == 2 and threshold != 0.5:
        y_pred = (y_probs[:, 1] >= threshold).astype(int)
    else:
        y_pred = y_probs.argmax(axis=1)

    # Metrics
    calculator = MetricsCalculator(num_classes=y_probs.shape[1])
    metrics = calculator.calculate_all_metrics(y_true, y_pred, y_probs)
    metrics.update(calculator.calculate_calibration_metrics(y_true, y_probs))

    # Uncertainty metrics
    if len(all_epistemic) > 0:
        uncertainty_analyzer = UncertaintyAnalyzer()
        metrics.update(uncertainty_analyzer.analyze_uncertainty(
            y_true, y_pred, np.array(all_epistemic),
            np.array(all_aleatoric) if len(all_aleatoric) > 0 else None
        ))

    return metrics, y_true, y_pred, y_probs