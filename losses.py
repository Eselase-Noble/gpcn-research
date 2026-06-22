"""
Loss functions for GPCN-ViT training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
# Import for type hints
from typing import Dict, Tuple, Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Lin et al. - "Focal Loss for Dense Object Detection" (ICCV 2017)

    FIX: now supports a PER-CLASS alpha (from the dataset's class weights).
    Previously a single scalar alpha was applied to every sample, so the
    computed class_weights were silently ignored whenever focal loss was on
    and the BreakHis benign/malignant imbalance was never corrected.
    """

    def __init__(self,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 reduction: str = 'mean',
                 class_weights: Optional[torch.Tensor] = None):
        """
        Args:
            alpha: Scalar weighting factor (used only if class_weights is None)
            gamma: Focusing parameter (higher = focus more on hard examples)
            reduction: 'mean', 'sum', or 'none'
            class_weights: Optional [num_classes] tensor of per-class alpha.
                           When provided it OVERRIDES the scalar alpha and
                           directly addresses class imbalance.
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if class_weights is not None:
            # Registered as a buffer so .to(device) moves it with the module.
            self.register_buffer('class_alpha', class_weights.float())
            self.use_per_class = True
            self.alpha = 1.0
        else:
            self.class_alpha = None
            self.use_per_class = False
            self.alpha = alpha

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predicted logits [B, num_classes]
            targets: Ground truth labels [B]

        Returns:
            loss: Focal loss value
        """
        # ce_loss = -log(p_t)  =>  p_t = exp(-ce_loss)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)

        # Per-sample alpha: gather the alpha for each sample's true class.
        if self.use_per_class:
            alpha_t = self.class_alpha.gather(0, targets)
        else:
            alpha_t = self.alpha

        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = alpha_t * focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss
    Khosla et al. - "Supervised Contrastive Learning" (NeurIPS 2020)
    """
    
    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07):
        """
        Args:
            temperature: Temperature parameter for softmax
            base_temperature: Base temperature for normalization
        """
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Feature vectors [B, D]
            labels: Ground truth labels [B]
            
        Returns:
            loss: Contrastive loss value
        """
        device = features.device
        batch_size = features.shape[0]
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(features, features.T) / self.temperature  # [B, B]
        
        # Create mask for positive pairs
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)  # [B, B]
        
        # Mask out diagonal (self-similarity)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # Compute log probability
        exp_logits = torch.exp(sim_matrix) * logits_mask
        log_prob = sim_matrix - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Compute mean of log-likelihood over positive pairs
        # Clamp to avoid div-by-zero when a sample has no positive pair in the batch
        pos_count = mask.sum(1).clamp(min=1)
        mean_log_prob_pos = (mask * log_prob).sum(1) / pos_count

        # Exclude samples with no positive pairs from the loss
        valid = mask.sum(1) > 0
        if not valid.any():
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss[valid].mean()

        return loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing
    """
    
    def __init__(self, smoothing: float = 0.1):
        """
        Args:
            smoothing: Label smoothing factor (0.0 = no smoothing)
        """
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predicted logits [B, num_classes]
            targets: Ground truth labels [B]
            
        Returns:
            loss: Smoothed cross-entropy loss
        """
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # Create smoothed targets
        num_classes = inputs.size(-1)
        targets_one_hot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets_smooth = targets_one_hot * self.confidence + (1 - targets_one_hot) * (self.smoothing / (num_classes - 1))
        
        # Compute loss
        loss = (-targets_smooth * log_probs).sum(dim=-1).mean()
        
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss with multiple components
    """
    
    def __init__(self,
                 use_focal: bool = True,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0,
                 use_contrastive: bool = True,
                 contrastive_weight: float = 0.1,
                 contrastive_temperature: float = 0.07,
                 use_label_smoothing: bool = True,
                 label_smoothing: float = 0.1,
                 class_weights: Optional[torch.Tensor] = None):
        """
        Args:
            use_focal: Use focal loss instead of cross-entropy
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
            use_contrastive: Add supervised contrastive loss
            contrastive_weight: Weight for contrastive loss
            contrastive_temperature: Temperature for contrastive loss
            use_label_smoothing: Use label smoothing
            label_smoothing: Label smoothing factor
            class_weights: Class weights for imbalanced data
        """
        super().__init__()
        
        self.use_contrastive = use_contrastive
        self.contrastive_weight = contrastive_weight
        
        # Main classification loss
        if use_focal:
            self.classification_loss = FocalLoss(
                alpha=focal_alpha,
                gamma=focal_gamma,
                class_weights=class_weights  # FIX: actually use the class weights
            )
        elif use_label_smoothing:
            self.classification_loss = LabelSmoothingCrossEntropy(
                smoothing=label_smoothing
            )
        else:
            self.classification_loss = nn.CrossEntropyLoss(weight=class_weights)
        
        # Contrastive loss
        if use_contrastive:
            self.contrastive_loss = SupervisedContrastiveLoss(
                temperature=contrastive_temperature
            )
    
    def forward(self, 
                logits: torch.Tensor, 
                labels: torch.Tensor,
                features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            logits: Predicted logits [B, num_classes]
            labels: Ground truth labels [B]
            features: Feature vectors [B, D] (required if use_contrastive=True)
            
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary with individual loss components
        """
        # Classification loss
        cls_loss = self.classification_loss(logits, labels)
        
        loss_dict = {'classification': cls_loss}
        total_loss = cls_loss
        
        # Contrastive loss
        if self.use_contrastive:
            if features is None:
                raise ValueError("Features required for contrastive loss")
            
            contrast_loss = self.contrastive_loss(features, labels)
            loss_dict['contrastive'] = contrast_loss
            total_loss = total_loss + self.contrastive_weight * contrast_loss
        
        loss_dict['total'] = total_loss
        
        return total_loss, loss_dict


class MixUpCrossEntropy(nn.Module):
    """
    Cross-entropy loss for MixUp/CutMix augmentation
    """
    
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, 
                pred: torch.Tensor,
                target_a: torch.Tensor,
                target_b: torch.Tensor,
                lam: float) -> torch.Tensor:
        """
        Args:
            pred: Predictions [B, num_classes]
            target_a: First set of labels [B]
            target_b: Second set of labels [B]
            lam: Mixing coefficient
            
        Returns:
            loss: Mixed loss value
        """
        return lam * self.criterion(pred, target_a) + (1 - lam) * self.criterion(pred, target_b)


def create_criterion(config, class_weights: Optional[torch.Tensor] = None) -> nn.Module:
    """
    Create loss criterion from configuration
    
    Args:
        config: Experiment configuration
        class_weights: Class weights for imbalanced data
        
    Returns:
        criterion: Loss function
    """
    criterion = CombinedLoss(
        use_focal=config.training.use_focal_loss,
        focal_alpha=config.training.focal_alpha,
        focal_gamma=config.training.focal_gamma,
        use_contrastive=config.training.use_contrastive_loss,
        contrastive_weight=config.training.contrastive_weight,
        contrastive_temperature=config.training.contrastive_temperature,
        use_label_smoothing=config.training.label_smoothing > 0,
        label_smoothing=config.training.label_smoothing,
        class_weights=class_weights
    )
    
    return criterion



