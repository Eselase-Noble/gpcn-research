"""
Complete GPCN-ViT Model with Uncertainty Quantification
FIXED VERSION - Drop-in replacement (same class names, same signatures)

CRITICAL FIX: GPCN now integrates WITH attention instead of replacing it
All class names and method signatures remain IDENTICAL for easy integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional, Tuple
import math

from gpcn_layer import ImprovedGPCNLayer, GraphPyramid


class GPCNAdapter(nn.Module):
    """
    FIXED: Adapter that integrates GPCN WITH ViT attention (not replacing)

    CHANGE: Now wraps the original attention block and adds GPCN processing
    OLD BEHAVIOR: Replaced attention (wrong!)
    NEW BEHAVIOR: Keeps attention + adds GPCN (correct!)
    """

    def __init__(self,
                 original_attn: nn.Module,  # NEW: Keep original attention
                 gpcn_layer: nn.Module,
                 embed_dim: int,
                 max_grid_size: int = 64):
        super().__init__()

        # CRITICAL FIX: Store original attention instead of replacing it
        self.original_attn = original_attn
        self.gpcn = gpcn_layer
        self.embed_dim = embed_dim
        self.max_grid_size = max_grid_size

        # Fusion layer to combine attention + graph features
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Learnable weight for fusion
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        Forward pass: Attention + GPCN integration

        Args:
            x: Input features [B, N, C]
            attn_mask: Attention mask (kept for compatibility)
            **kwargs: Absorbs extra kwargs (e.g. is_causal) from newer timm/PyTorch

        Returns:
            Fused features [B, N, C]
        """
        B, N, C = x.shape

        # Infer grid shape
        patch_count = N - 1
        H = W = int(math.sqrt(patch_count))

        if H * W != patch_count:
            H = int(math.sqrt(patch_count))
            W = (patch_count + H - 1) // H

        # ==========================================
        # STEP 1: Original ViT Attention (PRESERVED!)
        # ==========================================
        x_attn = self.original_attn(x, attn_mask=attn_mask)  # Run original attention

        # ==========================================
        # STEP 2: GPCN Processing
        # ==========================================
        cls_token = x_attn[:, :1, :]
        patches = x_attn[:, 1:, :]

        # Handle grid size
        target_patches = H * W
        if patches.shape[1] != target_patches:
            if patches.shape[1] < target_patches:
                pad_size = target_patches - patches.shape[1]
                padding = torch.zeros(B, pad_size, C, device=x.device, dtype=x.dtype)
                patches = torch.cat([patches, padding], dim=1)
            else:
                patches = patches[:, :target_patches, :]

        # Apply GPCN
        patches_gpcn = self.gpcn(patches, H, W)
        x_gpcn = torch.cat([cls_token, patches_gpcn], dim=1)

        # ==========================================
        # STEP 3: Fusion (Attention + Graph)
        # ==========================================
        x_combined = torch.cat([x_attn, x_gpcn], dim=-1)
        x_fused = self.fusion(x_combined)

        # Learnable residual
        alpha = torch.sigmoid(self.alpha)
        output = alpha * x_attn + (1 - alpha) * x_fused

        return output


class GPCNViT(nn.Module):
    """
    Vision Transformer with Graph Patch Correlation Network

    CLASS NAME: UNCHANGED
    SIGNATURE: UNCHANGED
    BEHAVIOR: FIXED (now integrates GPCN with attention)
    """

    def __init__(self,
                 num_classes: int = 2,
                 pretrained: bool = True,
                 num_gpcn_layers: int = 3,
                 k: int = 8,
                 use_hybrid: bool = True,
                 alpha: float = 0.5,
                 use_multi_scale: bool = True,
                 pyramid_levels: int = 3,
                 freeze_backbone: bool = False,
                 hidden_dim: int = 384,
                 dropout: float = 0.3,
                 backbone: str = 'vit_base_patch16_224',
                 backbone_weights_path: str = '',
                 use_gpcn: bool = True):
        super().__init__()

        self.num_classes = num_classes
        self.num_gpcn_layers = num_gpcn_layers
        self.use_multi_scale = use_multi_scale
        self.use_gpcn = use_gpcn

        # Load base ViT (backbone is now configurable — e.g. ImageNet ViT or a
        # histopathology foundation model such as Phikon / Lunit-DINO / UNI).
        self.vit = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0
        )

        # Optionally load custom backbone weights (e.g. a self-supervised / domain
        # pretrained checkpoint downloaded separately).
        if backbone_weights_path:
            self._load_backbone_weights(backbone_weights_path)

        self.embed_dim = self.vit.embed_dim

        # CRITICAL FIX: Now creates adapters that KEEP attention
        layer_indices = self._get_gpcn_layer_indices(num_gpcn_layers) if use_gpcn else []

        print(f"\n{'='*70}")
        print(f"GPCN-ViT ARCHITECTURE (FIXED VERSION)")
        print(f"{'='*70}")
        print(f"Backbone: {backbone}")
        print(f"Total ViT blocks: {len(self.vit.blocks)}")
        if use_gpcn:
            print(f"GPCN integrated at blocks: {layer_indices}")
            print(f"IMPORTANT: ViT attention is PRESERVED (not replaced)!")
        else:
            print(f"GPCN DISABLED (plain ViT baseline / ablation)")
        print(f"{'='*70}\n")

        for layer_idx in layer_indices:
            # CRITICAL FIX: Get and KEEP original attention
            original_block = self.vit.blocks[layer_idx]
            original_attn = original_block.attn  # Save original attention!

            # Create GPCN module
            if use_multi_scale:
                gpcn_module = GraphPyramid(
                    embed_dim=self.embed_dim,
                    num_levels=pyramid_levels,
                    k=k,
                    use_hybrid=use_hybrid,
                    alpha=alpha
                )
            else:
                gpcn_module = ImprovedGPCNLayer(
                    embed_dim=self.embed_dim,
                    k=k,
                    use_hybrid=use_hybrid,
                    alpha=alpha,
                    use_checkpoint=False
                )

            # FIXED: Create adapter that keeps attention + adds GPCN
            adapter = GPCNAdapter(
                original_attn=original_attn,  # Pass original attention
                gpcn_layer=gpcn_module,
                embed_dim=self.embed_dim
            )

            # Replace attention with adapter (adapter contains both!)
            self.vit.blocks[layer_idx].attn = adapter

        # Classification head (unchanged)
        self.head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        if freeze_backbone:
            self._freeze_backbone(layer_indices)

    def _load_backbone_weights(self, path: str):
        """Load custom (e.g. domain-pretrained) weights into the ViT backbone."""
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        state = ckpt.get('model', ckpt.get('state_dict', ckpt))
        # Strip common prefixes so the keys line up with timm's ViT.
        cleaned = {}
        for k, v in state.items():
            nk = k
            for prefix in ('module.', 'backbone.', 'encoder.', 'vit.'):
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]
            cleaned[nk] = v
        missing, unexpected = self.vit.load_state_dict(cleaned, strict=False)
        print(f"[backbone] loaded {path} | missing={len(missing)} unexpected={len(unexpected)}")

    def _get_gpcn_layer_indices(self, num_layers: int) -> list:
        """
        UNCHANGED: Same logic as before
        """
        total_blocks = len(self.vit.blocks)
        num_layers = max(1, min(num_layers, total_blocks))

        if num_layers == 1:
            indices = [0]
        elif num_layers == 2:
            indices = [0, total_blocks // 2]
        elif num_layers == 3:
            indices = [0, total_blocks // 2, total_blocks - 1]
        else:
            indices = torch.linspace(
                0, total_blocks - 1, steps=num_layers
            ).long().tolist()

        indices = sorted(list(set(indices)))
        indices = [idx for idx in indices if 0 <= idx < total_blocks]

        return indices

    def _freeze_backbone(self, gpcn_layer_indices: list):
        """UNCHANGED"""
        for name, param in self.vit.named_parameters():
            is_gpcn = any(f'blocks.{idx}.attn' in name for idx in gpcn_layer_indices)
            if not is_gpcn:
                param.requires_grad = False

    def unfreeze_backbone(self):
        """UNCHANGED"""
        for param in self.vit.parameters():
            param.requires_grad = True

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """UNCHANGED"""
        x = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.vit.pos_drop(x + self.vit.pos_embed)
        x = self.vit.blocks(x)
        x = self.vit.norm(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """UNCHANGED"""
        features = self.forward_features(x)
        cls_token = features[:, 0]
        logits = self.head(cls_token)
        return logits


class UncertaintyGPCNViT(nn.Module):
    """
    GPCN-ViT with Monte Carlo Dropout for uncertainty quantification

    CLASS NAME: UNCHANGED
    SIGNATURE: UNCHANGED
    BEHAVIOR: FIXED (MC Dropout now only affects Dropout, not BatchNorm)
    """

    def __init__(self, base_model: GPCNViT, num_samples: int = 10):
        super().__init__()
        self.base_model = base_model
        self.num_samples = num_samples

    def _enable_dropout(self, model: nn.Module):
        """FIXED: Only enable Dropout, not BatchNorm"""
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def _disable_dropout(self, model: nn.Module):
        """Helper to disable dropout"""
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.eval()

    def forward(self,
                x: torch.Tensor,
                return_uncertainty: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        SIGNATURE: UNCHANGED
        BEHAVIOR: FIXED (proper MC Dropout)
        """
        if not return_uncertainty:
            return self.base_model(x), None, None

        # FIXED: Keep model in eval but enable dropout
        self.base_model.eval()
        self._enable_dropout(self.base_model)

        predictions = []
        with torch.no_grad():
            for _ in range(self.num_samples):
                pred = F.softmax(self.base_model(x), dim=-1)
                predictions.append(pred)

        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        epistemic_uncertainty = predictions.var(dim=0).sum(dim=-1)
        predictive_entropy = -(mean_pred * torch.log(mean_pred + 1e-10)).sum(dim=-1)

        # Restore
        self._disable_dropout(self.base_model)

        return mean_pred, epistemic_uncertainty, predictive_entropy


class MultiMagnificationGPCNViT(nn.Module):
    """
    Multi-magnification GPCN-ViT

    CLASS NAME: UNCHANGED
    SIGNATURE: UNCHANGED
    BEHAVIOR: UNCHANGED (works with fixed base model)
    """

    def __init__(self,
                 base_model: GPCNViT,
                 num_magnifications: int = 4):
        super().__init__()
        self.base_model = base_model
        self.num_magnifications = num_magnifications

        self.mag_heads = nn.ModuleList([
            nn.Linear(base_model.embed_dim, base_model.num_classes)
            for _ in range(num_magnifications)
        ])

        self.mag_weights = nn.Parameter(torch.ones(num_magnifications) / num_magnifications)

        self.fusion = nn.MultiheadAttention(
            base_model.embed_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

    def forward(self,
                images_multi_mag: list,
                magnification_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """UNCHANGED"""
        if not isinstance(images_multi_mag, list):
            return self.base_model(images_multi_mag)

        features_list = []
        for img in images_multi_mag:
            feat = self.base_model.forward_features(img)[:, 0, :]
            features_list.append(feat)

        features_stacked = torch.stack(features_list, dim=1)
        fused_features, _ = self.fusion(
            features_stacked, features_stacked, features_stacked
        )

        predictions = []
        for i, head in enumerate(self.mag_heads):
            pred = head(fused_features[:, i, :])
            predictions.append(pred)

        predictions = torch.stack(predictions, dim=1)
        weights = F.softmax(self.mag_weights, dim=0)
        final_pred = (predictions * weights.view(1, -1, 1)).sum(dim=1)

        return final_pred


def create_model(config) -> nn.Module:
    """
    FUNCTION NAME: UNCHANGED
    SIGNATURE: UNCHANGED
    BEHAVIOR: Uses fixed GPCNViT (with proper integration)
    """
    print("\n" + "="*70)
    print("CREATING GPCN-ViT MODEL (FIXED VERSION)")
    print("="*70)
    print("IMPORTANT: GPCN integrates WITH attention (not replacing)")
    print("="*70 + "\n")

    base_model = GPCNViT(
        num_classes=config.model.num_classes,
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

    if config.model.use_uncertainty:
        model = UncertaintyGPCNViT(base_model, num_samples=config.model.mc_samples)
    else:
        model = base_model

    return model


def test_model():
    """UNCHANGED"""
    from config import get_default_config

    config = get_default_config()

    print("Creating model...")
    model = create_model(config)

    print(f"\nModel created successfully!")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    print("\nTesting forward pass...")
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)

    try:
        if config.model.use_uncertainty:
            logits, _, _ = model(x, return_uncertainty=False)
        else:
            logits = model(x)

        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {logits.shape}")
        print("  ✓ Forward pass successful!")

        if config.model.use_uncertainty:
            print("\nTesting with uncertainty estimation...")
            mean_pred, epistemic, entropy = model(x, return_uncertainty=True)
            print(f"  Mean predictions: {mean_pred.shape}")
            print(f"  Epistemic uncertainty: {epistemic.shape}")
            print(f"  Predictive entropy: {entropy.shape}")
            print("  ✓ Uncertainty estimation successful!")

        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED - Model ready for training!")
        print("="*70)
        return True
    except Exception as e:
        print(f"  ✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    test_model()