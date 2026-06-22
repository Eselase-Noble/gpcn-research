"""
Improved Graph Patch Correlation Network Layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from typing import Optional

from knn_builder import ImprovedKNNBuilder


class ImprovedGPCNLayer(MessagePassing):
    """
    Improved Graph Patch Correlation Network Layer with fixed normalization
    """
    
    def __init__(self, 
                 embed_dim: int,
                 k: int = 8,
                 use_hybrid: bool = False,
                 alpha: float = 0.5,
                 use_checkpoint: bool = False,
                 learnable_alpha: bool = False):
        super().__init__(aggr='add', node_dim=0)
        
        self.embed_dim = embed_dim
        self.k = k
        self.use_hybrid = use_hybrid
        self.use_checkpoint = use_checkpoint
        
        # Learnable alpha for hybrid KNN
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha))
        else:
            self.register_buffer('alpha', torch.tensor(alpha))
        
        # Query, Key, Value transformations (like attention)
        self.lin_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.lin_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.lin_v = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # Positional encoding network
        pos_encoding_dim = embed_dim // 4
        self.pos_encoder = nn.Sequential(
            nn.Linear(2, pos_encoding_dim),
            nn.GELU(),
            nn.Linear(pos_encoding_dim, pos_encoding_dim)
        )
        
        # Edge weight prediction network
        edge_input_dim = embed_dim * 2 + pos_encoding_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Update network
        self.update_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Forward pass with batch processing support.
        
        Args:
            x: Input features [B, N, D]
            H: Height of patch grid
            W: Width of patch grid
            
        Returns:
            Updated features [B, N, D]
        """
        B, N, D = x.shape
        device = x.device
        
        # Generate coordinates
        coords = self._generate_coordinates(H, W, device)
        
        # Pre-normalize
        x = self.norm1(x)
        
        # Process each sample in batch
        outputs = []
        for b in range(B):
            if self.use_checkpoint and self.training:
                out_b = torch.utils.checkpoint.checkpoint(
                    self._process_single_sample,
                    x[b], coords, H, W,
                    use_reentrant=False
                )
            else:
                out_b = self._process_single_sample(x[b], coords, H, W)
            outputs.append(out_b)
        
        return torch.stack(outputs, dim=0)
    
    def _process_single_sample(self, 
                               x_i: torch.Tensor, 
                               coords: torch.Tensor,
                               H: int, 
                               W: int) -> torch.Tensor:
        """Process a single sample in the batch."""
        N, D = x_i.shape
        device = x_i.device
        
        # Build KNN graph
        if self.use_hybrid:
            alpha_val = torch.sigmoid(self.alpha).item() if isinstance(self.alpha, nn.Parameter) else self.alpha.item()
            knn_indices = ImprovedKNNBuilder.build_hybrid_knn(
                coords, x_i, self.k, alpha_val
            )
        else:
            knn_indices = ImprovedKNNBuilder.build_geometry_knn(coords, self.k)
        
        # Build edge index
        edge_index = self._knn_to_edge_index(knn_indices, N)
        
        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=N)
        
        # Compute edge weights
        edge_weights = self._compute_edge_weights(x_i, coords, edge_index)
        
        # Compute CORRECTED symmetric normalization
        edge_norm = self._compute_normalization(edge_index, N, edge_weights)
        
        # Message passing
        aggregated = self.propagate(
            edge_index=edge_index,
            x=x_i,
            edge_norm=edge_norm,
            size=(N, N)
        )
        
        # Update with residual connection
        updated = self.update_mlp(torch.cat([x_i, aggregated], dim=-1))
        updated = self.norm2(x_i + self.dropout(updated))
        
        return updated
    
    def _generate_coordinates(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Generate normalized coordinates for patch grid."""
        coords = torch.stack(torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        ), dim=-1).reshape(-1, 2).float()
        
        # Normalize to [0, 1]
        if H > 1 and W > 1:
            coords = coords / torch.tensor([H-1, W-1], device=device, dtype=torch.float32)
        
        return coords
    
    def _knn_to_edge_index(self, knn_indices: torch.Tensor, N: int) -> torch.Tensor:
        """Convert KNN indices to edge index format."""
        device = knn_indices.device
        src = torch.arange(N, device=device).unsqueeze(1).repeat(1, self.k).reshape(-1)
        dst = knn_indices.reshape(-1)
        return torch.stack([src, dst], dim=0)
    
    def _compute_edge_weights(self, 
                             x: torch.Tensor, 
                             coords: torch.Tensor,
                             edge_index: torch.Tensor) -> torch.Tensor:
        """Compute learnable edge weights based on features and spatial relationship."""
        src, dst = edge_index
        
        # Feature component
        x_src = x[src]
        x_dst = x[dst]
        features = torch.cat([x_src, x_dst], dim=-1)
        
        # Spatial relationship with positional encoding
        coord_src = coords[src]
        coord_dst = coords[dst]
        spatial_rel = coord_dst - coord_src
        
        # Encode spatial relationship
        pos_features = self.pos_encoder(spatial_rel)
        
        # Combine features
        edge_features = torch.cat([features, pos_features], dim=-1)
        
        # Predict edge weights
        weights = self.edge_mlp(edge_features).squeeze(-1)
        
        return weights
    
    def _compute_normalization(self, 
                              edge_index: torch.Tensor, 
                              N: int,
                              edge_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute CORRECTED symmetric normalization with edge weights.
        
        Fixed version that doesn't double-count edge weights.
        """
        src, dst = edge_index
        
        # Degree computation WITHOUT edge weights (just count neighbors)
        deg = torch.zeros(N, device=edge_index.device, dtype=torch.float32)
        deg = deg.scatter_add(0, src, torch.ones_like(edge_weights))
        
        # Inverse square root of degree
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        
        # Symmetric normalization: D^(-1/2) * A * D^(-1/2)
        # Then multiply by edge weights
        src_norm = deg_inv_sqrt[src]
        dst_norm = deg_inv_sqrt[dst]
        edge_norm = src_norm * dst_norm * edge_weights
        
        return edge_norm
    
    def message(self, x_j: torch.Tensor, edge_norm: torch.Tensor) -> torch.Tensor:
        """Message function with normalized aggregation."""
        # Transform value
        v = self.lin_v(x_j)
        
        # Apply normalization (includes edge weights)
        return v * edge_norm.unsqueeze(-1)
    
    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        """Update function - returns aggregated messages."""
        return aggr_out


class GraphPyramid(nn.Module):
    """
    Multi-scale graph pyramid for hierarchical reasoning
    """
    
    def __init__(self, 
                 embed_dim: int,
                 num_levels: int = 3,
                 k: int = 8,
                 use_hybrid: bool = True,
                 alpha: float = 0.5):
        super().__init__()
        
        self.num_levels = num_levels
        self.embed_dim = embed_dim
        
        # GPCN layers at each level
        self.gpcn_layers = nn.ModuleList([
            ImprovedGPCNLayer(
                embed_dim=embed_dim,
                k=k,
                use_hybrid=use_hybrid,
                alpha=alpha
            ) for _ in range(num_levels)
        ])
        
        # Pooling layers (simple average pooling for now)
        # In a more advanced version, could use learnable pooling
        
        # Upsampling layers to bring features back to original resolution
        self.upsample_layers = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(num_levels - 1)
        ])
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * num_levels, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
    
    def _pool_features(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Pool features by 2x2 average pooling"""
        B, N, D = x.shape
        
        # Reshape to spatial grid
        x_spatial = x.reshape(B, H, W, D)
        
        # Perform 2x2 average pooling
        H_new = H // 2
        W_new = W // 2
        
        x_pooled = F.avg_pool2d(
            x_spatial.permute(0, 3, 1, 2),  # [B, D, H, W]
            kernel_size=2,
            stride=2
        ).permute(0, 2, 3, 1)  # [B, H/2, W/2, D]
        
        # Reshape back to sequence
        x_pooled = x_pooled.reshape(B, H_new * W_new, D)
        
        return x_pooled, H_new, W_new
    
    def _upsample_features(self, x: torch.Tensor, H_target: int, W_target: int) -> torch.Tensor:
        """Upsample features to target resolution"""
        B, N, D = x.shape
        H_current = int(N ** 0.5)
        W_current = H_current
        
        # Reshape to spatial grid
        x_spatial = x.reshape(B, H_current, W_current, D).permute(0, 3, 1, 2)
        
        # Bilinear upsampling
        x_upsampled = F.interpolate(
            x_spatial,
            size=(H_target, W_target),
            mode='bilinear',
            align_corners=False
        ).permute(0, 2, 3, 1)  # [B, H_target, W_target, D]
        
        # Reshape back to sequence
        x_upsampled = x_upsampled.reshape(B, H_target * W_target, D)
        
        return x_upsampled
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Forward pass through graph pyramid
        
        Args:
            x: Input features [B, N, D]
            H: Height of patch grid
            W: Width of patch grid
            
        Returns:
            Multi-scale fused features [B, N, D]
        """
        B, N, D = x.shape
        
        features_pyramid = []
        x_current = x
        H_current, W_current = H, W
        
        # Process at each scale
        for level_idx, gpcn in enumerate(self.gpcn_layers):
            # Apply GPCN at current scale
            x_current = gpcn(x_current, H_current, W_current)
            
            # Store features (will upsample later)
            features_pyramid.append((x_current, H_current, W_current))
            
            # Pool for next level (if not last level)
            if level_idx < self.num_levels - 1 and H_current > 1 and W_current > 1:
                x_current, H_current, W_current = self._pool_features(
                    x_current, H_current, W_current
                )
        
        # Upsample all features to original resolution and concatenate
        upsampled_features = []
        upsample_idx = 0

        for level_idx, (feat, H_feat, W_feat) in enumerate(features_pyramid):
            if H_feat != H or W_feat != W:
                if upsample_idx < len(self.upsample_layers):
                    feat_upsampled = self._upsample_features(feat, H, W)
                    feat_upsampled = self.upsample_layers[upsample_idx](feat_upsampled)
                    upsample_idx += 1
                else:
                    # Last level, just upsample without linear layer
                    feat_upsampled = self._upsample_features(feat, H, W)
            else:
                feat_upsampled = feat
            upsampled_features.append(feat_upsampled)

        # Concatenate multi-scale features
        multi_scale_features = torch.cat(upsampled_features, dim=-1)
        
        # Fusion
        fused = self.fusion(multi_scale_features)
        
        # Residual connection with input
        output = fused + x
        
        return output
