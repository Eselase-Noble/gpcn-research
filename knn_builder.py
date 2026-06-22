"""
Efficient KNN construction for patch graphs
"""

import torch
import torch.nn.functional as F
import math
from typing import Optional


class ImprovedKNNBuilder:
    """
    Improved KNN construction with better spatial hashing and hybrid methods
    """
    
    @staticmethod
    def build_geometry_knn(coords: torch.Tensor, k: int) -> torch.Tensor:
        """
        Build KNN graph from coordinates using spatial hashing for large grids.
        
        Args:
            coords: Normalized coordinates [N, 2]
            k: Number of neighbors
            
        Returns:
            knn_indices: [N, k] indices of k nearest neighbors
        """
        N = coords.shape[0]
        device = coords.device
        
        if N <= 1024:  # Small grid, use exact calculation
            spatial_dist = torch.cdist(coords, coords, p=2)
            knn_indices = torch.topk(spatial_dist, k=k+1, largest=False).indices[:, 1:]
        else:
            # Use spatial hashing for approximate KNN
            num_cells = max(int(math.sqrt(N / k)) + 1, 4)
            cell_size = 1.0 / num_cells
            
            # Assign each point to a grid cell
            cell_indices = torch.clamp(
                (coords / cell_size).long(),
                0, num_cells - 1
            )
            
            # Build cell dictionary
            cell_dict = {}
            for i in range(N):
                cell_key = tuple(cell_indices[i].tolist())
                if cell_key not in cell_dict:
                    cell_dict[cell_key] = []
                cell_dict[cell_key].append(i)
            
            # For each point, search neighboring cells
            knn_list = []
            for i in range(N):
                point_coord = coords[i]
                cell_key = tuple(cell_indices[i].tolist())
                
                # Get points from current and neighboring cells (3x3 neighborhood)
                candidates = []
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        neighbor_key = (
                            max(0, min(num_cells - 1, cell_key[0] + dx)),
                            max(0, min(num_cells - 1, cell_key[1] + dy))
                        )
                        if neighbor_key in cell_dict:
                            candidates.extend(cell_dict[neighbor_key])
                
                candidates = list(set(candidates))
                
                # Remove self
                if i in candidates:
                    candidates.remove(i)
                
                if len(candidates) >= k:
                    # Compute distances only to candidates
                    candidates_tensor = torch.tensor(candidates, device=device, dtype=torch.long)
                    cand_coords = coords[candidates_tensor]
                    dists = torch.norm(cand_coords - point_coord, dim=1)
                    
                    # Select k nearest
                    _, idx = torch.topk(dists, k=k, largest=False)
                    knn = candidates_tensor[idx]
                else:
                    # Not enough neighbors in local cells, use global search
                    all_dists = torch.norm(coords - point_coord, dim=1)
                    all_dists[i] = float('inf')  # Exclude self
                    _, idx = torch.topk(all_dists, k=min(k, N-1), largest=False)
                    knn = idx
                    
                    # Pad if needed (use nearest neighbor repetition)
                    if len(knn) < k:
                        padding = knn[-1].repeat(k - len(knn))
                        knn = torch.cat([knn, padding])
                
                knn_list.append(knn)
            
            knn_indices = torch.stack(knn_list)
        
        return knn_indices
    
    @staticmethod
    def build_hybrid_knn(coords: torch.Tensor, 
                        features: torch.Tensor,
                        k: int, 
                        alpha: float = 0.5) -> torch.Tensor:
        """
        Build hybrid KNN using both spatial and feature similarity.
        
        Args:
            coords: Normalized coordinates [N, 2]
            features: Feature vectors [N, D]
            k: Number of neighbors
            alpha: Weight for feature similarity (1-alpha for spatial)
            
        Returns:
            knn_indices: [N, k] indices of k nearest neighbors
        """
        N = coords.shape[0]
        device = coords.device
        
        if N <= 512:  # Small enough for exact calculation
            # Spatial distance (normalized)
            spatial_dist = torch.cdist(coords, coords, p=2)
            spatial_dist = spatial_dist / (spatial_dist.max() + 1e-8)
            
            # Feature distance (cosine similarity)
            features_norm = F.normalize(features, dim=-1)
            feat_sim = torch.mm(features_norm, features_norm.t())
            feat_dist = 1.0 - feat_sim
            
            # Combined distance
            combined_dist = (1 - alpha) * spatial_dist + alpha * feat_dist
            
            # Get k nearest neighbors (excluding self)
            knn_indices = torch.topk(combined_dist, k=k+1, largest=False).indices[:, 1:]
        else:
            # Use approximate method for large N
            # First get spatial KNN with larger k
            spatial_knn = ImprovedKNNBuilder.build_geometry_knn(coords, k * 2)
            
            # Then refine within spatial neighbors using features
            knn_list = []
            for i in range(N):
                spatial_neighbors = spatial_knn[i]
                
                # Compute feature similarity with spatial neighbors
                feat_i = features[i:i+1]  # [1, D]
                feat_neighbors = features[spatial_neighbors]  # [M, D]
                
                # Normalize
                feat_i_norm = F.normalize(feat_i, dim=-1)
                feat_neighbors_norm = F.normalize(feat_neighbors, dim=-1)
                
                # Cosine similarity
                similarities = torch.mm(feat_i_norm, feat_neighbors_norm.t()).squeeze(0)
                
                # Spatial distances to neighbors
                spatial_dist_to_neighbors = torch.norm(
                    coords[spatial_neighbors] - coords[i], dim=1
                )
                spatial_dist_to_neighbors = spatial_dist_to_neighbors / (
                    spatial_dist_to_neighbors.max() + 1e-8
                )
                
                # Feature distances
                feat_dist_to_neighbors = 1.0 - similarities
                
                # Combined distance
                combined = (1 - alpha) * spatial_dist_to_neighbors + alpha * feat_dist_to_neighbors
                
                # Select top k
                num_select = min(k, len(combined))
                _, idx = torch.topk(combined, k=num_select, largest=False)
                knn = spatial_neighbors[idx]
                
                # Pad if needed (use nearest neighbor repetition)
                if len(knn) < k:
                    padding = knn[-1].repeat(k - len(knn))
                    knn = torch.cat([knn, padding])
                
                knn_list.append(knn)
            
            knn_indices = torch.stack(knn_list)
        
        return knn_indices
    
    @staticmethod
    def build_adaptive_knn(coords: torch.Tensor,
                          features: torch.Tensor,
                          k: int,
                          alpha_learnable: torch.nn.Parameter) -> torch.Tensor:
        """
        Build KNN with learnable alpha parameter
        
        Args:
            coords: Normalized coordinates [N, 2]
            features: Feature vectors [N, D]
            k: Number of neighbors
            alpha_learnable: Learnable parameter for spatial vs feature weight
            
        Returns:
            knn_indices: [N, k] indices of k nearest neighbors
        """
        # Clamp alpha to [0, 1]
        alpha = torch.sigmoid(alpha_learnable)
        
        return ImprovedKNNBuilder.build_hybrid_knn(coords, features, k, alpha.item())
