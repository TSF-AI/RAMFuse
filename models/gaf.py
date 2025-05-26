import torch
import torch.nn as nn
from typing import Tuple

class GAFConverter(nn.Module):
    """Gramian Angular Field Converter Module 
    
    Args:
        seq_len (int): Time series window length L (default: 128)
        n_channels (int): Variable dimension D (includes target + auxiliary variables)
        target_idx (int): Index of target variable Y in the channel dimension (centered by default)
        norm_epsilon (float): Normalization epsilon to prevent division by zero (default: 1e-7)
    
    Input shape: (batch_size, seq_len, n_channels)
    Output shape: (batch_size, n_channels, height, width)
    """
    def __init__(self, 
                 seq_len: int = 128,
                 n_channels: int = 10,
                 target_idx: int = None,
                 norm_epsilon: float = 1e-7):
        super().__init__()
        self.seq_len = seq_len
        self.n_channels = n_channels
        self.norm_epsilon = norm_epsilon
        
        # Automatically center target variable
        self.target_idx = target_idx if target_idx is not None else n_channels // 2
        
        # Channel reordering indices (ensure target is centered)
        channel_order = list(range(n_channels))
        if channel_order[self.target_idx] != self.target_idx:
            channel_order.remove(self.target_idx)
            channel_order.insert(n_channels//2, self.target_idx)
        self.register_buffer('channel_order', torch.tensor(channel_order))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Min-max normalization to [-1, 1] range"""
        min_vals = x.min(dim=1, keepdim=True)[0]
        max_vals = x.max(dim=1, keepdim=True)[0]
        return 2 * (x - min_vals) / (max_vals - min_vals + self.norm_epsilon) - 1

    def _angular_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Angular coordinate transformation"""
        return torch.arccos(torch.clamp(x, -1.0 + self.norm_epsilon, 1.0 - self.norm_epsilon))

    def _construct_gaf(self, phi: torch.Tensor) -> torch.Tensor:
        """GAF matrix construction
        M_d(i,j) = cos(φ_i + φ_j)
        """
        phi_expand_i = phi.unsqueeze(2)  # (B, L, 1, D)
        phi_expand_j = phi.unsqueeze(1)  # (B, 1, L, D)
        return torch.cos(phi_expand_i + phi_expand_j)  # (B, L, L, D)

    def _reorder_channels(self, x: torch.Tensor) -> torch.Tensor:
        """Reorder channels to ensure target variable is centered
        Input: (..., n_channels)
        Output: (..., n_channels) with predefined order
        """
        return x[..., self.channel_order]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Complete conversion pipeline:
        1. Input validation
        2. Channel reordering 
        3. Normalization
        4. Angular transformation 
        5. GAF matrix construction 
        """
        # Input validation
        if x.dim() != 3:
            raise ValueError(f"Input shape must be (batch, seq_len, channels), got {x.shape}")
        if x.size(1) != self.seq_len:
            raise ValueError(f"Sequence length must be {self.seq_len}, got {x.size(1)}")
            
        # Channel reordering 
        x_reordered = self._reorder_channels(x)  # (B, L, D)
        
        # Per-variable processing 
        normalized = self._normalize(x_reordered)  # (B, L, D)
        phi = self._angular_transform(normalized)  # (B, L, D)
        
        # Construct GAF images
        gaf_images = self._construct_gaf(phi)  # (B, L, L, D)
        
        # Dimension adjustment for CNN compatibility
        return gaf_images.permute(0, 3, 1, 2)  # (B, D, L, L)

    def get_channel_mapping(self) -> dict:
        """Get channel-variable mapping (for interpretability analysis)"""
        return {f"channel_{i}": f"var_{orig_idx}" 
                for i, orig_idx in enumerate(self.channel_order.tolist())}
