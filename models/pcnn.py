import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class PositionAwareCNN(nn.Module):
    """Position-Aware Convolutional Network 
    
    Args:
        in_channels (int): Input channels (D), matching GAF output channels
        hidden_dims (List[Tuple[int, int]]): Conv layer config [(kernel_size, out_channels), ...]
        position_dim (int): Position encoding dimension (default: 2 for lat/lon)
        dropout (float): Spatial dropout rate (default: 0.1)
        activation (str): Activation type ('gelu'|'relu'|'silu', default: 'gelu')
    
    Input Shapes:
        x: (B, in_channels, H, W) ← GAF output
        positions: (B, H, W, position_dim)
    
    Output Shape:
        (B, final_dim, H, W)
    """
    def __init__(self,
                 in_channels: int = 10,
                 hidden_dims: List[Tuple[int, int]] = [(3, 64), (3, 128)],
                 position_dim: int = 2,
                 dropout: float = 0.1,
                 activation: str = "gelu"):
        super().__init__()
        
        # initialization
        self.layers = nn.ModuleList()
        current_channels = in_channels
        
        # Hierarchical convolution construction
        for idx, (kernel_size, out_channels) in enumerate(hidden_dims):
            self.layers.append(
                PCNNLayer(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    position_dim=position_dim,
                    dropout=dropout,
                    activation=activation,
                    stage=idx+1  # For depth-aware positional scaling
                )
            )
            current_channels = out_channels
        
        # Final projection (engineering enhancement)
        self.final_proj = nn.Conv2d(current_channels, current_channels//2, 1)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Forward pass
        
        Mathematical Form:
        H^{(l+1)} = ReLU(W^{(l)} * H^{(l)} + b^{(l)} + P^{(l)})
        
        Enhancements:
        1. LayerNorm instead of BN (better for small batches)
        2. Residual connections (stabilizes deep training)
        3. Depth-aware positional scaling
        """
        # Input validation
        if x.dim() != 4:
            raise ValueError(f"Input tensor must be 4D (B,C,H,W), got {x.shape}")
        if positions.shape[-1] != self.layers[0].position_dim:
            raise ValueError(f"Position dim should be {self.layers[0].position_dim}, got {positions.shape[-1]}")
        
        for layer in self.layers:
            x = layer(x, positions)
        
        return self.final_proj(x)

class PCNNLayer(nn.Module):
    """Single Position-Aware Conv Layer (Core Building Block)"""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 position_dim: int,
                 dropout: float,
                 activation: str,
                 stage: int = 1):
        super().__init__()
        self.stage = stage
        
        # Core components
        self.conv = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=kernel_size,
            padding=kernel_size//2,  # Maintain spatial dims
            bias=False
        )
        self.norm = nn.LayerNorm(out_channels)  # Channel-wise normalization
        self.pos_encoder = PositionEncoder(
            position_dim=position_dim,
            feat_dim=out_channels,
            scale_factor=1.0/stage  # Depth-dependent scaling
        )
        
        # Activation configuration
        self.act = self._get_activation(activation)
        self.dropout = nn.Dropout2d(dropout)
        
        # Residual pathway
        self.residual = nn.Identity()
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, 1)

    def _get_activation(self, name: str) -> nn.Module:
        """Configurable activation (paper doesn't specify type)"""
        activations = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU
        }
        return activations[name.lower()]()

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:

        residual = self.residual(x)
        
        # Main path
        x = self.conv(x)
        x = self.norm(x.permute(0,2,3,1)).permute(0,3,1,2)  # Channel-wise LayerNorm
        
        # Positional injection (core innovation)
        pos_feat = self.pos_encoder(positions).permute(0,3,1,2)  # (B,C,H,W)
        x = x + pos_feat
        
        # Non-linear transformation
        x = self.act(x)
        x = self.dropout(x)
        
        return x + residual  # Residual connection (engineering enhancement)

class PositionEncoder(nn.Module):
    """Domain-Specific Positional Encoder 
    
    Features:
    - Handles arbitrary position dimensions
    - Learnable scaling with depth awareness
    - High-frequency encoding (NeRF-style)
    """
    def __init__(self,
                 position_dim: int = 2,
                 feat_dim: int = 64,
                 scale_factor: float = 1.0,
                 num_freq_bands: int = 6):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale_factor))
        
        # Multi-frequency positional encoding
        self.freq_bands = 2.0 ** torch.linspace(0, num_freq_bands-1, num_freq_bands)
        self.position_proj = nn.Sequential(
            nn.Linear(position_dim * 2 * num_freq_bands, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, feat_dim)
        )
    
    def _encode(self, pos: torch.Tensor) -> torch.Tensor:
        """Generate high-frequency positional features"""
        pos = pos.unsqueeze(-1) * self.freq_bands.view(1,1,1,-1).to(pos.device)
        pos_sin = torch.sin(pos * self.scale.abs())
        pos_cos = torch.cos(pos * self.scale.abs())
        return torch.cat([pos_sin, pos_cos], dim=-1).flatten(start_dim=-2)
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Input shape: (B, H, W, position_dim)"""
        encoded = self._encode(positions)  # (B, H, W, position_dim*2*num_bands)
        return self.position_proj(encoded)  # (B, H, W, feat_dim)
