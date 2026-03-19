from neuralop.layers.fourier_continuation import FCLegendre
import torch
import torch.nn as nn
import numpy as np
from numpy.polynomial.legendre import Legendre
import warnings
from pathlib import Path
import tensorly as tl

class HybridSpectralPadder(nn.Module):
    def __init__(self, d=5, n_additional_pts=50):
        super().__init__()
        # Initialize the library's Legendre Continuation for the X-axis
        self.x_continuation = FCLegendre(d=d, n_additional_pts=n_additional_pts)

    def pad(self, x):
        # 1. Custom Mirror Y (Perfect for C-Mesh)
        x = torch.cat([x, x.flip(-1)], dim=-1)
        
        # 2. Library Polynomial X (Smooth C1+ continuity)
        # dim=(-2,0) targets the spatial X dimension
        x = self.x_continuation.extend(x, dim=(-2,))
        return x
    
    def forward(self, x):
        return self.pad(x)

    def unpad(self, x_padded):
        # 1. Library unpad X
        x = self.x_continuation.restrict(x_padded, dim=(-2,))
        
        # 2. Custom unpad Y
        y_original_size = x.shape[-1] // 2
        return x[..., :y_original_size]