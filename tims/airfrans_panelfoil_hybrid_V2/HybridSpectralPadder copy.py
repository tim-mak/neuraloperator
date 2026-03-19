from neuralop.layers.fourier_continuation import FCLegendre
import torch
import torch.nn as nn
import numpy as np
from numpy.polynomial.legendre import Legendre
import warnings
from pathlib import Path
import tensorly as tl
import torch.nn.functional as F

class HybridChannelWiseSpectralPadder(nn.Module):
    def __init__(self, d=2, n_additional_pts=50, x_axis_strategy=['legendre', 'legendre', 'legendre', 'taper', 'taper']):
        super().__init__()
        self.n_additional_pts = n_additional_pts
        self.x_axis_strategy = x_axis_strategy
        # Initialize FCLegendre for the 'legendre' strategy
        self.x_continuation = FCLegendre(d=d, n_additional_pts=n_additional_pts)

    def _taper_zero_pad(self, x):
        # Taper length: usually 10-15 pixels is enough to kill the derivative
        taper_len = 15 
        taper = torch.linspace(0, 1, taper_len, device=x.device)
        
        x_tapered = x.clone()
        # We only really need to taper the 'right' edge because that's where 
        # the continuation/padding happens in FCLegendre's logic
        x_tapered[:, :, -taper_len:, :] *= taper.flip(0).view(1, 1, -1, 1)
        
        # Pad at the END (dim -2) to match FCLegendre behavior
        # F.pad format is (W_left, W_right, H_top, H_bottom) -> for 4D (Y_left, Y_right, X_left, X_right)
        return F.pad(x_tapered, (0, 0, 0, self.n_additional_pts), mode='constant', value=0)

    def pad(self, x):
        # 1. Mirror Y (Dim -1)
        x = torch.cat([x, x.flip(-1)], dim=-1)

        # 2. Channel-wise X-Padding
        padded_channels = []
        for i, strategy in enumerate(self.x_axis_strategy):
            channel_data = x[:, i:i+1, :, :]
            if strategy == 'legendre':
                # FCLegendre.extend defaults to padding at the end
                padded_channels.append(self.x_continuation.extend(channel_data, dim=(-2,)))
            elif strategy == 'taper':
                padded_channels.append(self._taper_zero_pad(channel_data))
            else:
                # Fallback: simple zero pad at the end
                padded_channels.append(F.pad(channel_data, (0, 0, 0, self.n_additional_pts)))

        return torch.cat(padded_channels, dim=1)

    def unpad(self, x_padded):
        # 1. Unpad X (Dim -2): Since we padded at the end, we slice the beginning
        # This replaces self.x_continuation.restrict for multi-strategy safety
        original_x_len = x_padded.shape[-2] - self.n_additional_pts
        x = x_padded[:, :, :original_x_len, :]
        
        # 2. Unpad Y (Dim -1): Remove the mirrored half
        y_original_size = x.shape[-1] // 2
        return x[:, :, :, :y_original_size]

    def forward(self, x):
        return self.pad(x)