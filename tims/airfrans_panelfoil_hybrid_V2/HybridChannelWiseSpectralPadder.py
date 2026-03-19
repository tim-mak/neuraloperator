from neuralop.layers.fourier_continuation import FCLegendre
import torch
import torch.nn as nn
import numpy as np
from numpy.polynomial.legendre import Legendre
import warnings
from pathlib import Path
import tensorly as tl
import torch.nn.functional as F
import math

class HybridChannelWiseSpectralPadder(nn.Module):
    def __init__(self, d=2, n_additional_pts=50, x_axis_strategy=['legendre','legendre','legendre','taper','taper']):
        super().__init__()
        # We split the total padding between front and back
        self.n_additional_pts = n_additional_pts
        self.pad_front = n_additional_pts // 2
        self.pad_back = n_additional_pts - self.pad_front
        self.x_axis_strategy = x_axis_strategy
        
        # FCLegendre usually pads at the end, so we might need two objects 
        # or handle the shift manually. Let's handle the shift manually for clarity.
        self.x_continuation = FCLegendre(d=d, n_additional_pts=n_additional_pts)

    def _taper_zero_pad(self, x):
        # x is untouched: [B, 1, 256, Y_mirrored]
        
        # 1. Extract the exact boundary slices to anchor the taper
        left_boundary = x[:, :, 0:1, :]   # First pixel slice
        right_boundary = x[:, :, -1:, :]  # Last pixel slice
        
        # 2. Create smooth cosine curves
        # Left padding: scales from 0.0 up to 1.0
        t_left = (1 - torch.cos(torch.linspace(0, math.pi, self.pad_front, device=x.device))) / 2
        t_left = t_left.view(1, 1, -1, 1)
        
        # Right padding: scales from 1.0 down to 0.0
        t_right = (1 + torch.cos(torch.linspace(0, math.pi, self.pad_back, device=x.device))) / 2
        t_right = t_right.view(1, 1, -1, 1)
        
        # 3. Create the padding blocks
        left_pad = left_boundary * t_left
        right_pad = right_boundary * t_right
        
        # 4. Concatenate: [Pad_Front (25) + Original (256) + Pad_Back (25)]
        return torch.cat([left_pad, x, right_pad], dim=-2)

    def _legendre_pad(self, x):
        # FCLegendre typically appends to the end. 
        # To get symmetric padding, we extend then roll.
        x_ext = self.x_continuation.extend(x, dim=(-2,0))
        return torch.roll(x_ext, shifts=self.pad_front, dims=-2)

    def pad(self, x):
        # 1. Mirror Y
        x_mirrored = torch.cat([x, x.flip(-1)], dim=-1)
        padded_channels = []
        target_x_size = x.shape[-2] + self.n_additional_pts # e.g., 256 + 50 = 306

        for i, strategy in enumerate(self.x_axis_strategy):
            # Select single channel: [B, 1, 256, 64]
            channel_data = x_mirrored[:, i:i+1, :, :] 
            
            if strategy == 'legendre':
                print(f"Legendre for channel {i}")

                # 1. Generate the raw continuation (Size: 306)
                # FCLegendre natively puts 25 points at the front and 25 at the back
                p_raw = self.x_continuation.extend(channel_data, dim=(-2,))
                
                # 2. Extract the true padding from the absolute edges
                pad_left = p_raw[:, :, :self.pad_front, :] 
                pad_right = p_raw[:, :, -self.pad_back:, :]
                
                # 3. Glue them around the UNTOUCHED channel_data
                p = torch.cat([pad_left, channel_data, pad_right], dim=-2)
                
                padded_channels.append(p)
                
            elif strategy == 'taper':
                print(f"Taper for channel {i}")
                # Manually pad to reach the SAME target_x_size
                # Result: [B, 1, 306, 64]
                p = self._taper_zero_pad(channel_data)
                padded_channels.append(p)
                
            else:
                print("Fallback to zero pad for channel {}: Unknown strategy '{}'".format(i, strategy))
                # Fallback: simple zero pad [B, 1, 306, 64]
                p = F.pad(channel_data, (0, 0, self.pad_front, self.pad_back))
                padded_channels.append(p)

        return torch.cat(padded_channels, dim=1)

    def unpad(self, x_padded):
        # Snip the front and back of X
        # x_padded is [B, C, X_total, Y_total]
        start_x = self.pad_front
        end_x = x_padded.shape[-2] - self.pad_back
        x = x_padded[:, :, start_x:end_x, :]
        
        # Snip the mirrored Y
        y_original_size = x.shape[-1] // 2
        return x[:, :, :, :y_original_size]

    def forward(self, x):
        # required by nn.module
        return self.pad(x)