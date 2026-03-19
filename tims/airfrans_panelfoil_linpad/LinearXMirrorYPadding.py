from typing import List, Union
import torch
from torch import nn

class LinearXMirrorYPadding(nn.Module):
    """
    Extends the domain by mirroring Y and linearly interpolating X for periodicity.

    1. Mirroring (Y): Concatenates x with its flip along dim -1. 
       (B, C, X, Y) -> (B, C, X, 2Y)
    2. Linear Bridge (X): Adds padding to the start and end of X that transitions
       linearly from the last sample back to the first.
       (B, C, X, 2Y) -> (B, C, X + 2*padding_x, 2Y)

    Parameters
    ----------
    padding_x : int
        Number of samples to pad at both the start and end of the X dimension.
    """

    def __init__(self, padding_x: int):
        super().__init__()
        self.padding_x = padding_x

    def forward(self, x):
        return self.pad(x)

    def pad(self, x):
        # --- 1. Mirror Y (Last Dimension) ---
        # Result shape: (B, C, X, 2Y)
        x = torch.cat([x, x.flip(-1)], dim=-1)

        if self.padding_x <= 0:
            return x

        # --- 2. Linear Bridge X (Second to Last Dimension) ---
        # Get boundary values
        first_val = x[:, :, 0:1, :]  # (B, C, 1, 2Y)
        last_val = x[:, :, -1:, :]   # (B, C, 1, 2Y)

        # Create the linear weight vector: goes from 0 to 1
        # We use padding_x + 2 and slice [1:-1] to ensure the padding 
        # doesn't just duplicate the boundary pixels.
        t = torch.linspace(0, 1, self.padding_x + 2, device=x.device, dtype=x.dtype)
        t = t[1:-1].view(1, 1, -1, 1)  # Shape: (1, 1, padding_x, 1)

        # Interpolate: (1-t)*Last + t*First
        # This creates a smooth bridge from the end of the domain back to the start
        bridge = (1 - t) * last_val + t * first_val

        # Concatenate: [Bridge, Original, Bridge]
        # Result shape: (B, C, X + 2*padding_x, 2Y)
        return torch.cat([bridge, x, bridge], dim=-2)

    def unpad(self, x):
        """
        Removes the linear X-padding and the mirrored Y-half.
        """
        # Crop Y
        original_y = x.shape[-1] // 2
        x = x[..., :original_y]

        # Crop X
        if self.padding_x > 0:
            x = x[:, :, self.padding_x : -self.padding_x, :]
            
        return x