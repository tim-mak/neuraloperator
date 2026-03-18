from typing import List, Union

import torch
from torch import nn
from torch.nn import functional as F

from neuralop.utils import validate_scaling_factor

class MirrorPaddingY(nn.Module):
    """Extends the domain by reflecting the input about the second spatial dimension.

    Concatenates the input with its mirror along the last (Y) axis, doubling
    that dimension. No zero-fill — the reflected data preserves spectral
    smoothness and avoids the Gibbs ringing of zero-padding.

    Notes
    -----
    Expects inputs of shape (batch, channels, X, Y).
    e.g. (B, C, 256, 32) -> (B, C, 256, 64)

    The n_modes in the FNO should be doubled in the Y direction to compensate
    for the doubled domain size so that the same physical wavenumbers are retained.
    e.g. n_modes=[32, 8] -> n_modes=[32, 16]
    """

    def forward(self, x):
        return self.pad(x)

    def pad(self, x):
        """Reflect the input along the last spatial dimension."""
        return torch.cat([x, x.flip(-1)], dim=-1)

    def unpad(self, x):
        """Return the original Y extent by dropping the reflected half."""
        original_y = x.shape[-1] // 2
        return x[..., :original_y]