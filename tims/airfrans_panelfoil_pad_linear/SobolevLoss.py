import torch
from neuralop import H1Loss,  LpLoss
import torch.nn as nn

class SobolevLoss(nn.Module):
    """Custom wrapper to combine Magnitude (L2) and Smoothness (H1) penalties."""
    def __init__(self, l2_weight=0.5) :
        super().__init__()
        # BOTH must use reduction="mean" to keep the math balanced!
        self.l2_loss = LpLoss(d=2, p=2, reduction="mean")
        self.h1_loss = H1Loss(d=2, reduction="mean") 
        self.l2_weight = l2_weight
        self.h1_weight = 1.0 - l2_weight

    def forward(self, pred, target):
        return (self.l2_weight * self.l2_loss(pred, target)) + \
               (self.h1_weight * self.h1_loss(pred, target))