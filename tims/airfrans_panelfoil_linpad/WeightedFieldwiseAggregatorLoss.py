import torch
import torch.nn as nn
import logging
from torch import nn
from typing import Dict, List, Optional, Callable


class WeightedFieldwiseAggregatorLoss(nn.Module):
    """
    Weighted aggregator that slices multi-physics output channels (delta Cp,delta_u, delta_v, log_nutratio)
    and applies specific loss functions and weights to each, preserving 
    spatial dimensions for quadrature-based losses like LpLoss.
    """
    def __init__(self, losses: dict, mappings: dict, weights: dict = None, logging_enabled=True):
        super().__init__()
        # Ensure we have a loss and a mapping for every field
        assert mappings.keys() == losses.keys(), "Mappings and losses must share keys."
        
        self.losses = losses  #  nn.ModuleDict(losses) # ModuleDict handles device placement ( did not work here)
        self.mappings = mappings
        self.logging_enabled = logging_enabled
        
        # Default to equal weighting if none provided
        if weights is None:
            self.weights = {k: 1.0 / len(losses) for k in losses.keys()}
        else:
            self.weights = weights
        # Sum of weights for normalization 
        self.weight_sum = sum(self.weights.values())

    def forward(self, pred, y=None, **kwargs):

        # 1. If y is None, try to get it from kwargs (fallback)
        if y is None:
            y = kwargs.get('y')
            #print(f"⚠️ MetaLoss: y was None, got from kwargs: {type(y)}")

        # 2. Robust unwrapping for pred
        if isinstance(pred, dict):
            # Try 'y', then 'out', then the first value in the dict
            pred = pred.get('y', pred.get('out', next(iter(pred.values()))))

        # 3. Robust unwrapping for y (in case it's still a dict)
        if isinstance(y, dict):
            y = y.get('y', next(iter(y.values())))
        
        # 4. COMPREHENSIVE SAFETY CHECK with detailed debugging
        if pred is None or y is None or not torch.is_tensor(pred) or not torch.is_tensor(y):
            logging.warning("⚠️ AggregatorLoss: Invalid tensors received. Returning 0.0 loss.")
            
            device = pred.device if hasattr(pred, 'device') else 'cuda'
            zero_loss = torch.tensor(0.0, requires_grad=True, device=device)
            return (zero_loss, {}) if self.logging_enabled else zero_loss


        total_loss = 0.0
        loss_record = {}

        for field, indices in self.mappings.items():
            # Check if indices actually exist for this resolution

            #print(f"DEBUG: Processing field '{field}' with indices {indices}")
            p_field = pred[:, indices, ...]
            t_field = y[:, indices, ...]
            
            field_loss = self.losses[field](p_field, t_field)
            total_loss += (self.weights[field] / self.weight_sum) * field_loss
            
            if self.logging_enabled:
                # Standardizing key names for the Trainer's CSV/WandB logger
                # Clean up the name for WandB / CSV logging
                clean_name = field.replace('_delta', '')
                loss_record[f'loss_{clean_name}'] = field_loss.detach().item()
        if self.logging_enabled:
            return total_loss, loss_record
        else:
            return total_loss
    
    # 3. Explicitly define __call__ to ensure Python knows this object 
    # can be used as a function in your trainer.
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)