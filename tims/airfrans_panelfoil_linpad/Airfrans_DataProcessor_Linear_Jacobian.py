
import math

from neuralop.data.transforms.data_processors import DefaultDataProcessor
from logging import config
import torch
import torch.nn.functional as F
import math

class AirfransDataProcessor_Linear_Jacobian(DefaultDataProcessor):
    def __init__(self, in_normalizer=None, out_normalizer=None, xi_pad_frac=0.1, device='cuda'):
        super().__init__(in_normalizer=in_normalizer, out_normalizer=out_normalizer)
        self.device = device
        self.to(device)
        self.xi_pad_frac = xi_pad_frac # Total padding fraction (e.g., 0.1 for 10%)

    def preprocess(self, data, batched=True):
        """
        Standardizes inputs/outputs while ensuring the mask remains binary.
        """
        # 1. Standard Normalization via Super Class
        # This scales all channels (including the mask) based on your stats.
        processed = super().preprocess(data)
        x, y = processed['x'], processed['y']
         
        return {'x': x, 'y': y}

    def postprocess(self, output,data_dict):
        """
        By calling super(), we inherit the automatic de-normalization logic:
        - Training: Returns normalized tensor for the loss function.
        - Eval: Returns de-normalized units  for the Evaluator.
        """
        if 'y' in data_dict:
            #data_dict['y'] = data_dict['y'][..., :n_eta]
            data_dict['y'] = data_dict['y']
        return super().postprocess(output,data_dict)
    


