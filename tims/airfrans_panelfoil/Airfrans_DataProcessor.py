
import math

from neuralop.data.transforms.data_processors import DefaultDataProcessor
from logging import config
import torch
from neuralop.layers.padding import DomainPadding
import torch.nn.functional as F
import math

class AirfransDataProcessor(DefaultDataProcessor):
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
        # Get current dimensions: [Batch, Channel, Xi, Eta]
        n_xi = x.shape[2]
        #min_padded_size = int(math.ceil(n_xi * (1 + self.xi_pad_frac)))
        # Find next power of 2 (e.g., 282 -> 512) needed iby cuFFT for half precision
        #target_xi = 2**math.ceil(math.log2(min_padded_size))
        # 1. Dynamic Tangential Padding (xi axis)
        #pad_total = target_xi - n_xi
        #pad_front = pad_total // 2
        #pad_back = pad_total - pad_front

        # Manually apply zero padding along xi
        #x = F.pad(x, (0, 0, pad_front, pad_back), mode='replicate', value=0)  # Pad xi dimension zeros
        #y = F.pad(y, (0, 0, pad_front, pad_back), mode='replicate', value=0)  # Pad xi dimension zeros not good for log(nutratio)
        # Mirror  about the xi boundaries to create a replicate padding in the radial direction
        #x = torch.cat([x,torch.flip(x, dims=[-1])], dim=-1)
        #y = torch.cat([y,torch.flip(y, dims=[-1])], dim=-1)
        # Store padding for the postprocess step
        #self._current_pad_front = pad_front
        #self._current_n_xi = n_xi
         
        return {'x': x, 'y': y}

    def postprocess(self, output,data_dict):
        """
        By calling super(), we inherit the automatic de-normalization logic:
        - Training: Returns normalized tensor for the loss function.
        - Eval: Returns de-normalized units  for the Evaluator.
        """
        # Use the stored indices to slice back precisely
        #p_f = self._current_pad_front
        #n_xi = self._current_n_xi
        #n_eta = output.shape[-1] // 2 # The original was half of the mirrored total

        # We skip the first 12 (front pad) and take the next 256
        #output = output[..., p_f:p_f+n_xi, :n_eta]
        #output = output[..., :, :n_eta] # Since we mirrored, we can just take the first half of the xi dimension
        # 2. Also unpad the ground truth so the Loss Function is comparing  tensors of truth size
        if 'y' in data_dict:
            data_dict['y'] = data_dict['y']
        return super().postprocess(output,data_dict)