
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer

class SelectiveUnitGaussianNormalizer(UnitGaussianNormalizer):
    def __init__(self, dim, num_channels, eps=1e-5,  mask_channels=[5]):
        # Call the parent constructor
        super().__init__(mean=None, std=None, eps=eps, dim=dim)
        
        
        # Ensure mask_channels is always an iterable (list) for the comprehension
        if isinstance(mask_channels, int):
            self.mask_channels = [mask_channels]
        else:
            self.mask_channels = mask_channels

        # Dynamically build the list of channels to normalize
        self.channels_to_normalize = [
            i for i in range(num_channels) if i not in self.mask_channels
        ]

    def transform(self, x):
        # x is [Batch, 11, H, W]       
        # 11 channel input is 
        # [ X,Y,U_x,U_y,Cp_pot, exp_sdf,x_xi,x_eta,y_xi,y_eta,det_J]
        # Ensure mean and std are specifically the same shape as the slice
        # we want to normalize all channels except the exp_sdf (channel 5)

        # Create a clone to prevent in-place modification issues in PyTorch autograd
        x_norm = x.clone()
        channels = self.channels_to_normalize # [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
        
        m = self.mean
        s = self.std
        
        # 1. Full Tensor Case (Standard for Outputs)
        # If the model output is 5 channels and the normalizer has 5 channels

        channels = self.channels_to_normalize # [0, 1, 3, 4]

        
        # Slicing the stats to match the selected channels
        m_sub = m[:, channels, ...]
        s_sub = s[:, channels, ...]
        
        x_norm[:, channels, ...] = (x[:, channels, ...] - m_sub) / (s_sub + self.eps)
        return x_norm

    def inverse_transform(self, x):
        m = self.mean
        s = self.std

        channels = self.channels_to_normalize
        x_phys = x.clone()

        m = m[:, channels, ...]
        s = s[:, channels, ...]
        x_phys[:, channels, ...] = (x[:, channels, ...] * (s + self.eps)) + m
        return x_phys