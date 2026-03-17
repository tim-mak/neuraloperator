import torch
from tims.airfrans_panelfoil.Airfrans_DataProcessor import AirfransDataProcessor

class AirfoilEvaluator:
    def __init__(self, processor, device='cuda'):
        """
        Lighweight evaluator for converting FNO predictions to denormalized units for use during evaluation and diagnostics.
        
        Args:
            processor: Your AirfransDataProcessor containing the trained normalizers.
            device: Device for tensor operations.
        """
        self.processor = processor
        self.device = device

    @torch.no_grad()
    def physicsEvaluate(self, normalized_pred):
        """
        Converts FNO prediction tensor back to physical units (u, v, p, nut).
        Expects out_normalizer to be defined in the processor with correct stats for de-normalization.
        normalized_pred: [Batch, Channels, H, W] tensor from model.forward()
        """

        denorm_pred = self.processor.out_normalizer.inverse_transform(normalized_pred)
        
        return denorm_pred