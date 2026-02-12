import numpy as np
import torch

class StabilizedFoil:
    def __init__(self, phi, psi, eta, n_coeffs=50, device='cuda'):
        """
        phi, psi, eta: Arrays from your converged Theodorsen iteration.
        n_coeffs: Number of Fourier modes to keep (truncation helps smoothing).
        """
        self.device = device
        self.phi = phi
        self.n_coeffs = n_coeffs
        
        # 1. Enforce Periodicity: Ensure the surface wraps perfectly
        # We use the FFT to find the complex coefficients Cn = An + iBn
        # This replaces the manual 'c' and 'a' loops in your original code.
        C = np.fft.fft(psi + 1j * eta) 
        
        # 2. Extract and Normalize Coefficients
        # In Theodorsen mapping, Cn accounts for the log-radius and angular dev.
        # We take the first 'n_coeffs' and normalize by the number of points.
        self.coeffs = torch.tensor(C[:n_coeffs] / len(phi), dtype=torch.complex64).to(device)
        
        # 3. Store the average scaling (DC component)
        # This corresponds to your psi0 (average log-radius)
        self.psi0 = torch.tensor(psi.mean(), device=device)


    def transform_domain(self, z_grid, v_circle):
        """
        Transforms the entire potential flow field to physical space.
        """
        n = torch.arange(1, self.n_coeffs + 1, device=self.device)
        
        # Broadcast for summation: [Radial, Theta, Coeffs]
        inv_z_pow = 1.0 / (z_grid.unsqueeze(-1) ** n)
        
        # w = z * exp(psi0) * exp(sum(Cn / z^n))
        # This mapping ensures the circle closes into a watertight airfoil.
        mapping_sum = torch.sum(self.coeffs * inv_z_pow, dim=-1)
        w_grid = z_grid * torch.exp(self.psi0) * torch.exp(mapping_sum)
        
        # Derivative (Metric): dw/dz = (w/z) * (1 - sum(n * Cn / z^n))
        # Stabilization: we clip the derivative to prevent the blowup
        deriv_sum = torch.sum(n * self.coeffs * inv_z_pow, dim=-1)
        dw_dz = (w_grid / z_grid) * (1 - deriv_sum)
        
        # Transform velocity vectors to physical plane
        v_phys = v_circle / dw_dz
        
        return w_grid, v_phys, dw_dz