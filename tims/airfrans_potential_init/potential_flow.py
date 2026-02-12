import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import pyvista as pv

class PotentialFlowField:
    def __init__(self, R=1.0, V_inf=1.0, device='cuda'):
        self.R = R
        self.V_inf = V_inf
        self.device = device

    def solve(self, n_r=128, n_theta=128, ratio=1.02, alpha_deg=5.0, lift=0.8):
        # 1. Coordinate Setup (Polar)
        theta = torch.linspace(0, 2 * np.pi, n_theta, device=self.device)
        r = self.R * (ratio ** torch.arange(n_r, device=self.device))
        R_mesh, T_mesh = torch.meshgrid(r, theta, indexing='ij')
        self.R_mesh = R_mesh
        self.T_mesh = T_mesh

        self.z = torch.complex(R_mesh * torch.cos(T_mesh), R_mesh * torch.sin(T_mesh))
        
        # 2. Physics
        alpha = np.radians(alpha_deg)
        gamma = lift / self.V_inf 
        
        # 3. Complex Velocity dw/dz
        dw_dz = self.V_inf * (torch.exp(torch.tensor(-1j * alpha)) - 
                             (self.R**2 * torch.exp(torch.tensor(1j * alpha))) / self.z**2) + \
                (1j * gamma) / (2 * np.pi * self.z)
        
        V_complex = torch.conj(dw_dz)
        U = V_complex.real.cpu().numpy()
        V = V_complex.imag.cpu().numpy()
        Cp = 1.0 - (torch.abs(V_complex) / self.V_inf)**2
        
        return self.z.cpu().numpy().real, self.z.cpu().numpy().imag, U, V, Cp.cpu().numpy()



    def transform_velocity_to_cartestian(self,  v_pol, coeffs):
        """
        z_grid: [R, Theta] complex tensor
        v_circle: [R, Theta] complex velocity (u + iv)
        coeffs: [2, N] or [N, 2] complex Fourier coefficients from Theodorsen's expansion
                where axis 0/1 contains real parts, axis 1/0 contains imaginary parts
        """
        # Ensure coeffs is on the right device
        if not torch.is_tensor(coeffs):
            coeffs = torch.tensor(coeffs, device=self.device)
        coeffs = coeffs.to(self.device)
        
        # Handle different coefficient formats and convert to [N] complex tensor
        if coeffs.dim() == 2:
            if coeffs.shape[0] == 2:  # [2, N] format
                coeffs_complex = torch.complex(coeffs[0], coeffs[1])  # [N] complex
            elif coeffs.shape[1] == 2:  # [N, 2] format
                coeffs_complex = torch.complex(coeffs[:, 0], coeffs[:, 1])  # [N] complex
            else:
                raise ValueError(f"Unsupported coeffs shape: {coeffs.shape}")
        else:
            coeffs_complex = coeffs  # Already complex or 1D
            
        print(f"DEBUG: coeffs shape: {coeffs.shape}, coeffs_complex shape: {coeffs_complex.shape}")
            
        # 1. Calculate the summation terms with safeguards
        n_coeffs = len(coeffs_complex)
        n = torch.arange(1, n_coeffs + 1, device=self.device)
        
        # Prevent division by very small z values and limit high powers
        z_safe = torch.where(torch.abs(self.z) < 1e-6, 
                            torch.sgn(self.z) * 1e-6, 
                            self.z)
        
        # Limit the power to prevent exponential blowup
        max_power = 20  # Limit the highest power
        n_limited = torch.min(n, torch.tensor(max_power, device=self.device))
        
        # Broadcast z and n for the sum: [R, T, N]
        inv_z_pow = 1.0 / (z_safe.unsqueeze(-1) ** n_limited)
        
        # Clamp extreme values to prevent NaN/Inf
        inv_z_pow = torch.clamp(inv_z_pow.real, -1e6, 1e6) + \
                    1j * torch.clamp(inv_z_pow.imag, -1e6, 1e6)
        
        print(f"DEBUG: inv_z_pow shape: {inv_z_pow.shape}")
        
        # DEBUG: Scale down coefficients to prevent blowup
        print("DEBUG: Scaling Theodorsen coefficients to 10% to prevent Jacobian blowup")
        coeffs_complex = 0.1 * coeffs_complex  # Use 10% of original coefficients
        
        # Reshape coeffs to broadcast properly: [N] -> [1, 1, N]
        coeffs_complex = coeffs_complex.unsqueeze(0).unsqueeze(0)
        
        print(f"DEBUG: coeffs_complex final shape: {coeffs_complex.shape}")
        
        # 2. Map Coordinates with safeguards (w = z * exp(sum(Cn/z^n)))
        sum_coords = torch.sum(coeffs_complex * inv_z_pow, dim=-1)
        # Clamp the exponent to prevent overflow
        sum_coords_clamped = torch.clamp(sum_coords.real, -10, 10) + \
                            1j * torch.clamp(sum_coords.imag, -10, 10)
        w_grid = z_safe * torch.exp(sum_coords_clamped)
        
        # 3. Map Velocity with safeguards (V_phys = V_circ / (dw/dz))
        # derivative: dw/dz = (w/z) * (1 - sum(n*Cn/z^n))
        n = n.unsqueeze(0).unsqueeze(0)  # Also broadcast n properly
        n_limited = n_limited.unsqueeze(0).unsqueeze(0)  # Use limited n for derivative too
        sum_deriv = torch.sum(n_limited * coeffs_complex * inv_z_pow, dim=-1)
        # Clamp derivative sum to prevent instability
        sum_deriv_clamped = torch.clamp(sum_deriv.real, -100, 100) + \
                           1j * torch.clamp(sum_deriv.imag, -100, 100)
        dw_dz = (w_grid / z_safe) * (1 - sum_deriv_clamped)
        
        # Ensure v_pol is a tensor on the right device
        if not torch.is_tensor(v_pol):
            v_pol = torch.tensor(v_pol, device=self.device)
        else:
            v_pol = v_pol.to(self.device)
        
        print(f"DEBUG: v_pol shape: {v_pol.shape}, dw_dz shape: {dw_dz.shape}")
        
        # Check for problematic values before division
        print(f"DEBUG: dw_dz - min: {dw_dz.abs().min():.6f}, max: {dw_dz.abs().max():.6f}")
        print(f"DEBUG: dw_dz has NaN: {torch.isnan(dw_dz).any()}, has Inf: {torch.isinf(dw_dz).any()}")
        print(f"DEBUG: v_pol - min: {v_pol.abs().min():.6f}, max: {v_pol.abs().max():.6f}")
        print(f"DEBUG: v_pol has NaN: {torch.isnan(v_pol).any()}, has Inf: {torch.isinf(v_pol).any()}")
        
        # Debug the intermediate calculations that lead to dw_dz
        print(f"DEBUG: w_grid has NaN: {torch.isnan(w_grid).any()}, has Inf: {torch.isinf(w_grid).any()}")
        print(f"DEBUG: sum_deriv has NaN: {torch.isnan(sum_deriv).any()}, has Inf: {torch.isinf(sum_deriv).any()}")
        print(f"DEBUG: self.z has NaN: {torch.isnan(self.z).any()}, has Inf: {torch.isinf(self.z).any()}")
        
        # Handle potential shape mismatches and avoid division by very small numbers
        if v_pol.shape != dw_dz.shape:
            print(f"WARNING: Shape mismatch - v_pol: {v_pol.shape}, dw_dz: {dw_dz.shape}")
        
        # Clean up NaN values in dw_dz first
        dw_dz = torch.where(torch.isnan(dw_dz), torch.tensor(1.0 + 0j, device=self.device), dw_dz)
        
        # Add small epsilon to avoid division by very small numbers, using sgn for complex
        epsilon = 1e-8
        dw_dz_safe = torch.where(torch.abs(dw_dz) < epsilon, 
                                torch.sgn(dw_dz) * epsilon,  # Use sgn instead of sign for complex
                                dw_dz)
        
        v_phys = v_pol / dw_dz_safe
        
        # Check results
        print(f"DEBUG: v_phys - min: {v_phys.abs().min():.6f}, max: {v_phys.abs().max():.6f}")
        print(f"DEBUG: v_phys has NaN: {torch.isnan(v_phys).any()}, has Inf: {torch.isinf(v_phys).any()}")
        
        return w_grid, v_phys

if __name__ == "__main__":
    # 1. Solve on the polar grid
    solver = PotentialFlowField(device='cuda' if torch.cuda.is_available() else 'cpu')
    x_pol, y_pol, u_pol, v_pol, cp_pol = solver.solve(alpha_deg=0.0, lift=2.0)

    # 1. Get your data from your solver (assuming it's on CPU as numpy now)
    # x_pol, y_pol, u_pol, v_pol, cp_pol from your previous class

    # 2. PyVista needs a 3D coordinate array (x, y, z)
    # Even for 2D, we just set Z=0
    points = np.zeros((x_pol.size, 3))
    points[:, 0] = x_pol.flatten()
    points[:, 1] = y_pol.flatten()
    points[:, 2] = 0.0

    # 3. Create the grid object
    # dims are (n_theta, n_r, 1)
    grid = pv.StructuredGrid()
    grid.points = points
    grid.dimensions = [x_pol.shape[1], x_pol.shape[0], 1]

    # 4. Add your physics data to the grid
    grid.point_data["Velocity"] = np.vstack((u_pol.flatten(), v_pol.flatten(), np.zeros_like(u_pol.flatten()))).T
    
    grid.point_data['u_pol'] = u_pol.flatten()
    grid.point_data['v_pol'] = v_pol.flatten()
    grid.point_data["Cp"] = cp_pol.flatten()

    # 5. Plotting
    plotter = pv.Plotter()



    # Alternative approach: Create separate meshes for different data
    # Pressure field mesh
    grid_cp = grid.copy()
    grid_cp.set_active_scalars("u_pol")
    
    # Velocity field mesh for streamlines  
    grid_vel = grid.copy()
    grid_vel.set_active_vectors("Velocity")
    
    # Add the Pressure Map (Contour/Surface) first
    plotter.add_mesh(grid_cp, scalars="v_pol",
                      cmap="RdBu_r", 
                      clim = [-3, 1.0],
                      lighting=False, show_edges=True)
    
    # Add Streamlines using velocity mesh
    #streamline_seeds = pv.Line(pointa=(-3.5, -2.0, 0), pointb=(-3.5, 2.0, 0), resolution=30)
    #streamlines = grid_vel.streamlines_from_source(streamline_seeds, vectors="Velocity", 
    #                                               integration_direction='forward', max_time=10)
    #plotter.add_mesh(streamlines, color="black", line_width=2)
    # Add the Cylinder (for visual reference)
    #cylinder = pv.Cylinder(center=(0, 0, 0), direction=(0, 0, 1), radius=1.0, height=0.1)
    #plotter.add_mesh(cylinder, color="white")

    plotter.view_xy()
    plotter.show()

