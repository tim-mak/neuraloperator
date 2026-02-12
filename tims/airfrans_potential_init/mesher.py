
import numpy as np

class ConformalGridGenerator:

    def __init__(self, x_coords, y_coords):
        self.x_coords = x_coords
        self.y_coords = y_coords

    def generate_conformal_grid(self, n_radial=100, far_field_radius=20.0):
        x_grid = np.zeros((n_radial, len(self.x_coords)))
        y_grid = np.zeros((n_radial, len(self.x_coords)))
        
        x_grid[0, :] = self.x_coords
        y_grid[0, :] = self.y_coords
        
        dr = 0.01 
        growth_rate = 1.05
        
        # Define a center for the radial expansion (mid-chord)
        center_x = -1.0
        center_y = 0.0
        
        for j in range(1, n_radial):
            # 1. Calculate Cauchy-Riemann Normals (Physics-based)
            dx_dxi = -np.gradient(x_grid[j-1, :])
            dy_dxi = -np.gradient(y_grid[j-1, :])
            nx = -dy_dxi
            ny = dx_dxi
            mag_n = np.sqrt(nx**2 + ny**2) + 1e-12
            
            # 2. Calculate Radial Vectors (Geometry-based for circularity)
            rx = x_grid[j-1, :] - center_x
            ry = y_grid[j-1, :] - center_y
            mag_r = np.sqrt(rx**2 + ry**2) + 1e-12
            
            # 3. Apply the Blender Weight
            # Use a power of 2 or 3 so the near-field remains strictly normal
            w = (j / n_radial)**3
            
            # 4. Blend and Re-normalize
            # Near-field (w=0) follows C-R; Far-field (w=1) follows circle
            ux = (1 - w) * (nx / mag_n) + w * (rx / mag_r)
            uy = (1 - w) * (ny / mag_n) + w * (ry / mag_r)
            
            mag_final = np.sqrt(ux**2 + uy**2)

            # Before Step 5: Extrude Layer
            # Identify the TE point (assuming it's at indices 0 and -1)
            # Force the TE normals to point slightly 'away' from the chord line
            # 1. Identify TE Indices (Assuming index 0 and -1 are the TE)
            # Force a 15-degree 'V' wedge at the start of the march
            te_fan = np.deg2rad(1)

            # Upper TE point: Rotate vector UP (outward)
            ux[0], uy[0] = (ux[0]*np.cos(te_fan) - uy[0]*np.sin(te_fan),
                            ux[0]*np.sin(te_fan) + uy[0]*np.cos(te_fan))

            # Lower TE point: Rotate vector DOWN (outward)
            ux[-1], uy[-1] = (ux[-1]*np.cos(-te_fan) - uy[-1]*np.sin(-te_fan),
                            ux[-1]*np.sin(-te_fan) + uy[-1]*np.cos(-te_fan))

            # 2. Vector Field Smoothing (Critical for d1ea59.jpg stability)
            # This spreads the 'fanning' to neighboring points so the mesh doesn't kink
            from scipy.ndimage import gaussian_filter1d
            ux = gaussian_filter1d(ux, sigma=2.0, mode='wrap')
            uy = gaussian_filter1d(uy, sigma=2.0, mode='wrap')
            
            # 5. Extrude Layer
            x_grid[j, :] = x_grid[j-1, :] + (ux / mag_final) * dr
            y_grid[j, :] = y_grid[j-1, :] + (uy / mag_final) * dr
            
            dr *= growth_rate

        x_grid, y_grid = self.solve_poisson_smooth(x_grid, y_grid, iterations=2000)
            
        return x_grid, y_grid
    

    def solve_poisson_smooth(self,x, y, iterations=2000):
        # Fixed boundaries: Layer 0 (airfoil) and Layer -1 (outer circle)
        for _ in range(iterations):
            # Interior points (all layers except first and last)
            # Standard 5-point Laplacian stencil
            x[1:-1, 1:-1] = 0.25 * (x[2:, 1:-1] + x[:-2, 1:-1] + 
                                    x[1:-1, 2:] + x[1:-1, :-2])
            y[1:-1, 1:-1] = 0.25 * (y[2:, 1:-1] + y[:-2, 1:-1] + 
                                    y[1:-1, 2:] + y[1:-1, :-2])
            
            # Handle the periodic 'wrap-around' at the Trailing Edge
            x[:, 0] = x[:, -1] = 0.5 * (x[:, 0] + x[:, -1])
            y[:, 0] = y[:, -1] = 0.5 * (y[:, 0] + y[:, -1])
        return x, y