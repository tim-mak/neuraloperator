import numpy as np
from scipy.interpolate import splprep, splev
from scipy.interpolate import CubicSpline, interp1d
from pathlib import Path
import matplotlib.pyplot as plt
import airfrans as af
import pyvista as pv
import torch

class PotentialFlowField:
    def __init__(self, R=1.0, V_inf=1.0, device='cuda'):
        self.R = R
        self.V_inf = V_inf
        self.device = device

    def solve_on_mapped_grid(self, grid_coords, metrics, alpha_deg=5.0):
        """
        Solve potential flow around a cylinder and map it onto the O-grid.
        
        The key insight: the O-grid is a mapping from (xi, eta) to (x, y).
        - xi (axis 0): wraps around the airfoil (circumferential)
        - eta (axis 1): goes from surface (eta=0) to far field (eta=N)
        
        We solve potential flow on a CONCENTRIC cylinder whose radius matches
        the mean airfoil surface radius, centered at the airfoil centroid.
        """
        n_xi, n_eta = grid_coords.shape[0], grid_coords.shape[1]
        x_phys, y_phys = grid_coords[:,:,0], grid_coords[:,:,1]
        
        # ---- Step 1: Find airfoil centroid and effective radius ----
        # Surface points are at eta=0 (innermost layer)
        x_surf = x_phys[:, 0]
        y_surf = y_phys[:, 0]
        
        # Centroid of the airfoil surface
        x_center = np.mean(x_surf)
        y_center = np.mean(y_surf)
        
        # Shift all grid coordinates so airfoil centroid is at origin
        x = x_phys - x_center
        y = y_phys - y_center
        
        # Effective cylinder radius = mean distance from centroid to surface
        r_surface = np.sqrt((x_surf - x_center)**2 + (y_surf - y_center)**2)
        R_cyl = np.mean(r_surface)
        
        print(f"DEBUG: Airfoil centroid: ({x_center:.4f}, {y_center:.4f})")
        print(f"DEBUG: Effective cylinder R: {R_cyl:.4f}")
        print(f"DEBUG: Surface r range: {r_surface.min():.4f} to {r_surface.max():.4f}")
        
        # ---- Step 2: Polar coordinates centered on airfoil ----
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        
        # Clamp r so no point is inside the cylinder
        r = np.maximum(r, R_cyl)
        
        print(f"DEBUG: r range (clamped): {r.min():.4f} to {r.max():.4f}")
        
        # ---- Step 3: Potential flow around cylinder of radius R_cyl ----
        alpha = np.radians(alpha_deg)
        
        # Kutta condition circulation for lift
        gamma = 4 * np.pi * self.V_inf * R_cyl * np.sin(alpha)
        
        # Complex coordinates (centered on airfoil)
        z = r * np.exp(1j * theta)
        
        # Complex velocity: dw/dz for cylinder with circulation
        dw_dz = self.V_inf * (np.exp(-1j * alpha) - 
                              (R_cyl**2 * np.exp(1j * alpha)) / z**2) + \
                (1j * gamma) / (2 * np.pi * z)
        
        # Cartesian velocity components (in shifted frame, but velocity is frame-invariant)
        U_cart = np.real(dw_dz)
        V_cart = -np.imag(dw_dz)
        
        # Pressure coefficient
        v_mag_sq = U_cart**2 + V_cart**2
        Cp = 1.0 - v_mag_sq / self.V_inf**2
        
        # Polar velocity components (for reference)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        U_polar = U_cart * cos_theta + V_cart * sin_theta    # radial
        V_polar = -U_cart * sin_theta + V_cart * cos_theta   # tangential
        
        print(f"DEBUG: U_cart range: {U_cart.min():.4f} to {U_cart.max():.4f}")
        print(f"DEBUG: V_cart range: {V_cart.min():.4f} to {V_cart.max():.4f}")
        print(f"DEBUG: Cp range: {Cp.min():.4f} to {Cp.max():.4f}")
        
        # Return original (un-shifted) physical coordinates with computed fields
        return x_phys, y_phys, U_polar, V_polar, Cp, U_cart, V_cart, Cp

def get_te_angle(x, y):
    # Assuming x, y are (512, 40)
    # The surface is at eta=0 (index 0)
    surface_x = x[:, 0]
    surface_y = y[:, 0]
    
    # 1. Identify points near the TE (Start and End of the array)
    # Lower surface tangent (start of array)
    dx_lower = surface_x[0] - surface_x[1]
    dy_lower = surface_y[1] - surface_y[0]

    print(f"DEBUG: TE pos (approx): ({surface_x[0]:.6f}, {surface_y[0]:.6f})")
    print(f"DEBUG: TE pos (approx): ({surface_x[1]:.6f}, {surface_y[1]:.6f})")
    print(f"DEBUG: Lower surface tangent vector: ({dx_lower:.6f}, {dy_lower:.6f})")
    
    # Upper surface tangent (end of array)
    dx_upper = surface_x[-2] - surface_x[-1]
    dy_upper = surface_y[-1] - surface_y[-2]

    print(f"DEBUG: TE pos (approx): ({surface_x[-1]:.6f}, {surface_y[-1]:.6f})")
    print(f"DEBUG: TE pos (approx): ({surface_x[-2]:.6f}, {surface_y[-2]:.6f})")
    print(f"DEBUG: Upper surface tangent vector: ({dx_upper:.6f}, {dy_upper:.6f})")
    
    # 2. Calculate the average direction vector
    # This represents the bisector of the TE angle
    dx_avg = (np.abs(dx_lower) + np.abs(dx_upper)) / 2.0
    dy_avg = (np.abs(dy_lower) - np.abs(dy_upper)) / 2.0
    
    # 3. Compute the angle in radians
    te_angle = np.arctan2(dy_avg, dx_avg)
    
    return te_angle

def sort_airfoil_to_sellig(coords):
    """
    Sort airfoil coordinates counter-clockwise starting from trailing edge.
    Preserves sharp trailing edge while using spline smoothing elsewhere.
    """
    if hasattr(coords, 'cpu'):  # Handle PyTorch tensors
        coords = coords.cpu().numpy()
    
    coords = np.array(coords)
    
    # Find trailing edge (point with maximum x) and leading edge (minimum x)
    te_idx = np.argmax(coords[:, 0])
    le_idx = np.argmin(coords[:, 0])
    te_point = coords[te_idx]
    le_point = coords[le_idx]
    
    # Separate into upper and lower surfaces based on y-coordinate relative to TE
    upper_surface = []
    lower_surface = []
    
    for i, point in enumerate(coords):
        if i == te_idx:
            continue  # Skip trailing edge for now
        
        if point[1] > te_point[1]:  # Above trailing edge
            upper_surface.append(i)
        else:  # Below or at trailing edge level
            lower_surface.append(i)
    
    # Identify very small LE region for special angular sorting
    chord_length = coords[:, 0].max() - coords[:, 0].min()
    le_region_size = 0.02 * chord_length  # Only 2% of chord
    le_threshold = le_point[0] + le_region_size
    
    # Separate points into LE region and far-field
    upper_le = [i for i in upper_surface if coords[i, 0] < le_threshold]
    upper_far = [i for i in upper_surface if coords[i, 0] >= le_threshold]
    lower_le = [i for i in lower_surface if coords[i, 0] < le_threshold]
    lower_far = [i for i in lower_surface if coords[i, 0] >= le_threshold]
    
    # Sort far-field regions by x-coordinate (proven method)
    upper_far = sorted(upper_far, key=lambda i: -coords[i, 0])  # TE to LE
    lower_far = sorted(lower_far, key=lambda i: coords[i, 0])   # LE to TE
    
    # For LE region, use arc length progression for smooth ordering
    def sort_by_arc_length(indices, start_point):
        """Sort points by cumulative arc length from start_point"""
        if not indices:
            return []
        
        points = coords[indices]
        # Calculate distances from start point
        distances = [np.linalg.norm(p - start_point) for p in points]
        
        # Sort by distance (closest first)
        sorted_pairs = sorted(zip(indices, distances), key=lambda x: x[1])
        
        # Build path by always choosing closest unvisited point
        path = [sorted_pairs[0][0]]  # Start with closest point
        remaining = [idx for idx, _ in sorted_pairs[1:]]
        
        while remaining:
            current_point = coords[path[-1]]
            # Find closest remaining point
            distances_to_current = [(idx, np.linalg.norm(coords[idx] - current_point)) 
                                  for idx in remaining]
            next_idx, _ = min(distances_to_current, key=lambda x: x[1])
            path.append(next_idx)
            remaining.remove(next_idx)
            
        return path
    
    # Sort LE regions by arc length progression
    if upper_le:
        # Start from the point closest to where upper far-field ends
        if upper_far:
            start_point = coords[upper_far[-1]]  # Last point of upper far-field
        else:
            start_point = te_point
        upper_le = sort_by_arc_length(upper_le, start_point)
        
    if lower_le:
        # Start from the leading edge point itself
        lower_le = sort_by_arc_length(lower_le, le_point)
    
    # Combine: TE -> upper far -> upper LE -> lower LE -> lower far
    ordered_indices = [te_idx] + upper_far + upper_le + lower_le + lower_far
    rough_coords = coords[ordered_indices]
    
    # Downsample if we have too many points (>500 is excessive for Arbifoil)
    if len(rough_coords) > 500:
        # Find the true leading edge point - handle multiple points at min x-coordinate
        min_x = np.min(rough_coords[:, 0])
        x_tolerance = 1e-6  # Small tolerance for floating point comparison
        le_candidates = np.where(np.abs(rough_coords[:, 0] - min_x) <= x_tolerance)[0]
        
        # Debug: Print all LE candidate points
        print(f"\nLE Debug: Found {len(le_candidates)} candidate points at min_x = {min_x:.8f}")
        for i, idx in enumerate(le_candidates):
            x, y = rough_coords[idx, 0], rough_coords[idx, 1]
            print(f"  Candidate {i}: idx={idx}, coord=({x:.8f}, {y:.8f})")
        
        if len(le_candidates) > 1:
            # Multiple points at min x - choose the one closest to y=0 (geometric center)
            le_y_distances = np.abs(rough_coords[le_candidates, 1])
            best_candidate_rel_idx = np.argmin(le_y_distances)
            le_rough_idx = le_candidates[best_candidate_rel_idx]
            print(f"  Selected candidate {best_candidate_rel_idx} (idx={le_rough_idx}) as closest to y=0")
        else:
            le_rough_idx = le_candidates[0]
            print(f"  Single candidate selected: idx={le_rough_idx}")
        
        # Show points around the LE region for context
        le_vicinity = 10  # Show ±10 points around LE
        start_idx = max(0, le_rough_idx - le_vicinity)
        end_idx = min(len(rough_coords), le_rough_idx + le_vicinity + 1)
        print(f"\nPoints around LE (idx {start_idx} to {end_idx-1}):")
        for i in range(start_idx, end_idx):
            x, y = rough_coords[i, 0], rough_coords[i, 1]
            marker = " <-- LE" if i == le_rough_idx else ""
            print(f"  [{i:3d}]: ({x:.6f}, {y:.6f}){marker}")
        
        # Keep every other point to reduce density
        step = len(rough_coords) // 400  # Target ~400 points to leave room for critical points
        step = max(2, step)  # At least every other point
        downsampled_indices = list(range(0, len(rough_coords), step))
        
        # Always preserve critical points: TE (first), LE, and last point
        critical_indices = {0, le_rough_idx, len(rough_coords) - 1}
        
        # Add critical indices that might be missing
        for idx in critical_indices:
            if idx not in downsampled_indices:
                downsampled_indices.append(idx)
        
        # Sort indices to maintain order
        downsampled_indices = sorted(set(downsampled_indices))
        
        rough_coords = rough_coords[downsampled_indices]
        print(f"Downsampled from {len(ordered_indices)} to {len(rough_coords)} points")
        print(f"Preserved LE at idx {le_rough_idx}, coord: ({rough_coords[downsampled_indices.index(le_rough_idx), 0]:.6f}, {rough_coords[downsampled_indices.index(le_rough_idx), 1]:.6f})")
        if len(le_candidates) > 1:
            print(f"Found {len(le_candidates)} candidate LE points, selected most centered")
    
    # Apply spline smoothing only to the middle sections, preserving TE sharpness
    try:
        # Find points that are NOT near the trailing edge
        x_range = coords[:, 0].max() - coords[:, 0].min()
        te_threshold = coords[:, 0].max() - 0.05 * x_range  # Within 5% of TE
        
        # Identify TE region points to preserve
        te_region_mask = rough_coords[:, 0] >= te_threshold
        
        if np.sum(~te_region_mask) > 10:  # Enough points for spline fitting
            # Split into sections: TE region, middle section, back to TE region
            te_start_indices = np.where(te_region_mask)[0]
            middle_indices = np.where(~te_region_mask)[0]
            
            if len(middle_indices) > 0:
                # Get the middle section for spline fitting
                middle_start = middle_indices[0]
                middle_end = middle_indices[-1]
                
                middle_coords = rough_coords[middle_start:middle_end+1]
                
                # Fit spline only to the middle section (not closed)
                tck, u = splprep([middle_coords[:, 0], middle_coords[:, 1]], 
                                s=0, per=False)  # Not periodic
                
                # Resample middle section
                n_middle = len(middle_coords)
                u_new = np.linspace(0, 1, n_middle)
                x_smooth, y_smooth = splev(u_new, tck)
                smooth_middle = np.column_stack([x_smooth, y_smooth])
                
                # Reconstruct: preserve TE regions, smooth middle
                final_coords = np.copy(rough_coords)
                final_coords[middle_start:middle_end+1] = smooth_middle
                
                return final_coords
        
        # Fallback: just return the rough coordinates with proper ordering
        return rough_coords
        
    except Exception as e:
        print(f"Partial spline fitting failed: {e}, returning rough coordinates")
        return rough_coords


def get_smoothed_airfoil_spline( coords, n_target=512):
        """
        Creates a periodic cubic spline of the airfoil and resamples it.
        z_points: Complex numpy array or tensor of coordinates.
        n_target: Target number of points for the Fornberg iteration.
        """


        # 1. Calculate cumulative arc length for the parameter 's'
        dx = np.diff(coords[:,0])
        dy = np.diff(coords[:,1])
        ds = np.sqrt(dx**2 + dy**2)
        s = np.concatenate(([0], np.cumsum(ds)))
        
        # 2. Fit periodic cubic splines for x and y separately
        # bc_type='periodic' ensures smooth closure at the trailing edge
        cs_x = CubicSpline(s, coords[:,0], bc_type='periodic')
        cs_y = CubicSpline(s, coords[:,1], bc_type='periodic')
        
        # 3. Resample onto a uniform arc-length grid
        s_new = np.linspace(0, s[-1], n_target, endpoint=True)
        x_new = cs_x(s_new)
        y_new = cs_y(s_new)

        return np.column_stack([x_new, y_new])

def debug_mesh_connectivity(grid_coords, surface_points):
    # 1. Check Surface (Airfoil) level
    dist_surf = np.linalg.norm(surface_points[0] - surface_points[-1])
    print(f"--- Surface Connectivity (TE) ---")
    print(f"Point 0: {surface_points[0]}")
    print(f"Point N: {surface_points[-1]}")
    print(f"Distance: {dist_surf:.2e}")
    
    # 2. Check full Grid level (all layers)
    # Compare xi=0 and xi=N for all eta layers
    dist_layers = np.linalg.norm(grid_coords[0, :, :] - grid_coords[-1, :, :], axis=1)
    max_gap = np.max(dist_layers)
    mean_gap = np.mean(dist_layers)
    
    print(f"\n--- Grid Extrusion Connectivity ---")
    print(f"Max gap along wake: {max_gap:.2e}")
    print(f"Mean gap along wake: {mean_gap:.2e}")

    if max_gap > 1e-7:
        print("\n[!] WARNING: Mesh is DISCONNECTED at the Trailing Edge.")
        print("This will cause the Jacobian to explode or vanish at the seam.")
    else:
        print("\n[✓] SUCCESS: Mesh is physically connected.")



def calculate_metrics(grid):
# Calculate gradients of the physical coordinates w.r.t logical indices
    # PyVista allows us to access the underlying points
    pts = grid.points.reshape(grid.dimensions[0], grid.dimensions[1], 3)
    
    # Compute derivatives using numpy gradient
    # dx/d_xi, dx/d_eta, etc.
    grad_x = np.gradient(pts[:,:,0])
    grad_y = np.gradient(pts[:,:,1])
    
    x_xi, x_eta = grad_x[0], grad_x[1]
    y_xi, y_eta = grad_y[0], grad_y[1]
    
    # Jacobian: Area of the transformation
    jacobian = np.abs(x_xi * y_eta - x_eta * y_xi)
    
    # Metric coefficients
    alpha = x_eta**2 + y_eta**2
    gamma = x_xi**2 + y_xi**2
    
    metrics ={'J': jacobian,
              'x_xi': x_xi,
              'x_eta': x_eta,
              'y_xi': y_xi,
              'y_eta': y_eta}
    
    return jacobian.flatten(), alpha.flatten(), gamma.flatten(), metrics

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import pyvista as pv

def visualize_pyvista(grid_coords, metrics_dict, scalar_name="Jacobian", log_scale=False):
    n_xi, n_eta, _ = grid_coords.shape
    
    # 1. Create the Physical Grid (Airfoil)
    physical_grid = pv.StructuredGrid(grid_coords[:,:,0], grid_coords[:,:,1], grid_coords[:,:,2])
    
    # 2. Create the Canonical Grid (Perfect Circle)
    # We map indices to theta [0, 2pi] and r [1, 2]
    theta = np.linspace(0, 2*np.pi, n_xi)
    r = np.linspace(1, 2, n_eta)
    THETA, R = np.meshgrid(theta, r, indexing='ij')
    
    x_can = R * np.cos(THETA)
    y_can = R * np.sin(THETA)
    z_can = np.zeros_like(x_can)
    
    canonical_grid = pv.StructuredGrid(x_can, y_can, z_can)
    
    # 3. Attach Metrics to both grids
    # This proves the data is preserved during the transformation
    for name, data in metrics_dict.items():
        physical_grid.point_data[name] = data.flatten()
        canonical_grid.point_data[name] = data.flatten()

    # 4. Interactive Visualization
    plotter = pv.Plotter(shape=(1, 2))
    
    plotter.subplot(0, 0)
    plotter.add_mesh(physical_grid, scalars=scalar_name, log_scale=log_scale, cmap="viridis", show_edges=True)
    plotter.add_title("Physical Space: Airfoil Geometry")
    plotter.view_xy()
    
    plotter.subplot(0, 1)
    plotter.add_mesh(canonical_grid, scalars=scalar_name, log_scale=log_scale, cmap="viridis", show_edges=True)
    plotter.add_title("Canonical Space: Uniform Circle")
    plotter.view_xy()
    
    plotter.link_views() # Moving one rotates/zooms the other
    plotter.show()


# Example usage with your data
# metrics_dict = {'alpha': alpha.reshape(n_xi, n_eta), ...}
# visualize_mapping(grid_data, metrics_dict)

def generate_aligned_o_grid(surface_points, n_layers=40, expansion_ratio=1.02, R_far=8.0):
    n_pts = len(surface_points)
    # 1. Identify the TE angle to align the far-field circle
    te_angle = np.arctan2(surface_points[0, 1], surface_points[0, 0])
    
    grid_coords = np.zeros((n_pts, n_layers, 3))
    
    # 2. Use a stretch factor for radial layers (better for CFD/Potential Flow)
    # Using a geometric progression ensures more points near the airfoil
    indices = np.arange(n_layers)
    weights = (expansion_ratio**indices - 1) / (expansion_ratio**(n_layers - 1) - 1)
    
    for j in range(n_layers):
        w = weights[j] # Radial weight (0 at surface, 1 at far-field)
        
        # Create circle points starting EXACTLY at the TE angle
        # This prevents the 'twist' seen in your previous debug output
        theta = np.linspace(te_angle, te_angle + 2*np.pi, n_pts)
        circle_points = np.column_stack([np.cos(theta), np.sin(theta)]) * R_far
        
        # Blend the airfoil and the circle
        grid_coords[:, j, :2] = (1 - w) * surface_points + w * circle_points
        
        # 3. CRITICAL: Force the seam to be identical
        # This replaces the 7.81e-01 gap with 0.0
        grid_coords[-1, j, :2] = grid_coords[0, j, :2]
        
    return grid_coords



def plot_airfoil(coords, theta, x_circle, y_circle, metrics, title="Airfoil Shape"):
        # 2. Visualization
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Physical Domain: The Airfoil
    ax[0].plot(coords[:, 0], coords[:, 1], 'b-', label='Airfoil Surface')
    ax[0].scatter(coords[0, 0], coords[0, 1], color='red', zorder=5, label='Start (TE)')
    ax[0].quiver(coords[10, 0], coords[10, 1], 
                coords[11, 0]-coords[10, 0], 
                coords[11, 1]-coords[10, 1], 
                scale=0.5, color='green', label='Direction')
    ax[0].set_title("Physical Domain: Airfoil ($x, y$)\n(AirFrans Sorted Points)")
    ax[0].set_aspect('equal')
    ax[0].legend()
    ax[0].grid(True, linestyle='--', alpha=0.6)

    # Canonical Domain: The Circle
    ax[1].plot(x_circle, y_circle, 'r-', label='Canonical Boundary')
    ax[1].scatter(x_circle[0], y_circle[0], color='red', zorder=5, label='Start ($\theta=0$)')
    # Drawing the "Mapping Lines" to show correspondence
    for i in range(0, len(theta), len(theta)//20):
        ax[1].annotate(f"{i}", (x_circle[i], y_circle[i]), textcoords="offset points", xytext=(5,5), fontsize=8)

    ax[1].set_title("Canonical Domain: Circle ($\cos\\theta, \sin\\theta$)\n(Target for Potential Flow)")
    ax[1].set_aspect('equal')
    ax[1].legend()
    ax[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()


def laplacian_smooth(grid_coords, n_iter=100, relax=0.1):
    # grid_coords: (N_xi, N_eta, 3)
    smoothed = grid_coords.copy()
    for _ in range(n_iter):
        # Average neighbors in the computational domain
        # This reduces the 'waviness' 
        left  = np.roll(smoothed,  1, axis=0)
        right = np.roll(smoothed, -1, axis=0)
        up    = np.roll(smoothed,  1, axis=1)
        down  = np.roll(smoothed, -1, axis=1)
        
        target = (left + right + up + down) / 4.0
        smoothed[1:-1, 1:-1, :] += relax * (target[1:-1, 1:-1, :] - smoothed[1:-1, 1:-1, :])
        
        # Re-enforce periodicity at the TE
        smoothed[-1, :, :] = smoothed[0, :, :]
    return smoothed


def plot_pv(grid,):
            # --- Plotting ---
        plotter = pv.Plotter(shape=(1, 4))

        # Subplot 1: The Mesh
        plotter.subplot(0, 0)
        plotter.add_mesh(grid, show_edges=True, color="white")
        plotter.add_title("O-Grid Topology")
        plotter.view_xy()
        # Subplot 2: Jacobian Heatmap
        plotter.subplot(0, 1)
        plotter.add_mesh(grid, scalars="Jacobian", cmap="viridis", log_scale=True, show_edges=True)
        plotter.add_title("Jacobian (Cell Area Metric)")
        plotter.view_xy()

        # Subplot 3: Alpha Heatmap
        plotter.subplot(0, 2)
        plotter.add_mesh(grid, scalars="Alpha", cmap="viridis", log_scale=True, show_edges=True)
        plotter.add_title("Alpha (Radial Stretch)")
        plotter.view_xy()

        # Subplot 3: Gamma Heatmap
        plotter.subplot(0, 3)
        plotter.add_mesh(grid, scalars="Gamma", cmap="viridis", log_scale=True, show_edges=True)
        plotter.add_title("Gamma (Circumferential Stretch)")
        plotter.view_xy()
        plotter.show()

def convert(dataset_root, output_folder, xlen, ylen, xoffset, grid_size):

    archive_dir = Path(output_folder) / "Archive"
    training_dir = Path(output_folder) / "TrainingX5Y4"
    Path(archive_dir).mkdir(parents=True, exist_ok=True)
    Path(training_dir).mkdir(parents=True, exist_ok=True)


    sim_dirs = [str(d) for d in Path(dataset_root).iterdir() if d.is_dir()]
    sim_dirs[:5]  # Process only the first 5 simulations for testing
    #for sim_dir in tqdm(sim_dirs[:2], desc="Processing Simulations"):
    for sim_dir in sim_dirs[4:5]:
        #try:

        name = Path(sim_dir).name

        simulation = af.Simulation(root=dataset_root, name=name)
            # 1. Extract Metadata from Name/Sim
        parts = name.split('_')
        v_mag_inf = float(parts[2]) 
        aoa_deg = float(parts[3])
        aoa_rad = np.deg2rad(aoa_deg)
        foil = simulation.airfoil
        sorted_foil_points = sort_airfoil_to_sellig(foil.points[:, :2])  # Only take x, y coordinates

        if not np.allclose(sorted_foil_points[-1,:], sorted_foil_points[0,:], atol=1e-6):
            print("Closing airfoil loop by adding trailing edge point at the end.")
            sorted_foil_points = np.vstack([sorted_foil_points, sorted_foil_points[0,:]])  
        
        foil_points_spline = get_smoothed_airfoil_spline(sorted_foil_points, n_target=512)

        theta = np.linspace(0, 2*   np.pi, len(foil_points_spline[:, 0]))

        print(f"Theta range: {theta[0]:.2f} to {theta[-1]:.2f} (should be 0 to 2pi)")
        print(f"First point TE {foil_points_spline[0]} Last point TE {foil_points_spline[-1]}")

        x_circle = np.cos(theta)
        y_circle = np.sin(theta)

        plot_airfoil(foil_points_spline, theta, x_circle, y_circle, metrics=None, title=f"Airfoil Shape - AoA: {aoa_deg}°")

        # --- Execution ---
        # Using your sorted_foil_points
        grid_data = generate_aligned_o_grid(foil_points_spline, n_layers=40, expansion_ratio=1.1, R_far=8)
        # laplacian smoothing to reduce waviness in the grid (optional, can be commented out for testing)
        grid_data = laplacian_smooth(grid_data, n_iter=50, relax=0.1)


        # Run debug
        debug_mesh_connectivity(grid_data, foil_points_spline)

        # Create PyVista Structured Grid
        grid = pv.StructuredGrid(grid_data[:,:,0], grid_data[:,:,1], grid_data[:,:,2])
        
        jac, alpha, gamma, metrics = calculate_metrics(grid)

        potflow = PotentialFlowField(R=1.0, V_inf=1.0, device='cuda')

        x, y, U_canon, V_canon, Cp_canon, U_phys, V_phys, Cp_phys = potflow.solve_on_mapped_grid(grid_data, metrics, alpha_deg=aoa_deg)

        


        grid.point_data["Jacobian"] = jac
        grid.point_data["Alpha"] = alpha  #(Radial Stretch)
        grid.point_data["Gamma"] = gamma  #(Circumferential Stretch)
        grid.point_data["Cp"] = Cp_phys.flatten()
        grid.point_data["U"] = U_phys.flatten()
        grid.point_data["V"] = V_phys.flatten() 

        # Visualize using your calculated Jacobian
        visualize_pyvista(grid_data, {"Jacobian": jac,
                                       "Alpha": alpha, 
                                       "Gamma": gamma, 
                                       "Cp": Cp_phys.flatten(),
                                       "U": U_phys.flatten(),
                                       "V": V_phys.flatten()
                                       },
                                       scalar_name="Cp",
                                       log_scale=False)
        #visualize_bridge_pyvista(grid_data, {"Jacobian": jac, "Alpha": alpha, "Gamma": gamma}, scalar_name="Alpha")



if __name__ == "__main__":

    PATH_TO_DATASET = "/home/timm/Projects/PIML/Dataset"
    PATH_TO_OUTPUT = "/home/timm/Projects/PIML/Dataset_Cyl_Pot_FNO_X5Y4"
    GRID_SIZE = (64, 64)
    REGENERATE = True  # Set to True to regenerate all pt files
    xlen = float(6.0)   # domain length to be sampled  (-xlen/2, xlen/2)
    ylen = float(3.0)   # domain height to be sampled  (-ylen/2, ylen/2)
    xoffset = float(1.0)  # x-offset to be sampled

    convert(PATH_TO_DATASET, PATH_TO_OUTPUT, xlen, ylen, xoffset, GRID_SIZE)