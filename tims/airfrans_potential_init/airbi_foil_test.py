from ast import Try
import os
import torch
import numpy as np
from scipy.interpolate import griddata
from scipy.integrate import trapezoid
import airfrans as af
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import pyvista as pv
import concurrent.futures
from functools import partial
from airfoil_stabilizer import StabilizedFoil
import pyvista as pv
import numpy as np
import sys

from potential_flow import PotentialFlowField

# Add the arbifoil directory to Python path
sys.path.insert(0, '/home/timm/Projects/PIML/arbifoil')

from scipy.interpolate import splprep, splev
import arbifoil  
import zencfg

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



def process_airfrans_to_pt_archive(dataset_root, input_folder,archive_dir, training_dir ,xlen, ylen, xoffset, grid_size=(128, 128)):
    name = Path(input_folder).name

    simulation = af.Simulation(root=dataset_root, name=name)

    # 1. Extract Metadata from Name/Sim
    parts = name.split('_')
    v_mag_inf = float(parts[2]) 
    aoa_deg = float(parts[3])
    aoa_rad = np.deg2rad(aoa_deg)
    
    # Inlet Components
    u_inf = v_mag_inf * np.cos(aoa_rad)
    v_inf = v_mag_inf * np.sin(aoa_rad)
    nu_mol = simulation.NU
    rho = simulation.RHO
    reynolds = (v_mag_inf * 1.0 / nu_mol)  # assuming chord=1.0
    log_re = np.log10(reynolds)
    #print(f"Processing {name}: v_inf={v_inf}, aoa={aoa_deg}, nu_mol={nu_mol}, rho={rho}, log(Re)={log_re}")

    # 2. Grid Sampling
    mesh = simulation.internal
    xmin, xmax = (-xlen/2 + xoffset, xlen/2 + xoffset)
    ymin, ymax = (-ylen/2, ylen/2)

    x_range = np.linspace(xmin, xmax, grid_size[0])
    y_range = np.linspace(ymin, ymax, grid_size[1])

    grid = pv.RectilinearGrid(x_range, y_range, np.array([mesh.center[2]]))
    sampled = grid.sample(mesh)

    # 3. Raw Field Extraction
    sdf_raw = sampled.point_data['implicit_distance'].reshape(grid_size)
    u_raw = sampled.point_data['U'][:, 0].reshape(grid_size)
    v_raw = sampled.point_data['U'][:, 1].reshape(grid_size)
    p_raw = sampled.point_data['p'].reshape(grid_size)
    nut_raw = sampled.point_data['nut'].reshape(grid_size)

    # Clean NaNs immediately
    u_raw = np.nan_to_num(u_raw, nan=u_inf)
    v_raw = np.nan_to_num(v_raw, nan=v_inf)
    p_raw = np.nan_to_num(p_raw, nan=0.0)
    nut_raw = np.nan_to_num(nut_raw, nan=0.0)

    mask = np.where(sdf_raw < 0, 0.0, 1.0)

    # 4. Non-Dimensionalization (The "Step 3" Logic)
    # Velocity Deficit
    u_ndef = (u_inf - u_raw) / v_mag_inf   
    v_ndef = (v_inf - v_raw) / v_mag_inf   

    # set velocity deficit to 1.0 inside the body (sdf < 0)
    u_ndef[mask == 1.0] = 1.0
    v_ndef[mask == 1.0] = 1.0
    
    # Pressure Coefficient (Cp)
    cp = p_raw / (0.5 * rho * v_mag_inf**2)
    
    # Log Turbulent Viscosity Ratio (log10(nut/nu_mol))
    # We use a floor of 1e-10 to prevent log(0)
    log_nut_ratio = np.log10(np.clip((nut_raw + 1e-10) / nu_mol, a_min=1e-10, a_max=None))

    # check nu?_mol > 0
    if nu_mol <= 0:
        raise ValueError(f"Invalid molecular viscosity (nu_mol): {nu_mol}. Must be positive.")
    
    log_nut_ratio = np.log10(np.clip(nut_raw, 1e-12, None) / nu_mol)
    # 5. Build Inputs (x)

    sdf_fixed = np.nan_to_num(sdf_raw, nan=0.0)
    # x: [u_bc, v_bc, mask, sdf]
    x_data = np.stack([np.full_like(sdf_raw, u_inf), 
                       np.full_like(sdf_raw, v_inf), 
                       mask, sdf_fixed])

    # 6. Build Targets (y)
    # y_raw: Standard SI units
    y_raw = np.stack([u_raw, v_raw, p_raw, nut_raw])
    # y_ndim: Non-Dimensional units
    y_ndim = np.stack([u_ndef, v_ndef, cp, log_nut_ratio])


    airfoil_coords = Path(input_folder) / f"{name}_aerofoil.vtp"

    # For extracting the airfoil coordinates
    #foil = pv.read(airfoil_coords)
    foil = simulation.airfoil
    # Extract point coordinates
    foil_points = foil.points  # (N, 3) array

    #print(f" Airfoil points shape: {foil_points.shape}")

    # Calculate aerodynamic coefficients for verification
    ((cd, cdp, cdv), (cl, clp, clv)) = simulation.force_coefficient(compressible=False,reference=False)

    print(f"  Cd: {cd:.5f}, Cdp: {cdp:.5f}, Cdv: {cdv:.5f}, Cl: {cl:.5f}, Clp: {clp:.5f}, Clv: {clv:.5f}")

    # sort coordinates into sellig order for Arbifoil
    sorted_foil_points = sort_airfoil_to_sellig(foil_points[:, :2])  # Only take x, y coordinates
    print(f" Sorted foil points shape: {sorted_foil_points.shape}")
    debug = True
    if debug:
        # Check ordering - find trailing edge (max x)
        te_idx = np.argmax(sorted_foil_points[:, 0])
        te_point = sorted_foil_points[te_idx]
        
        # Check if we start near trailing edge
        first_point = sorted_foil_points[0]
        print(f"Trailing edge point: ({te_point[0]:.4f}, {te_point[1]:.4f}) at index {te_idx}")
        print(f"First point: ({first_point[0]:.4f}, {first_point[1]:.4f})")
        print(f"Distance from first to TE: {np.linalg.norm(first_point - te_point):.4f}")
        
        # Check anti-clockwise ordering by looking at first few points after TE
        n_check = min(10, len(sorted_foil_points))
        print(f"\nFirst {n_check} points after sorting:")
        for i in range(n_check):
            x, y = sorted_foil_points[i, 0], sorted_foil_points[i, 1]
            print(f"  [{i:2d}]: ({x:.4f}, {y:.4f})")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Full airfoil plot        
        ax1.scatter(sorted_foil_points[:, 0], sorted_foil_points[:, 1], c='b', s=10)
        ax1.plot(sorted_foil_points[:, 0], sorted_foil_points[:, 1], 'b-', alpha=0.3, linewidth=1)
        
        # Annotate every 20th point to show ordering direction
        step = max(1, len(sorted_foil_points) // 20)
        for i in range(0, len(sorted_foil_points), step):
            ax1.annotate(f'{i}', (sorted_foil_points[i, 0], sorted_foil_points[i, 1]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8, 
                        bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.7))
        
        # Highlight start point
        ax1.scatter(sorted_foil_points[0, 0], sorted_foil_points[0, 1], c='red', s=50, marker='s', 
                   label=f'Start (idx=0)')
        # Highlight trailing edge if different
        if te_idx != 0:
            ax1.scatter(te_point[0], te_point[1], c='green', s=50, marker='^', 
                       label=f'TE (idx={te_idx})')
        
        ax1.set_title(f'Full Airfoil - Sorted Points\n(Every {step}th point labeled)')
        ax1.legend()
        ax1.set_aspect('equal', adjustable='box')
        
        # Zoomed leading edge plot
        le_idx_plot = np.argmin(sorted_foil_points[:, 0])  # LE in final sorted points
        le_point_plot = sorted_foil_points[le_idx_plot]
        
        # Define LE zoom region (±10% of chord around LE)
        x_range = sorted_foil_points[:, 0].max() - sorted_foil_points[:, 0].min()
        zoom_x_range = 0.1 * x_range
        zoom_y_range = 0.05 * x_range  # Smaller y range for detail
        
        # Find points in LE region
        le_mask = ((np.abs(sorted_foil_points[:, 0] - le_point_plot[0]) < zoom_x_range) & 
                   (np.abs(sorted_foil_points[:, 1] - le_point_plot[1]) < zoom_y_range))
        le_region_indices = np.where(le_mask)[0]
        
        # Plot LE region with all point indices
        ax2.scatter(sorted_foil_points[:, 0], sorted_foil_points[:, 1], c='lightgray', s=5, alpha=0.3)
        ax2.scatter(sorted_foil_points[le_region_indices, 0], sorted_foil_points[le_region_indices, 1], 
                   c='blue', s=20)
        ax2.plot(sorted_foil_points[:, 0], sorted_foil_points[:, 1], 'b-', alpha=0.2, linewidth=1)
        
        # Annotate ALL points in LE region
        for idx in le_region_indices:
            ax2.annotate(f'{idx}', (sorted_foil_points[idx, 0], sorted_foil_points[idx, 1]), 
                        xytext=(3, 3), textcoords='offset points', fontsize=6, 
                        bbox=dict(boxstyle='round,pad=0.1', fc='cyan', alpha=0.8))
        
        # Highlight the actual LE point
        ax2.scatter(le_point_plot[0], le_point_plot[1], c='red', s=100, marker='*', 
                   label=f'LE (idx={le_idx_plot})')
        
        ax2.set_xlim(le_point_plot[0] - zoom_x_range, le_point_plot[0] + zoom_x_range)
        ax2.set_ylim(le_point_plot[1] - zoom_y_range, le_point_plot[1] + zoom_y_range)
        ax2.set_title(f'Leading Edge Region Detail\n({len(le_region_indices)} points shown)')
        ax2.legend()
        ax2.set_aspect('equal', adjustable='box')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()  
    
    
    # Calculate the pressure coefficient distribution from inviscid flow theory for comparison  
    
    # Ensure airfoil is properly closed (TE point repeated at end)
    if not np.allclose(sorted_foil_points[0], sorted_foil_points[-1], atol=1e-6):
        print("Warning: Airfoil not closed, adding trailing edge point at end")
        sorted_foil_points = np.vstack([sorted_foil_points, sorted_foil_points[0]])
        print(f"Closed airfoil points shape: {sorted_foil_points.shape}")
    
    invi_foil = arbifoil.foil(datFileName=None, coords=sorted_foil_points)

    if debug:
        print("\nTesting inviscid flow calculations with sorted foil coordinates...")
        print("Test Map")
        invi_foil.testMap()
        print( "plot")
        invi_foil.plot()
        print(f"Cp at  AoA {aoa_deg} degrees: ")
        invi_foil.C_p(aoa_deg)

    # Debug the Theodorsen coefficients
    theta, psi, eta = invi_foil.getCoeffs()


    
    print(f"\nTheodorsen coefficient debug:")
    print(f"theta range: {theta.min():.4f} to {theta.max():.4f}")
    print(f"theta[0]: {theta[0]:.4f}, theta[-1]: {theta[-1]:.4f}")
    print(f"psi range: {psi.min():.4f} to {psi.max():.4f}")
    print(f"eta range: {eta.min():.4f} to {eta.max():.4f}")
    
    # Check if theta starts at 0 (trailing edge)
    if abs(theta[0]) > 0.1:
        print(f"WARNING: theta doesn't start at 0! Starting at {theta[0]:.4f}")
    
    # Check for continuity at trailing edge
    te_jump_psi = abs(psi[0] - psi[-1])
    te_jump_eta = abs(eta[0] - eta[-1]) 
    print(f"Trailing edge jumps - psi: {te_jump_psi:.6f}, eta: {te_jump_eta:.6f}")
    
    if te_jump_psi > 0.01 or te_jump_eta > 0.01:
        print("WARNING: Large discontinuity at trailing edge!")
        
    # Show first few and last few points
    print(f"\nFirst 3 theta/psi/eta values:")
    for i in range(min(3, len(theta))):
        print(f"  [{i}]: θ={theta[i]:.4f}, ψ={psi[i]:.4f}, η={eta[i]:.4f}")
    
    print(f"Last 3 theta/psi/eta values:")
    for i in range(max(0, len(theta)-3), len(theta)):
        print(f"  [{i}]: θ={theta[i]:.4f}, ψ={psi[i]:.4f}, η={eta[i]:.4f}")

    # Transform back to Cartesian for comparison
    x_cart = np.cos(theta) * psi
    y_cart = np.sin(theta) * psi

    solver = PotentialFlowField(invi_foil, aoa_deg, v_mag_inf)
    
    Cl = invi_foil.C_L(aoa_deg)
    print(f"Calculated Cl from Theodorsen: {Cl:.5f}")


    stabilizer = StabilizedFoil(theta, psi, eta, n_coeffs=50, device='cuda')   

    w_grid,v_phys,dw_dz = stabilizer.transform_domain(z_grid,v_circle)

    x_phys, y_phys, u_phys, v_phys, Cp = solver.solve(x_range, 128, 128, 1.02, aoa_deg, lift=Cl)
    
    # 1. Extract physical coordinates from the complex w_grid
    # w_grid shape is [n_r, n_theta]
    #x_phys = w_grid.real.cpu().numpy()
    #y_phys = w_grid.imag.cpu().numpy()

    # 2. Create the PyVista point array (N, 3)
    points = np.zeros((x_phys.size, 3))
    points[:, 0] = x_phys.flatten()
    points[:, 1] = y_phys.flatten()
    points[:, 2] = 0.0 # Keep it 2D

    # 3. Format the Velocity and Pressure Data
    # V_phys is complex: u + iv
    u_phys = v_phys.real.cpu().numpy().flatten()
    v_phys_comp = v_phys.imag.cpu().numpy().flatten()
    v_vec = np.stack((u_phys, v_phys_comp, np.zeros_like(u_phys)), axis=1)

    # Calculate Cp based on the physical velocity magnitude
    # props['v_inf'] from your 2026-02-02 archive
    v_inf = v_mag_inf
    v_mag = np.linalg.norm(v_vec, axis=1)
    cp_phys = 1.0 - (v_mag / v_inf)**2

    # plot Cp distribution from inviscid solver using pyvista

    # 3. Create the grid object
    # dims are (n_theta, n_r, 1)
    grid = pv.StructuredGrid()
    grid.points = points
    grid.dimensions = [x_phys.shape[1], x_phys.shape[0], 1]

    # 4. Add your physics data to the grid
    grid.point_data["Velocity"] = np.vstack((u_phys.flatten(), v_phys_comp.flatten(), np.zeros_like(u_phys.flatten()))).T
    grid.point_data["Cp"] = cp_phys.flatten()

    # 5. Plotting
    plotter = pv.Plotter()

    # Placeholder for the PyVista visualization and data packaging
    print("Theodorsen analysis completed successfully!")
    print("The airfoil coordinate sorting and conformal mapping are working.")
    print("Next step: Implement velocity field transformation for full CFD comparison.")
    
    return None  # Skip the rest for now
    
    
    # Alternative approach: Create separate meshes for different data
    # Pressure field mesh
    grid_cp = grid.copy()
    grid_cp.set_active_scalars("Cp")
    
    # Velocity field mesh for streamlines  
    #grid_vel = grid.copy()
    #grid_vel.set_active_vectors("Velocity")
    
    # Add the Pressure Map (Contour/Surface) first
    plotter.add_mesh(grid_cp, scalars="Cp",
                      cmap="RdBu_r", 
                      clim = [-3, 1.0],
                      lighting=False, show_edges=False)
    
    # Add Streamlines using velocity mesh
    streamline_seeds = pv.Line(pointa=(-3.5, -2.0, 0), pointb=(-3.5, 2.0, 0), resolution=30)
    #streamlines = grid_vel.streamlines_from_source(streamline_seeds, vectors="Velocity", 
    #                                               integration_direction='forward', max_time=10)
    #plotter.add_mesh(streamlines, color="black", line_width=2)
    # Add the Cylinder (for visual reference)
    #cylinder = pv.Cylinder(center=(0, 0, 0), direction=(0, 0, 1), radius=1.0, height=0.1)
    #plotter.add_mesh(cylinder, color="white")

    plotter.view_xy()
    plotter.show()
    """

    # TODO: Package everything once variables are properly defined
    """
    # 7. Package everything
    archive_dict = {
        'x': torch.tensor(x_data, dtype=torch.float32),
        'y_raw': torch.tensor(y_raw, dtype=torch.float32),
        'y_ndim': torch.tensor(y_ndim, dtype=torch.float32),
        'props': {
            'v_mag_inf': torch.tensor(v_mag_inf, dtype=torch.float32),
            'aoa_deg':  torch.tensor(aoa_deg, dtype=torch.float32),
            'nu_mol':  torch.tensor(nu_mol, dtype=torch.float32),
            'rho':  torch.tensor(rho, dtype=torch.float32),
            'reynolds': torch.tensor(reynolds, dtype=torch.float32),
            'log_reynolds': torch.tensor(log_re, dtype=torch.float32),
            'grid_bounds': torch.tensor([xmin, xmax, ymin, ymax], dtype=torch.float32),
            'airfoil_points': torch.from_numpy(foil_points).to(torch.float32),
            'cd': torch.tensor(cd, dtype=torch.float32),
            'cl': torch.tensor(cl, dtype=torch.float32),
            'cdp': torch.tensor(cdp, dtype=torch.float32),
            'clp': torch.tensor(clp, dtype=torch.float32),
            'cdv': torch.tensor(cdv, dtype=torch.float32),
            'clv': torch.tensor(clv, dtype=torch.float32)
        }
    
    }

    # Non-dim BCs
    u_bc = np.cos(aoa_rad)
    v_bc = np.sin(aoa_rad)

    # Package into 5-channel input for normalized training
    x_train = np.stack([
        np.full_like(sdf_raw, u_bc), 
        np.full_like(sdf_raw, v_bc),     
        mask, 
        sdf_fixed,
        np.full_like(sdf_raw, log_re), 
    ])



    # 8. Package specifically for the Training Loop (Non-Dimensional + props)
    training_dict = {
        'x': torch.tensor(x_train, dtype=torch.float32),
        'y': torch.tensor(y_ndim, dtype=torch.float32),
        'props': {
            'v_mag_inf': torch.tensor(v_mag_inf, dtype=torch.float32),
            'aoa_deg':  torch.tensor(aoa_deg, dtype=torch.float32),
            'nu_mol':  torch.tensor(nu_mol, dtype=torch.float32),
            'rho':  torch.tensor(rho, dtype=torch.float32),
            'reynolds': torch.tensor(reynolds, dtype=torch.float32),
            'log_reynolds': torch.tensor(log_re, dtype=torch.float32),
            'grid_bounds': torch.tensor([xmin, xmax, ymin, ymax], dtype=torch.float32),
            'airfoil_points': torch.from_numpy(foil_points).to(torch.float32),
            'cd': torch.tensor(cd, dtype=torch.float32),
            'cl': torch.tensor(cl, dtype=torch.float32),
            'cdp': torch.tensor(cdp, dtype=torch.float32),
            'clp': torch.tensor(clp, dtype=torch.float32),
            'cdv': torch.tensor(cdv, dtype=torch.float32),
            'clv': torch.tensor(clv, dtype=torch.float32)
        }
    }
    # 9. Save both to distinct locations
    
    archive_path = Path(archive_dir) / f"{name}_archive_G{grid_size[0]}x{grid_size[1]}.pt"
    train_path = Path(training_dir) / f"{name}_train_G{grid_size[0]}x{grid_size[1]}.pt"

    #torch.save(archive_dict, archive_path)
    #torch.save(training_dict, train_path)

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
            process_airfrans_to_pt_archive(dataset_root, sim_dir, archive_dir, training_dir, xlen, ylen, xoffset, grid_size)
        #except Exception as e:
        #    print(f"Error processing {sim_dir}: {e}")
        #    continue



if __name__ == "__main__":

    PATH_TO_DATASET = "/home/timm/Projects/PIML/Dataset"
    PATH_TO_OUTPUT = "/home/timm/Projects/PIML/Dataset_Cyl_Pot_FNO_X5Y4"
    GRID_SIZE = (64, 64)
    REGENERATE = True  # Set to True to regenerate all pt files
    xlen = float(6.0)   # domain length to be sampled  (-xlen/2, xlen/2)
    ylen = float(3.0)   # domain height to be sampled  (-ylen/2, ylen/2)
    xoffset = float(1.0)  # x-offset to be sampled
    # Rectangular grids
    #convert(PATH_TO_DATASET,PATH_TO_OUTPUT, xlen,ylen,xoffset, GRID_SIZE )
    #GRID_SIZE = (128, 128)
    #convert(PATH_TO_DATASET,PATH_TO_OUTPUT, xlen,ylen,xoffset, GRID_SIZE )
    #GRID_SIZE = (256, 256)
    #convert(PATH_TO_DATASET,PATH_TO_OUTPUT, xlen,ylen,xoffset, GRID_SIZE )
    #GRID_SIZE = (512, 512)
    #convert(PATH_TO_DATASET,PATH_TO_OUTPUT, xlen,ylen,xoffset, GRID_SIZE )
    GRID_SIZE = (1024, 1024)

    convert(PATH_TO_DATASET,PATH_TO_OUTPUT, xlen,ylen,xoffset, GRID_SIZE )   

    #process_airfrans_to_pt_archive(PATH_TO_DATASET,PATH_TO_OUTPUT, xlen,ylen,xoffset, GRID_SIZE )   

    manifest_path = Path(PATH_TO_DATASET) / "manifest.json"
    #copy to output directories
    import shutil
    shutil.copy(manifest_path, Path(PATH_TO_OUTPUT) / "Archive" / "manifest.json")
    shutil.copy(manifest_path, Path(PATH_TO_OUTPUT) / "TrainingX5Y4" / "manifest.json")