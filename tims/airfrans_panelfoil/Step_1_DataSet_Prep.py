
from curses import window
from doctest import master
from pathlib import Path
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import interpolate
from matplotlib import cm
from scipy.interpolate import interp1d
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve
from scipy.spatial import KDTree
from scipy.interpolate import splrep, splev
import sys, os
sys.path.insert(0, '/home/timm/Projects/PIML/subfoil')
from airfoil_utils import BSplineFoil
import vortexSourcePanelfoil_VecAnalytic as vortexSourcePanelfoil
import pyvista as pv
import airfrans as af
import concurrent.futures
from functools import partial
from tqdm import tqdm

import torch
import torch.nn.functional as F

from BlockMeshInterpolator import concatenate_blocks, extract_arrays, resample_block_pchip,  chain_stitch_orientation, compute_jacobian_metrics, concatenate_blocks_ml_tensor, read_openfoam_results, read_structured_grids_from_vtm, write_single_structured_cmesh_vtm, write_stitched_raw_subgrids_vtm, write_sampled_blocks_vtm

from scipy.interpolate import PchipInterpolator


def plot_vtm_block_wiremesh(
    grid,
    block_name="block",
    i_cell_range=None,
    j_cell_range=None,
    pt_cloud=None,
    filled=False,
    channel=None,
    cmap_vmin=None,
    cmap_vmax=None,
    title=None,
    interactive=False,
    line_color="k",
    line_width=0.6,
    figsize=(10, 5),

):
    """
    Plot a StructuredGrid block as a wiremesh using Matplotlib only.

        Ranges are specified in cell indices (inclusive):
            i_cell_range=(i0, i1), j_cell_range=(j0, j1)

        This helper is intentionally 2D-only: it always plots the k=0 plane.
        The function converts cell ranges to node extents and plots all i/j lines
        in XY physical space.

        If filled=True, a filled contour-like mesh is drawn from a point-data
        channel on the selected subset. For vector channels, the magnitude is
        used for coloring. Use cmap_vmin/cmap_vmax to clamp color range.
    """

    def _clamp_cell_range(rng, max_cell, axis_name):
        if max_cell < 0:
            raise ValueError(f"Grid has no cells along {axis_name} axis.")
        if rng is None:
            return (0, max_cell)
        r0, r1 = int(rng[0]), int(rng[1])
        if r0 > r1:
            r0, r1 = r1, r0
        r0 = max(0, min(r0, max_cell))
        r1 = max(0, min(r1, max_cell))
        return (r0, r1)

    ni, nj, nk = [int(v) for v in grid.dimensions]
    nc_i = ni - 1
    nc_j = nj - 1
    if nk < 1:
        raise ValueError("StructuredGrid has invalid k dimension.")

    i0c, i1c = _clamp_cell_range(i_cell_range, nc_i, "i")
    j0c, j1c = _clamp_cell_range(j_cell_range, nc_j, "j")
    k0c = 0
    k1c = 0

    # Convert inclusive cell ranges to inclusive node ranges.
    i0n, i1n = i0c, i1c + 1
    j0n, j1n = j0c, j1c + 1
    k0n, k1n = k0c, k1c + 1

    sub = grid.extract_subset([i0n, i1n, j0n, j1n, k0n, k1n])
    si, sj, sk = [int(v) for v in sub.dimensions]
    pts3d = np.asarray(sub.points).reshape((si, sj, sk, 3), order='F')
    
    cld_pts = []
    if pt_cloud is not None:
        pt_cloud = np.asarray(pt_cloud)
        query_pts = pts3d.reshape((-1, 3), order='F')
        tree = KDTree(pt_cloud)

        for pt in query_pts:
            dist, idx = tree.query(pt)
            cld_pts.append(pt_cloud[idx])
            print(f"Closest point to {pt} is {pt_cloud[idx]} with distance {float(dist):.3e}")

    if interactive and plt.get_backend().lower() == "agg":
        try:
            plt.switch_backend("TkAgg")
        except Exception:
            print("Interactive mode requested but TkAgg backend is unavailable; using current backend.")

    fig, ax = plt.subplots(figsize=figsize)

    if filled:
        if channel is None:
            raise ValueError("filled=True requires a point-data channel name via 'channel'.")
        if channel not in sub.point_data:
            raise KeyError(
                f"Channel '{channel}' not found in point_data. Available: {list(sub.point_data.keys())}"
            )

        raw = np.asarray(sub.point_data[channel])
        if raw.ndim == 1:
            scalars = raw
        elif raw.ndim == 2 and raw.shape[1] in (2, 3):
            scalars = np.linalg.norm(raw, axis=1)
        elif raw.ndim == 2:
            scalars = raw[:, 0]
        else:
            scalars = raw.reshape((raw.shape[0], -1))[:, 0]

        scalar3d = scalars.reshape((si, sj, sk), order='F')
        x2d = pts3d[:, :, 0, 0]
        y2d = pts3d[:, :, 0, 1]
        s2d = scalar3d[:, :, 0]
        mesh = ax.pcolormesh(
            x2d,
            y2d,
            s2d,
            shading='auto',
            cmap='viridis',
            vmin=cmap_vmin,
            vmax=cmap_vmax,
        )
        plt.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04, label=channel)

    # Draw mesh lines in i- and j-directions on the k=0 plane.
    kk = 0
    for jj in range(sj):
        ax.plot(
            pts3d[:, jj, kk, 0],
            pts3d[:, jj, kk, 1],
            color=line_color,
            linewidth=line_width,
        )
    for ii in range(si):
        ax.plot(
            pts3d[ii, :, kk, 0],
            pts3d[ii, :, kk, 1],
            color=line_color,
            linewidth=line_width,
        )
    if cld_pts:
        cld_pts = np.array(cld_pts)
        ax.scatter(cld_pts[:, 0], cld_pts[:, 1], color='red', s=2, label='Raw Mesh Closest Points')
        ax.legend()

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title is None:
        title = (
            f"{block_name} wiremesh | cell i={i0c}:{i1c}, "
            f"j={j0c}:{j1c}, k=0"
        )
    ax.set_title(title)
    ax.grid(True, alpha=0.2)

    # Keep interactive windows open until the user closes them manually.
    if interactive:
        plt.ioff()
        plt.show(block=True)
    else:
        plt.show()

    return fig, ax, sub

def extract_and_resample_wss(block_subgrid, target_ni):
    """
    Extracts wallShearStress from the j=0 wall and resamples it 
    to match the new 1024 topology.
    """
    ni, nj, nk = block_subgrid.dimensions
    
    # 1. Grab the raw WSS vector field (usually X, Y, Z components)
    if 'wallShearStress' not in block_subgrid.point_data:
        # If this is a wake block with no wall, just return zeros
        return np.zeros((3, target_ni), dtype=np.float32)
        
    wss_flat = block_subgrid.point_data['wallShearStress']
    
    # 2. Reshape to (Ni, Nj, 3 components) and slice strictly at j=0
    wss_2d = wss_flat.reshape((ni, nj, nk, 3), order='F')[:, :, 0, :]
    wss_wall_true_length = wss_2d[:, 0, :] # Shape: [Ni, 3]
    
    # 3. 1D PCHIP to the target length
    old_s = np.linspace(0, 1, ni)
    new_s = np.linspace(0, 1, target_ni)
    
    wss_resampled = np.zeros((3, target_ni), dtype=np.float32)
    for comp in range(3): # For X, Y, Z shear components
        interp = PchipInterpolator(old_s, wss_wall_true_length[:, comp])
        wss_resampled[comp, :] = interp(new_s)
        
    return wss_resampled


def build_resampled_structured_subgrids(resampled_blocks, block_order, ml_features):
    """
    Convert resampled block dictionaries into chain-ordered StructuredGrids.

    This adapts the ML-friendly dict format:
      resampled_blocks[name] = {'data': [C, ni, nj], 'x': [ni, nj], 'y': [ni, nj], ...}
    into the geometry/data format expected by concatenate_blocks:
      [(name, StructuredGrid), ...]
    """
    subgrids = []
    for name in block_order:
        if name not in resampled_blocks:
            raise KeyError(f"Missing block '{name}' in resampled_blocks.")

        block = resampled_blocks[name]
        d = np.asarray(block['data'])
        x = np.asarray(block['x'])
        y = np.asarray(block['y'])

        if d.ndim != 3:
            raise ValueError(f"Block '{name}' data must be [C, ni, nj], got shape {d.shape}.")
        if x.shape != y.shape:
            raise ValueError(f"Block '{name}' x/y shape mismatch: {x.shape} vs {y.shape}.")

        c, ni, nj = d.shape
        if x.shape != (ni, nj):
            raise ValueError(
                f"Block '{name}' coordinate shape {x.shape} incompatible with data shape {(c, ni, nj)}."
            )
        if len(ml_features) != c:
            raise ValueError(
                f"Block '{name}' channel mismatch: len(ml_features)={len(ml_features)} vs C={c}."
            )

        g = pv.StructuredGrid()
        g.dimensions = (ni, nj, 1)

        pts = np.zeros((ni, nj, 1, 3), dtype=np.float64)
        pts[:, :, 0, 0] = x
        pts[:, :, 0, 1] = y
        g.points = pts.reshape((-1, 3), order='F')

        for idx, feat in enumerate(ml_features):
            g.point_data[feat] = d[idx].ravel(order='F')

        subgrids.append((name, g))

    return subgrids


def get_cmesh(SIM_PATH, VTM_PATH, FOAM_PATH,  J_MAX, target_ni_map, simulation):
    

    U_inf = simulation.inlet_velocity
    nu = simulation.NU

    OUT_DIR = "structured_block"

    output_type = 'vts'

    if not os.path.exists(FOAM_PATH):
        os.system(f"touch {FOAM_PATH}")  # create empty file to satisfy PyVista reader

    # Read OpenFOAM results using PyVista's OpenFOAM reader also converts to point data and merges blocks
    foam_combined = read_openfoam_results(FOAM_PATH, fields_to_keep=['U','p','nut','wallShearStress'])

    # Combine all blocks into a single unstructured grid.

    # use_all_points=True keeps ghost/boundary points
    print("\nCombining OpenFOAM blocks into a single mesh for interpolation...")
    #foam_combined = foam_results.combine(merge_points=True)
    #foam_combined = foam_results.read().merge(progress_bar=True)  # Use the reader's built-in merging with progress bar for large meshes
    print(f"\nCombined OpenFOAM mesh: type={type(foam_combined).__name__}  "
          f"pts={foam_combined.n_points}  cells={foam_combined.n_cells}")
    print(f"Available arrays: {foam_combined.array_names}")

    # add additional point data needed for interpolation and physics-informed ML features


    foam_combined["nut_ratio_cuberoot"] = np.cbrt((foam_combined["nut"]+1e-12) / nu)
    foam_combined["Cp"] = foam_combined["p"] / (0.5 * U_inf**2)  # Note no rho because incompressible Openfoam results 
    foam_combined["U_of_x"] = foam_combined["U"][:, 0]  # X-component of velocity
    foam_combined["U_of_y"] = foam_combined["U"][:, 1]  # Y-component of velocity
    
    for target in ["nut", "nut_ratio_cuberoot"]:
        # 2. Count the 'Poison'
        nan_count = np.isnan(foam_combined["nut"]).sum()
        inf_count = np.isinf(foam_combined["nut"]).sum()
        total_bad = nan_count + inf_count

        if total_bad > 0:
            print(f"❌ Skipping {SIM_PATH}: Found {total_bad} non-finite values in {target} field.")
            return None # Or handle the skip in your worker
        else:
            print(" Found no NANs in {target}  from {SIM_PATH}")

    # Drop the U Vector field to avoid confusion and potential issues with sampling; we will reconstruct it from the components if needed
    if 'U' in foam_combined.cell_data:  foam_combined.cell_data.remove('U')
    if 'U' in foam_combined.point_data: foam_combined.point_data.remove('U')
    
    print(" Read Structured Grids from VTM file for structured block-wise processing...")
    print(" Note that VTS files must be written with os.precision(16) to avoid single precision rounding issues!")
    grids = read_structured_grids_from_vtm(VTM_PATH)

    # Physical traversal order around the C-mesh (clockwise, i increases along chain):
    #   block_0 → wake (forward, toward TE)
    #   block_3 → lower surface TE region
    #   block_5 → lower surface LE / LE wrap
    #   block_4 → upper surface LE region
    #   block_2 → upper surface TE region
    #   block_1 → wake (back, away from TE)
    BLOCK_ORDER = ['block_0', 'block_3', 'block_5', 'block_4', 'block_2', 'block_1']

    grid_by_name = {name: grid for name, grid in grids}

    # extract_subset takes a single flat extent: [i_min, i_max, j_min, j_max, k_min, k_max]
    #J_MAX = 216
    #I0_min = 58
    # extract subgrids with correct i extents, keeping all j points up to J_MAX
    print(f"\nExtracting subgrids with specified i extents and J_MAX limit...{J_MAX}")
    subgrids_raw = []
    num_wake_cells = 0
    for bname in BLOCK_ORDER:
        grid = grid_by_name[bname]
        ni, nj, nk = grid.dimensions
        if bname == 'block_0':
            sub = grid.extract_subset([0, ni - 1, 0, min(J_MAX, nj - 1), 0, 1])
            num_wake_cells = sub.dimensions[0]  # length along i-axis
        elif bname == 'block_1':
            sub = grid.extract_subset([0, ni - 1, 0, min(J_MAX, nj - 1), 0, 1])
        else:
            sub = grid.extract_subset([0, ni - 1, 0, min(J_MAX, nj - 1), 0, 1])

        # resample to new std block size
        subgrids_raw.append((bname, sub))
        print(f"'{bname}': {grid.dimensions} → {sub.dimensions}  "
              f"pts={sub.n_points}  cells={sub.n_cells}")
        
    print("\nExporting non-sampled fields on each original blockMesh block for debugging...")
    # write_sampled_blocks_vtm(
    #     subgrids_raw,
    #     SIM_PATH,
    #     name="non_sampled_blockmesh"
    # )
    # plot_subgrids(subgrids_raw, title="Raw Extracted Subgrids Before Resampling")

    # Optional local wiremesh debug plot for a selected block/range.
    plot_block = 'block_5'
    grid = grid_by_name[plot_block]
    ni, nj, nk = [int(v) for v in grid.dimensions]


    print("\nChain-stitching i-axis orientations:")
    subgrids = chain_stitch_orientation(subgrids_raw)

    # Sample the stitched-but-unresampled blocks first so raw export can include
    # the full flow fields from foam_combined without changing topology.
    sampled_subgrids = []
    for bname, sub in subgrids:
        sampled_sub = sub.sample(foam_combined, snap_to_closest_point=True)
        sampled_subgrids.append((bname, sampled_sub))




    print("\nExporting sampled fields on each original blockMesh block for debugging...")
    # write_sampled_blocks_vtm(
    #     sampled_subgrids,
    #     SIM_PATH,
    #     name="sampled_blockmesh_fields"
    # )


    # export the stitched raw C-mesh for visualization and sanity checking in Paraview; this is the stitched but not yet resampled data on the original grid topology
    merged = concatenate_blocks(sampled_subgrids, z_ref=0)

    # _ = plot_vtm_block_wiremesh(
    #     merged,
    #     block_name=plot_block,
    #     i_cell_range=(200,300),
    #     j_cell_range=(0, 100),
    #     pt_cloud=foam_combined.points,
    #     filled=True,
    #     channel='U_of_x',
    #     cmap_vmin = -0.1,
    #     cmap_vmax = 1,
    #     title=f"Original {plot_block} sampled filled mesh with closest points from OpenFOAM mesh",
    #     interactive=True,
    # )

    print("\nExporting stitched but not yet resampled C-mesh to VTM for Paraview visualization...")
    # write_stitched_raw_subgrids_vtm(
    #     sampled_subgrids,
    #     SIM_PATH, 
    #     name="stitched_cmesh_raw"
    # )

    
    # Dictionary to hold our final resampled numpy arrays for each block
    resampled_blocks = {}
    # 1. Initialize empty list and strict exclusions
    ML_FEATURES = [] 

    EXCLUDE_FROM_VOLUMETRIC = ["wallShearStress"] # Must exclude the raw U vector!


    for name, sub_sampled in sampled_subgrids:
        print(f"Sampling '{name}'...")


        # Jacobian metric tensors at both points and cell centres for physics-informed interpolation
        jac_point, jac_center = compute_jacobian_metrics(sub_sampled)
        
        # Add the Jacobian metrics to the point data of the subgrid for later use 
        for key, arr in jac_point.items():
            print(f"Adding point data '{key}' to '{name}' with shape {arr.shape} and range [{arr.min():.3e}, {arr.max():.3e}]")
            # need both z-planes
            nz = sub_sampled.dimensions[2]
            arr = np.tile(arr, nz)
            sub_sampled.point_data[key] = arr

        # Lock the dynamic feature order on the first block
        if not ML_FEATURES:
            all_keys = list(sub_sampled.point_data.keys())
            # Sorting guarantees the same channel indices every single run
            DROP_CHANNELS = ['vtkGhostType', 'vtkValidPointMask', 'blockIndex']
            # 2. Build the final clean list
            ML_FEATURES = sorted([
                k for k in all_keys 
                if k not in EXCLUDE_FROM_VOLUMETRIC and k not in DROP_CHANNELS
            ])
            print(f"Locked in ML_FEATURES channel order: {ML_FEATURES}")   

        
        print(f"ML_FEATURES channel order: {ML_FEATURES}")   

        block_data, block_x, block_y = extract_arrays(sub_sampled, ML_FEATURES)

        # --- PRE-RESAMPLE SANITY CHECK ---
        features_to_check = ML_FEATURES # or a subset like ['p', 'U_of_x', 'nut']
        for idx, feat_name in enumerate(ML_FEATURES):
            data_slice = block_data[idx, :, :]
            if not np.isfinite(data_slice).all():
                num_nans = np.isnan(data_slice).sum()
                print(f"⚠️ [Data Prep] Patching {num_nans} non-finite values in: {feat_name}")

                # Create a mask of valid points
                mask = np.isfinite(data_slice)
                
                # If there are only a few NaNs, we interpolate from neighbors
                if num_nans < 10: 
                    # Get coordinates of all points
                    x_indices, y_indices = np.indices(data_slice.shape)
                    
                    # Use nearest neighbor interpolation to fill the holes
                    interp = interpolate.NearestNDInterpolator(
                        np.column_stack((x_indices[mask], y_indices[mask])), 
                        data_slice[mask]
                    )
                    
                    # Fill only the NaN spots
                    data_slice[~mask] = interp(np.column_stack((x_indices[~mask], y_indices[~mask])))
                    block_data[idx, :, :] = data_slice
                    print(f"   ✅ Successfully patched {feat_name} via Nearest Neighbor.")
                else:
                    # If the whole field is junk, we still want to know
                    raise ValueError(f"CRITICAL: Too many NaNs ({num_nans}) in {feat_name}. Simulation is likely diverged.")

        # 4. Apply the 1D PCHIP Resampling to this specific block
        # Resampling in the i_direction along constant j slices
        # To common size in i-direction 
        target_ni = target_ni_map[name]
        d_resampled, x_resampled, y_resampled = resample_block_pchip(
            block_data, block_x, block_y, target_ni=target_ni, block_name=name
        )
        # Extract wall shear stress 
        wss_blk = extract_and_resample_wss(sub_sampled, target_ni)


        # Save to dictionary for concatenation
        resampled_blocks[name] = {
            'data': d_resampled,
              'x': x_resampled,
              'y': y_resampled,
              'w': wss_blk
        }
    

    # ── Concatenate into a single StructuredGrid and save ─────────────────────
    print("\nConcatenating blocks along i-axis...")
    resampled_subgrids = build_resampled_structured_subgrids(resampled_blocks, BLOCK_ORDER, ML_FEATURES)
    mesh_concat = concatenate_blocks(resampled_subgrids, z_ref=0)
    
    ni_concat, nj_concat, _ = [int(v) for v in mesh_concat.dimensions]
    # _ = plot_vtm_block_wiremesh(
    #     mesh_concat,
    #     block_name="Resampled Mesh",
    #     i_cell_range=(200,300),
    #     j_cell_range=(0, 50),
    #     filled=True,
    #     channel='U_of_x',
    #     cmap_vmin = -0.1,
    #     cmap_vmax = 1.0,
    #     title=f"Resampled Mesh Interpolated  filled mesh with closest points from OpenFOAM mesh",
    #     interactive=True,
    # )
    # ── 5. Assemble the Final 1024 Tensor ─────────────────────
    # Pass the dictionary directly to the new assembler
    master_tensor, master_x, master_y , master_w = concatenate_blocks_ml_tensor(resampled_blocks, BLOCK_ORDER)
    
    return master_tensor, master_x, master_y, master_w, ML_FEATURES
    

def cell_centers(X, Y):
    """Return cell-centre coordinates for a structured grid.

    For a node array of shape (n_eta, n_xi) the cell centres are the
    average of the 4 surrounding corner nodes, giving shape
    (n_eta-1, n_xi-1).  Using cell centres avoids sampling on or
    immediately adjacent to the airfoil surface.
    """
    Xc = 0.25 * (X[:-1, :-1] + X[:-1, 1:] + X[1:, :-1] + X[1:, 1:])
    Yc = 0.25 * (Y[:-1, :-1] + Y[:-1, 1:] + Y[1:, :-1] + Y[1:, 1:])
    return Xc, Yc

def run_panel_method(x_wall, y_wall, X, Y,
                     alpha_aoa=-5.0, U_inf=1.0,path='./potential'):

    # need to flip coordinates from CW to CCW order for PanelFoil convention; skip last point since it's a duplicate of the first (closed loop)                 

    x_ccw = x_wall[-1:0:-1]
    y_ccw = y_wall[-1:0:-1]

    panel_foil = vortexSourcePanelfoil.PanelFoil(x_ccw, y_ccw, U_inf, alpha_aoa)
    #panel_foil.plot_panels()
    #panel_foil.debug_normals()

    panel_foil.solve_vortex_and_source_strengths()
    panel_foil.compute_tangential_velocities()
    panel_foil.compute_pressure_coefficients()
    panel_foil.plot_pressure_coefficients(path=f'{path}/pressure_coefficients.png')
    plt.close('all')
    panel_foil.check_solution()
    cl_pot = panel_foil.compute_lift_coefficient()
    panel_foil.compute_coefficients_from_pressure()

    # Compute velocity field on the full grid; j=0 (wall) is singular so we
    # evaluate at j=1 and copy those values down to j=0 afterwards.
    U_out, V_out, *_ = panel_foil.compute_velocity_field(X, Y)
    # Calculate velocity at points off the wall (j=1) and copy down to wall (j=0) to avoid singularity issues; 
    U_out[:, 0] = U_out[:, 1]
    V_out[:, 0] = V_out[:, 1]

    U  = U_out
    V  = V_out
    Cp = 1.0 - (U_out**2 + V_out**2) / U_inf**2
    Cp[:, 0] = Cp[:, 1]

    return X, Y, U, V, Cp, cl_pot

def plot_physical_space(data_dict, path, name, show_plots=False, nu=1e-5):
    """
    Plots the physical C-mesh space using purely Matplotlib.
    
    Args:
        data_dict: Dictionary mapping field names to 2D numpy arrays of shape (Ni, Nj).
                   MUST contain "X" and "Y" coordinate arrays.
        path: Base directory to save the plots.
        name: Name of the simulation/airfoil.
        show_plots: Boolean to display plots interactively.
    """
    phys_dir = os.path.join(path, "physical")
    if not os.path.exists(phys_dir):
        os.makedirs(phys_dir)

    # Extract coordinates
    X = data_dict.get("X")
    Y = data_dict.get("Y")
    ni,nj = X.shape
    if X is None or Y is None:
        raise ValueError("data_dict must contain 'X' and 'Y' coordinate arrays.")

    # Helper function for scalar fields
    def plot_scalar(ax, field_name, title, clims=None):
        if field_name == "nut":
            data_dict['nut'] = np.power(data_dict['nut_ratio_cuberoot'], 3)* nu - 1e-12 # Invert the cube root transformation to get back to nut for plotting
        if field_name not in data_dict:
            ax.axis('off')
            return
            
        data = data_dict[field_name]
        
        # pcolormesh handles the curvilinear grid perfectly
        kwargs = {'cmap': 'RdBu_r', 'shading': 'nearest'}
        if clims:
            kwargs['vmin'] = clims[0]
            kwargs['vmax'] = clims[1]
            
        mesh = ax.pcolormesh(X, Y, data, **kwargs)
        ax.set_title(title)


        # Overlay mesh lines using minor ticks
        stride_i = max(1, data.shape[0] // 10)
        stride_j = max(1, data.shape[1] // 10)
        ax.set_xticks(np.arange(-0.5, data.shape[0], stride_i), minor=True)
        ax.set_yticks(np.arange(-0.5, data.shape[1], stride_j), minor=True)
        ax.grid(which='minor', color='white', linewidth=0.3, alpha=0.5)
        ax.tick_params(which='minor', length=0)
        
        # Enforce physical aspect ratio so the airfoil isn't distorted
        ax.set_aspect('equal', adjustable='box')
        plt.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)

    # ---------------------------------------------------------
    # 1. Aerodynamics Plot (3x3)
    # ---------------------------------------------------------
    fig1, axes1 = plt.subplots(3, 3, figsize=(18, 12))
    fig1.suptitle(f"{name} - Aerodynamics (Physical Space) X{ni}_Y{nj}", fontsize=16)

    data_plots = [
        (0, 0, "Cp", "Cp", (-1.0, 1.0)),
        (0, 1, "U_x", "U_x", (-0.2, 1.2)),
        (0, 2, "U_y", "U_y", (-0.2, 0.2)),
        (1, 0, "Cp_pot", "Cp_pot", (-1.0, 1.0)),
        (1, 1, "U_x_pot", "U_x_pot", (-0.2, 1.2)),
        (1, 2, "U_y_pot", "U_y_pot", (-0.2, 0.2)),
        (2, 0, "Cp_delta", "Cp_delta", (-0.1, 0.1)),
        (2, 1, "U_x_delta", "U_x_delta", (-0.1, 0.1)),
        (2, 2, "U_y_delta", "U_y_delta", (-0.1, 0.1))
    ]



    for r, c, field_name, label, clims in data_plots:
        plot_scalar(axes1[r, c], field_name, label, clims)
        # Now these limits will actually work and the airfoil will stay proportional
        axes1[r, c].set_xlim(-4, 6)
        axes1[r, c].set_ylim(-4, 4) # Note: -5 to 5 is usually enough for Y if X is -5 to 10

    plt.tight_layout()

    fig1.savefig(os.path.join(phys_dir, f"{name}_physical_space_X{ni}_Y{nj}.png"), dpi=300, bbox_inches='tight')
    if show_plots: plt.show()
    plt.close(fig1)

    # ---------------------------------------------------------
    # 2. Vector Field Plot (1x3)
    # ---------------------------------------------------------
    fig0, axes0 = plt.subplots(1, 3, figsize=(18, 5))
    fig0.suptitle(f"{name} - Normalized Velocity Vectors X{ni}_Y{nj}", fontsize=16)
    
    # Subsampling strides (equivalent to your PyVista rate=(15, 5, 1))
    s_i, s_j = 15, 5
    X_sub = X[::s_i, ::s_j]
    Y_sub = Y[::s_i, ::s_j]

    vector_plots = [
        (0, "U_x", "U_y", "U_rans_field", (-0.2, 0.2)),
        (1, "U_x_pot", "U_y_pot", "U_pot_field", (-0.2, 0.2)),
        (2, "U_x_delta", "U_y_delta", "U_delta_field", (-0.1, 0.1))
    ]

    for c, ux_name, uy_name, label, clims in vector_plots:
        ax = axes0[c]
        if ux_name not in data_dict or uy_name not in data_dict:
            ax.axis('off')
            continue
            
        U = data_dict[ux_name][::s_i, ::s_j]
        V = data_dict[uy_name][::s_i, ::s_j]
        
        # Normalize vectors to match PyVista visual
        mag = np.hypot(U, V) + 1e-8
        U_norm = U / mag
        V_norm = V / mag
        
        # Plot airfoil outline (the wall j=0)
        ax.plot(X[:, 0], Y[:, 0], color='black', linewidth=1.5)
        
        # Quiver plot, colored by the V-component to match your PyVista script
        q = ax.quiver(X_sub, Y_sub, U_norm, V_norm, V, 
                      cmap="RdBu_r", scale=25, width=0.003, clim=clims)
        
        ax.set_title(label)
        ax.set_aspect('equal', adjustable='datalim')
        
        # Zoom in tight around the airfoil for better arrow visibility
        ax.set_xlim([np.min(X[:, 0]) - 0.5, np.max(X[:, 0]) + 1.5])
        ax.set_ylim([np.min(Y[:, 0]) - 1.0, np.max(Y[:, 0]) + 1.0])
        plt.colorbar(q, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig0.savefig(os.path.join(phys_dir, f"{name}_vectors_physical_space_X{ni}_Y{nj}.png"), dpi=300, bbox_inches='tight')
    if show_plots: plt.show()
    plt.close(fig0)

    # ---------------------------------------------------------
    # 3. Mesh Coefficients Plot (2x3)
    # ---------------------------------------------------------
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 8))
    fig2.suptitle(f"{name} - Mesh Coefficients (Physical Space) X{ni}_Y{nj}", fontsize=16)

    metric_plots = [
        (0, 0, "x_xi", "x_xi"),   (0, 1, "x_eta", "x_eta"),
        (1, 0, "y_xi", "y_xi"),   (1, 1, "y_eta", "y_eta"),
        (1, 2, "det_J", "det_J")
    ]

    for r, c, field_name, label in metric_plots:
        plot_scalar(axes2[r, c], field_name, label)
        
    axes2[0, 2].axis('off') # Hide empty subplot
    plt.tight_layout()
    fig2.savefig(os.path.join(phys_dir, f"{name}_meshcoeffs_X{ni}_Y{nj}.png"), dpi=300, bbox_inches='tight')
    if show_plots: plt.show()
    plt.close(fig2)

    # ---------------------------------------------------------
    # 4. Turbulence and SDF Plot (2x2)
    # ---------------------------------------------------------
    fig3, axes3 = plt.subplots(2, 2, figsize=(12, 8))
    fig3.suptitle(f"{name} - SDF & Turbulence (Physical Space) X{ni}_Y{nj}", fontsize=16)

    turb_plots = [
        (0, 0, "nut", "nut", (0,0.005)),
        (1, 0, "nut_ratio_cuberoot", "(nut/nu)^(1/3)", (0,3)),
        (0, 1, "sdf", "sdf", (0,5)),
        (1, 1, "exp_sdf", "exp(-k*sdf)", (0,1))
    ]

    for r, c, field_name, label, clims in turb_plots:
        plot_scalar(axes3[r, c], field_name, label, clims)
        axes3[r, c].set_xlim(-4, 6)
        axes3[r, c].set_ylim(-4, 4) # Note: -5 to 5 is usually enough for Y if X is -5 to 10

    plt.tight_layout()
    fig3.savefig(os.path.join(phys_dir, f"{name}_sdf_nut_X{ni}_Y{nj}.png"), dpi=300, bbox_inches='tight')
    if show_plots: plt.show()
    plt.close(fig3)

def debug_plot_latent_space(data_dict, path, name, show_plots=False, nu=1e-5):
    """Quick debug view: U_x and nut side-by-side in latent (computational) space."""
    latent_dir = os.path.join(path, "latent")
    os.makedirs(latent_dir, exist_ok=True)
    ni, nj = data_dict.get("X", np.zeros((1, 1))).shape

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{name} - debug latent  X{ni}_Y{nj}", fontsize=14)

    ax = axes[0]
    if "U_x" in data_dict:
        im = ax.imshow(data_dict["U_x"].T, cmap="jet", aspect="auto",
                       origin="lower", vmin=-0.1, vmax=1.0)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("U_x")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[1]
    if "nut_ratio_cuberoot" in data_dict:
        nut = np.power(data_dict["nut_ratio_cuberoot"], 3) * nu - 1e-12
        im = ax.imshow(nut.T, cmap="viridis", aspect="auto",
                       origin="lower", vmin=0.0, vmax=5e-4)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("nut")
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    fig.savefig(os.path.join(latent_dir, f"{name}_debug_latent_X{ni}_Y{nj}.png"),
                dpi=150, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)

def plot_latent_space(data_dict, path, name, show_plots=False, nu = 1e-5):
    """
    Plots the computational (latent) space using purely Matplotlib.
    
    Args:
        data_dict: Dictionary mapping field names to 2D numpy arrays of shape (Ni, Nj).
        path: Base directory to save the plots.
        name: Name of the simulation/airfoil.
        show_plots: Boolean to display plots interactively.
    """
    latent_dir = os.path.join(path, "latent")
    if not os.path.exists(latent_dir):
        os.makedirs(latent_dir)
    ni,nj    = data_dict.get("X", np.zeros((1,1))).shape

    # Helper function to plot a single subplot
    def plot_field(ax, field_name, title, clims=None):
        if field_name == "exp_sdf":
            # Now created on the fly instead of saving to disk
            data_dict['exp_sdf'] = np.exp(-5.0 * data_dict['sdf'])  # Example transformation for better visualization
        if field_name == "nut":
            # Invert the cube root transformation to get back to nut for plotting
            data_dict['nut'] = np.power(data_dict['nut_ratio_cuberoot'], 3) * nu - 1e-12
        if field_name not in data_dict:
            ax.axis('off')
            return
            
        data = data_dict[field_name]
        
        # .T transposes (Ni, Nj) to (Nj, Ni) so 'i' is the X-axis and 'j' is the Y-axis
        # origin='lower' puts the wall (j=0) at the bottom of the image
        kwargs = {'cmap': 'RdBu_r', 'aspect': 'auto', 'origin': 'lower'}
        if clims:
            kwargs['vmin'] = clims[0]
            kwargs['vmax'] = clims[1]
            
        im = ax.imshow(data.T, **kwargs)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Overlay mesh lines using minor ticks
        stride_i = max(1, data.shape[0] // 10)
        stride_j = max(1, data.shape[1] // 10)
        ax.set_xticks(np.arange(-0.5, data.shape[0], stride_i), minor=True)
        ax.set_yticks(np.arange(-0.5, data.shape[1], stride_j), minor=True)
        ax.grid(which='minor', color='white', linewidth=0.3, alpha=0.5)
        ax.tick_params(which='minor', length=0)

        # Hide major tick labels for a clean "latent" look
        ax.set_xticks([])
        ax.set_yticks([])

    # ---------------------------------------------------------
    # 1. Aerodynamics Plot (3x3)
    # ---------------------------------------------------------
    fig1, axes1 = plt.subplots(3, 3, figsize=(15, 12))
    fig1.suptitle(f"{name} - Aerodynamics (Latent Space) X{ni}_Y{nj}", fontsize=16)

    data_plots = [
        (0, 0, "Cp", "Cp", (-1.0, 1.0)),
        (0, 1, "U_x", "U_x", (-0.2, 1.2)),
        (0, 2, "U_y", "U_y", (-0.2, 0.2)),

        (1, 0, "Cp_pot", "Cp_pot", (-1.0, 1.0)),
        (1, 1, "U_x_pot", "U_x_pot", (-0.2, 1.2)),
        (1, 2, "U_y_pot", "U_y_pot", (-0.2, 0.2)),

        (2, 0, "Cp_delta", "Cp_delta", (-0.1, 0.1)),
        (2, 1, "U_x_delta", "U_x_delta", (-0.1, 0.1)),
        (2, 2, "U_y_delta", "U_y_delta", (-0.1, 0.1))
    ]

    for r, c, field_name, label, clims in data_plots:
        plot_field(axes1[r, c], field_name, label, clims)

    plt.tight_layout()
    fig1.savefig(os.path.join(latent_dir, f"{name}_latent_space_X{ni}_Y{nj}.png"), dpi=300, bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close(fig1)

    # ---------------------------------------------------------
    # 2. Mesh Metrics Plot (2x3)
    # ---------------------------------------------------------
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8))
    fig2.suptitle(f"{name} - Mesh Coefficients (Latent Space) X{ni}_Y{nj}", fontsize=16)

    metric_plots = [
        (0, 0, "x_xi", "x_xi"),   (0, 1, "x_eta", "x_eta"),
        (1, 0, "y_xi", "y_xi"),   (1, 1, "y_eta", "y_eta"),
        (1, 2, "det_J", "det_J")
    ]

    for r, c, field_name, label in metric_plots:
        plot_field(axes2[r, c], field_name, label)
        
    # Hide the empty subplot at (0, 2)
    axes2[0, 2].axis('off')

    plt.tight_layout()
    fig2.savefig(os.path.join(latent_dir, f"{name}_meshcoeffs_X{ni}_Y{nj}.png"), dpi=300, bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close(fig2)

    # ---------------------------------------------------------
    # 3. Turbulence and SDF Plot (2x2)
    # ---------------------------------------------------------
    fig3, axes3 = plt.subplots(2, 2, figsize=(10, 8))
    fig3.suptitle(f"{name} - SDF & Turbulence (Latent Space) X{ni}_Y{nj}", fontsize=16)

    turb_plots = [
        (0, 0, "nut", "nut", (0,0.005)),
        (1, 0, "nut_ratio_cuberoot", "(nut/nu)^(1/3)", (0,3)),
        (0, 1, "sdf", "sdf", (0,5)),
        (1, 1, "exp_sdf", "exp(-k*sdf)", (0,1))
    ]

    for r, c, field_name, label, clims in turb_plots:
        plot_field(axes3[r, c], field_name, label, clims)

    plt.tight_layout()
    fig3.savefig(os.path.join(latent_dir, f"{name}_sdf_nut_X{ni}_Y{nj}.png"), dpi=300, bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close(fig3)

def save_to_pytorch( path, name, archive_dict):

    if not os.path.exists(path):
        os.makedirs(path)
    print(f"Preparing to save tensors for {name} with shape {archive_dict['x'].shape} and properties {list(archive_dict.keys())}")

    nc, nx, ny = archive_dict['x'].shape

    print(f"Saving tensors to PyTorch file: {name}_C_mesh_{nx}x{ny}.pt")

    # Save the tensors to a file
    save_path = os.path.join(path, f"{name}_C_mesh_{nx}x{ny}.pt")
    torch.save(archive_dict, save_path)
    print(f"Saved cell data tensors to {save_path}")

def compute_nodal_metrics(X_grid, Y_grid):
    """
    Computes the Jacobian metrics for a 2D grid.
    Assumes X_grid and Y_grid are shape (n_xi, n_eta).
    Used to recalculate new jacobians on downsampled grids
    """
    # np.gradient computes derivatives along axis 0 (xi) and axis 1 (eta)
    x_xi, x_eta = np.gradient(X_grid, axis=(0, 1))
    y_xi, y_eta = np.gradient(Y_grid, axis=(0, 1))
    
    # Calculate the determinant of the Jacobian matrix
    det_J = (x_xi * y_eta) - (x_eta * y_xi)
    
    return x_xi, x_eta, y_xi, y_eta, det_J


def calculate_sdf(master_tensor, master_x, master_y, k=5.0):
    """
    Calculates the SDF using the NumPy coordinate grids and appends 
    exp_sdf as a new channel to the master_tensor.
    """
    print("\nCalculating SDF on the 1024x216 master grid...")
    wake_length_start = 128
    airfoil_start = wake_length_start
    airfoil_end = 1024 - 127 # Exclude the trailing wake block
    # 1. Extract the Wall Coordinates
    # In your stitched topology, the wall is strictly at index j=0
    wall_x = master_x[airfoil_start:airfoil_end, 0]  # Shape: (1024,)
    wall_y = master_y[airfoil_start:airfoil_end, 0]  # Shape: (1024,)
    wall_xy = np.column_stack((wall_x, wall_y)) # Shape: (1024, 2)
    
    # 2. Flatten all coordinates for the KDTree query
    all_x = master_x.flatten()
    all_y = master_y.flatten()
    all_xy = np.column_stack((all_x, all_y))    # Shape: (221184, 2)
    
    # 3. Query the KDTree
    tree = KDTree(wall_xy)
    sdf_flat, _ = tree.query(all_xy, workers=-1)
    
    # 4. Reshape back to the 2D spatial grid
    sdf_2d = sdf_flat.reshape(master_x.shape)   # Shape: (1024, 216)
    sdf_fixed = np.nan_to_num(sdf_2d, nan=1e-12).astype(np.float32)
        
    print(f"SDF Range: min={sdf_fixed.min():.6f}, max={sdf_fixed.max():.6f}, mean={sdf_fixed.mean():.6f}")

    # 6. Append exp_sdf to the master_tensor
    # We add a dummy channel dimension [1, 1024, 216] so it concatenates properly
    sdf_expanded = np.expand_dims(sdf_fixed, axis=0)
    master_tensor_updated = np.concatenate([master_tensor, sdf_expanded], axis=0)
    
    print(f"Updated Master Tensor Shape: {master_tensor_updated.shape}")
    
    return  sdf_fixed

def process_airfrans_calc_potentialflow(name, AF_ROOT, OF_ROOT, STORAGE_DIR,  J_MAX, output_type='vts', show_plots=False):
        # recommend using J_MAX = 217 to get the full 216 j points 
        print(f"AF_ROOT: {AF_ROOT}  OF_ROOT: {OF_ROOT}  STORAGE_DIR: {STORAGE_DIR}   J_MAX: {J_MAX}  output_type: {output_type}")
        # check if archive exists

        archive_out = f"{STORAGE_DIR}/{name}_C_mesh_1025x217.pt"
        #if os.path.exists(archive_out):
        #    print(f" Skipping {name}   Found existing file {archive_out}")
        #    return

        simulation = af.Simulation(root=AF_ROOT, name=name)
        AOA = simulation.angle_of_attack * 180 / np.pi
        U_inf = simulation.inlet_velocity
        nu = 1.56e-5  ##   need to use kinematic viscosity from OpenFoam simulations not simulation.NU
        rho = simulation.RHO
        reynolds =  U_inf / nu  
        log_re = np.log(reynolds)   

        print(f"\nProcessing simulation: {name}")
        SIM_PATH = f"{OF_ROOT}/{name}"
        VTM_PATH = f"{SIM_PATH}/constant/blockMeshVTK/blockMesh.vtm"
        FOAM_PATH = f"{SIM_PATH}/touch.foam"

        # Add +1 to account for the dropped overlap nodes during concatenation.
        # The final block ('block_1') stays at 128 because its last node is kept.
        target_ni_map = {
            'block_0': 129, 'block_3': 129, 'block_5': 257, 
            'block_4': 257, 'block_2': 129, 'block_1': 129
        }

        master_tensor, master_x, master_y, master_w, ML_FEATURES = get_cmesh(SIM_PATH, VTM_PATH, FOAM_PATH, J_MAX =J_MAX , target_ni_map=target_ni_map,simulation=simulation)
        print(f"Extracted C-mesh ML Tensor: type={type(master_tensor).__name__}  "
          f"data_shape={master_tensor.shape}  coords_shape={master_x.shape}  wss_shape={master_w.shape}")


        

        nx, ny, nz = master_tensor.shape
        # get leading edge and trailing edge indices from the target_ni_map
        I_START = target_ni_map['block_0']
        I_END = nx - target_ni_map['block_3']

        x_wall = master_x[I_START:I_END, 0]  # i varies along the wall
        y_wall = master_y[I_START:I_END, 0]
        print(f"Extracted wall points: {len(x_wall)}")
        # Airfrans wall points
        af_points = simulation.airfoil.points # (N,3)

        x_wall_af = af_points[:, 0]
        y_wall_af = af_points[:, 1]
        print(f"Extracted wall points from Airfrans: {len(x_wall_af)}")

        x_wall_af_max = np.max(x_wall_af)
        x_wall_af_min = np.min(x_wall_af)
        print(f"Airfrans wall x range: [{x_wall_af_min:.3f}, {x_wall_af_max:.3f}]")
        x_wall_max = np.max(x_wall)
        x_wall_min = np.min(x_wall)
        print(f"OpenFOAM wall x range: [{x_wall_min:.3f}, {x_wall_max:.3f}]")
        
        # Need to calculate the SDF /implicit distance from the wall to the first cell centres 
        sdf_grid = calculate_sdf( master_tensor, master_x, master_y )
        print (f"Running panel method with {len(x_wall)} wall points...")
        print(f"Angle of attack: {AOA:.2f} degrees")
        print(f"Freestream velocity: {U_inf:.2f} m/s")
        print(f" Shape of X: {master_x.shape}  Y: {master_y.shape} ")

        PATH_POTENTIAL = SIM_PATH + "/potential"
        os.makedirs(PATH_POTENTIAL, exist_ok=True)  

        X_off, Y_off, U_pot, V_pot, Cp_pot, cl_pot = run_panel_method(x_wall,y_wall, master_x, master_y,  alpha_aoa=AOA, U_inf=1.0, path=PATH_POTENTIAL)
        
        U_of_x = master_tensor[ML_FEATURES.index("U_of_x")]
        U_of_y = master_tensor[ML_FEATURES.index("U_of_y")]
        p_of   = master_tensor[ML_FEATURES.index("p")]
        nut = master_tensor[ML_FEATURES.index("nut")]
        nut_ratio_cuberoot = master_tensor[ML_FEATURES.index("nut_ratio_cuberoot")]
        # Extract the Jacobian metrics from the master tensor
        x_xi  = master_tensor[ML_FEATURES.index("x_xi_p")]
        x_eta = master_tensor[ML_FEATURES.index("x_eta_p")]
        y_xi  = master_tensor[ML_FEATURES.index("y_xi_p")]
        y_eta = master_tensor[ML_FEATURES.index("y_eta_p")]
        det_J = master_tensor[ML_FEATURES.index("det_p")]

        # 2. Normalize OpenFOAM CFD Data
        U_x = U_of_x / U_inf
        U_y = U_of_y / U_inf
        Cp_of = p_of / (0.5 * U_inf**2)  # Note no rho because incompressible Openfoam results 

        # 3. Calculate Deltas and ML Targets
        U_x_delta = U_x - U_pot
        U_y_delta = U_y - V_pot
        Cp_delta = Cp_of - Cp_pot
        # Lift from panel method vs Airfrans forces for verification

        print(f"From OpenFoam Shape of X: {master_x.shape}  Y: {master_y.shape}  ")
        
        
        x_data_spatial = np.stack([
            master_x, master_y,
            U_pot, V_pot,
            Cp_pot,
            sdf_grid,
            x_xi, x_eta, y_xi, y_eta, det_J  # Jacobian metrics
        ], axis=0)

        # Final X_Channel Map
        X_CHANNEL_MAP = {
            'X': 0, 
            'Y': 1,
            'U_pot': 2, 
            'V_pot': 3, 
            'Cp_pot': 4,
            'sdf': 5,
            'x_xi': 6, 
            'x_eta': 7, 
            'y_xi': 8, 
            'y_eta': 9, 
            'det_J': 10
        }
        y_delta_spatial = np.stack([ Cp_delta,
                                     U_x_delta,
                                     U_y_delta,
                                     nut_ratio_cuberoot
        ], axis=0)

        y_out_spatial = np.stack([
            U_of_x,
            U_of_y,
            p_of, 
            nut,
        ], axis=0)

        print(f"Input Tensor Shape: {x_data_spatial.shape}")      # Expected: (11, 1024, 216)
        print(f"Delta Target Shape: {y_delta_spatial.shape}")     # Expected: (4, 1024, 216)
        print(f"Full Output Shape:  {y_out_spatial.shape}")      # Expected: (4, 1024, 216)

        # Create a dictionary of features for plotting
        plot_dict = {
            "X": master_x,
             "Y": master_y,
            "Cp": Cp_of, "U_x": U_x, "U_y": U_y,
            "Cp_pot": Cp_pot, "U_x_pot": U_pot, "U_y_pot": V_pot,
            "Cp_delta": Cp_delta, "U_x_delta": U_x_delta, "U_y_delta": U_y_delta,
            "nut_ratio_cuberoot": nut_ratio_cuberoot,
            "sdf": sdf_grid, 
            # Jacobian metrics
            "x_xi": x_xi,
            "x_eta": x_eta,
            "y_xi": y_xi,
            "y_eta": y_eta,
            "det_J": det_J,
        }

        plot_physical_space(plot_dict, SIM_PATH, name, show_plots=False, nu = simulation.NU)
        
        debug_plot_latent_space(plot_dict, SIM_PATH, name, show_plots=False,nu = simulation.NU)

        plot_latent_space(plot_dict, SIM_PATH, name, show_plots=False,nu = simulation.NU)
        
        # Export stitched structured C-mesh as a single-block VTM for visualization.
        write_single_structured_cmesh_vtm(plot_dict, SIM_PATH, name=name)

        
        # Assemble tensors and save to PyTorch format for ML training

        # Calculate aerodynamic coefficients for verification
        ((cd, cdp, cdv), (cl, clp, clv)) = simulation.force_coefficient(compressible=False,reference=False)

        print(f" Airfrans alpha = {AOA:.4f} Cl = {cl:.5f}, Cl_pot = {cl_pot:.5f}, Cl_delta = {cl - cl_pot:.5f}")

        print(f"Shape of output array y_delta_out: {y_delta_spatial.shape}  sample: {y_delta_spatial[0]}")

        props = {
                'v_mag_inf': torch.tensor(U_inf, dtype=torch.float32),
                'aoa_deg':  torch.tensor(AOA, dtype=torch.float32),
                'nu_mol':  torch.tensor(nu, dtype=torch.float32),
                'rho':  torch.tensor(rho, dtype=torch.float32),
                'reynolds': torch.tensor(reynolds, dtype=torch.float32),
                'log_reynolds': torch.tensor(log_re, dtype=torch.float32),
                'cd': torch.tensor(cd, dtype=torch.float32),
                'cl': torch.tensor(cl, dtype=torch.float32),
                'cdp': torch.tensor(cdp, dtype=torch.float32),
                'clp': torch.tensor(clp, dtype=torch.float32),
                'cdv': torch.tensor(cdv, dtype=torch.float32),
                'clv': torch.tensor(clv, dtype=torch.float32),
                'cl_pot': torch.tensor(cl_pot, dtype=torch.float32),
                'name': name
            }


        # put tensors and properties into a dictionary for saving
        archive_dict = {
            'x': torch.tensor(x_data_spatial, dtype=torch.float32),
            'y_delta': torch.tensor(y_delta_spatial, dtype=torch.float32),
            'y': torch.tensor(y_out_spatial, dtype=torch.float32),
            'wss': torch.tensor(master_w, dtype=torch.float32),  # <--- Add master_w here
            'props': props
        }
        print(f" Save Complete Archive")
        save_to_pytorch( STORAGE_DIR, name, archive_dict)




# 1. Move the wrapper OUTSIDE any other function
def global_worker_wrapper(arg_tuple):
    """
    Takes a single tuple of arguments because executor.map 
    is happiest with a single iterable.
    """
    name, AF_ROOT, OF_ROOT, STORAGE_DIR,  J_MAX = arg_tuple
    failed_log = os.path.join(STORAGE_DIR, "failed_simulations.txt")
    try:
        return process_airfrans_calc_potentialflow(
            name,
            AF_ROOT=AF_ROOT,
            OF_ROOT=OF_ROOT,
            STORAGE_DIR=STORAGE_DIR,
            J_MAX=J_MAX
        )
    except Exception as e:
        import traceback
        error_msg = f"{name}: {type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(f"\n❌ ERROR processing '{name}':\n{error_msg}", flush=True)
        os.makedirs(STORAGE_DIR, exist_ok=True)
        with open(failed_log, "a") as f:
            f.write(f"{name}\t{type(e).__name__}: {e}\n")
        return None

if __name__ == "__main__":
    
    AF_ROOT = "/home/timm/Projects/PIML/Dataset"
    OF_ROOT = "/home/timm/storage/AF_NO_DATASET/OF_dataset"      
    STORAGE_DIR ="/home/timm/storage/AF_NO_DATASET/Archive"

    sim_dirs = [d.name for d in Path(OF_ROOT).iterdir() if d.is_dir()]
    #sim_dirs = sim_dirs[:15]
    #sim_dirs = ["airFoil2D_SST_37.983_9.999_6.128_2.638_10.544"]
    #sim_dirs = ["airFoil2D_SST_58.831_-3.563_2.815_4.916_10.078", "airFoil2D_SST_76.326_4.848_6.47_5.914_13.755", "airFoil2D_SST_52.526_2.212_4.618_3.827_10.191"]
    #sim_dirs = ["airFoil2D_SST_60.23_2.984_6.976_4.303_9.408"]
    # clip mesh in radial direction to take only the first 158 cells in j-direction to match airfrans training data and avoid large uniform regions far from the wall which can cause ML models to struggle with learning the important near-wall gradients
    J_MAX = 216 

    # 2. Prepare the list of argument tuples for the map
    # This bundles everything the worker needs into one package per simulation
    job_args = [
        (name, AF_ROOT, OF_ROOT,  STORAGE_DIR,  J_MAX) 
        for name in sim_dirs
    ]
    
    # If num_workers is None, it uses all available logical cores - 2.
    # May need to limit workers due to memory
    num_workers=4
    if num_workers is None:
        num_workers = max(1, os.cpu_count() - 2)
    
    print(f"🚀 Starting Multiprocessing with {num_workers} workers...")

    # We use ProcessPoolExecutor for CPU-bound tasks like PyVista sampling
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Map the sim_dirs to the processing function
        list(tqdm(executor.map(global_worker_wrapper, job_args), total=len(job_args), desc="Processing Airfoils"))
        


                




