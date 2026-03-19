
from curses import window
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import interp1d
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve

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

from BlockMeshInterpolator import chain_stitch_orientation, compute_jacobian_metrics, concatenate_blocks, read_openfoam_results, read_structured_grids_from_vtm

def get_cmesh(SIM_PATH, VTM_PATH, FOAM_PATH, I0_min, J_MAX, output_type='vts'):

    OUT_DIR = "structured_block"

    output_type = 'vts'

    if not os.path.exists(FOAM_PATH):
        os.system(f"touch {FOAM_PATH}")  # create empty file to satisfy PyVista reader


    foam_results = read_openfoam_results(FOAM_PATH)

    # Merge all OpenFOAM blocks into one UnstructuredGrid so we can call .interpolate()
    # use_all_points=True keeps ghost/boundary points; progress_bar for large meshes
    foam_combined = foam_results.combine(merge_points=True)
    print(f"\nCombined OpenFOAM mesh: type={type(foam_combined).__name__}  "
          f"pts={foam_combined.n_points}  cells={foam_combined.n_cells}")
    print(f"Available arrays: {foam_combined.array_names}")

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

    # need only first 158 points in j direction to cover cartesian domain used in Airfrans data
    # extract_subset takes a single flat extent: [i_min, i_max, j_min, j_max, k_min, k_max]
    #J_MAX = 158
    #I0_min = 58
    # extract subgrids with correct i extents, keeping all j points up to J_MAX
    subgrids_raw = []
    num_wake_cells = 0
    for bname in BLOCK_ORDER:
        grid = grid_by_name[bname]
        ni, nj, nk = grid.dimensions
        if bname == 'block_0':
            sub = grid.extract_subset([I0_min, ni - 1, 0, min(J_MAX, nj - 1), 0, 1])
            num_wake_cells = sub.dimensions[0]  # length along i-axis
        elif bname == 'block_1':
            sub = grid.extract_subset([I0_min, ni - 1, 0, min(J_MAX, nj - 1), 0, 1])
        else:
            sub = grid.extract_subset([0, ni - 1, 0, min(J_MAX, nj - 1), 0, 1])
        subgrids_raw.append((bname, sub))
        print(f"'{bname}': {grid.dimensions} → {sub.dimensions}  "
              f"pts={sub.n_points}  cells={sub.n_cells}")

    print("\nChain-stitching i-axis orientations:")
    subgrids = chain_stitch_orientation(subgrids_raw)

    # Keep only the fields we need in the source mesh
    CELL_FIELDS  = ["U", "p", "nut"]           # cell-centred → cell_data
    POINT_FIELDS = ["p", "wallShearStress"]     # node-interpolated → point_data
    KEEP = set(CELL_FIELDS + POINT_FIELDS)
    for arr in list(foam_combined.array_names):
        if arr not in KEEP:
            if arr in foam_combined.cell_data:  foam_combined.cell_data.remove(arr)
            if arr in foam_combined.point_data: foam_combined.point_data.remove(arr)
    print(f"Retained arrays in source: {foam_combined.array_names}")

    # ── Sample each block, compute Jacobians, collect ─────────────────────────
    # NOTE: vtkProbeFilter's internal cell locator is NOT thread-safe when the
    # same source (foam_combined) is shared across threads → segfault (exit 139).
    # VTK's own SMP threading already parallelises within each .sample() call,
    # so sequential block iteration is sufficient.
    sampled_list = []
    for name, sub in subgrids:
        print(f"Sampling '{name}'...")

        # Cell-centred fields: U, p, nut
        cc = sub.cell_centers()
        cc_sampled = cc.sample(foam_combined)
        out = sub.copy(deep=False)
        out.clear_data()
        for field in CELL_FIELDS:
            if field in cc_sampled.point_data:
                out.cell_data[field] = cc_sampled.point_data[field]
            else:
                print(f"  WARNING: cell field '{field}' not found for '{name}'")

        # Node fields: p, wallShearStress
        pts_sampled = sub.sample(foam_combined)
        for field in POINT_FIELDS:
            if field in pts_sampled.point_data:
                out.point_data[field] = pts_sampled.point_data[field]
            else:
                print(f"  WARNING: point field '{field}' not found for '{name}'")

        # Jacobian metric tensors at both points and cell centres for physics-informed interpolation
        jac_point,jac_center = compute_jacobian_metrics(sub)
        for key, arr in jac_center.items():
            out.cell_data[key] = arr
            print(f"Adding cell data '{key}' to '{name}' with shape {arr.shape} and range [{arr.min():.3e}, {arr.max():.3e}]")
        
        for key, arr in jac_point.items():
            print(f"Adding point data '{key}' to '{name}' with shape {arr.shape} and range [{arr.min():.3e}, {arr.max():.3e}]")
            # need both z-planes
            nz = out.dimensions[2]
            arr = np.tile(arr, nz)
            out.point_data[key] = arr


        sampled_list.append((name, out))

    # ── Concatenate into a single StructuredGrid and save ─────────────────────
    print("\nConcatenating blocks along i-axis...")
    merged = concatenate_blocks(sampled_list, z_ref=0.0)
    print(f"Merged grid: type={type(merged).__name__}  "
          f"pts={merged.n_points}  cells={merged.n_cells}  "    
            f"cell_data={list(merged.cell_data.keys())}  "
            f"point_data={list(merged.point_data.keys())}")
    

    return merged, num_wake_cells

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

    # Use cell-centre coordinates: centres are half a cell away from the wall
    # so k=0 centres are never on a panel and near-singular behaviour is avoided.
    X_c, Y_c = cell_centers(X, Y)   # shape (n_eta-1, n_xi-1)

    U_out, V_out, *_ = panel_foil.compute_velocity_field(X_c, Y_c)
    U  = U_out
    V  = V_out
    Cp = 1.0 - (U_out**2 + V_out**2) / U_inf**2

    return X_c, Y_c, U, V, Cp, cl_pot


def plot_physical_space(cmesh,path, name, show_plots=False  ):
    
    X = cmesh.points[:, 0]
    Y = cmesh.points[:, 1]

    #n_eta, n_xi = cmesh.dimensions

    n_xi, n_eta, nz = cmesh.dimensions
    print(f"Grid dimensions: n_xi={n_xi}, n_eta={n_eta}, nz={nz}")

    # Build a flat 3-D point array (z = 0)
    Z = np.zeros_like(X)
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])


    pl1 = pv.Plotter(shape=(3, 3), off_screen=not show_plots)

    data_plots = [
        (0, 0, "Cp", "Cp", (-1.0, 1.0)),
        (0, 1, "U_x", "U_x", (-1.0, 1.0)),
        (0, 2, "U_y", "U_y", (-0.2, 0.2)),

        (1, 0, "Cp_pot", "Cp_pot", (-1.0, 1.0)),
        (1, 1, "U_x_pot", "U_x_pot", (-1.0, 1.0)),
        (1, 2, "U_y_pot", "U_y_pot", (-0.2, 0.2)),

        (2, 0, "Cp_delta", "Cp_delta", (-0.2, 0.2)),
        (2, 1, "U_x_delta", "U_x_delta", (-0.1, 0.1)),
        (2, 2, "U_y_delta", "U_y_delta", (-0.1, 0.1))
    ]

    for r, c, field, label, clims in data_plots:
        pl1.subplot(r, c)
        pl1.add_mesh(cmesh.copy(), scalars=field, cmap="RdBu_r", clim=clims, show_edges=False, edge_color='lightgray', line_width=0.4)
        pl1.add_text(label)
        pl1.view_xy()

    pl1.show(auto_close=False)
    if not os.path.exists(f"{path}/physical/"):
        os.makedirs(f"{path}/physical/")
    pl1.screenshot(f"{path}/physical/{name}_physical_space.png", window_size=[1024, 768]*2)
    pl1.close()  

    U_rans = np.zeros((cmesh.n_points, 3))
    U_rans[:, 0] = cmesh.point_data["U_x"] # U_x
    U_rans[:, 1] = cmesh.point_data["U_y"] # U_y
    U_rans_norms = np.linalg.norm(U_rans, axis=1, keepdims=True)
    U_pot = np.zeros((cmesh.n_points, 3))
    U_pot[:, 0] = cmesh.point_data["U_x_pot"] # U_x
    U_pot[:, 1] = cmesh.point_data["U_y_pot"] # U_y
    U_pot_norms = np.linalg.norm(U_pot, axis=1, keepdims=True)
    U_delta = np.zeros((cmesh.n_points, 3))
    U_delta[:, 0] = cmesh.point_data["U_x_delta"] # U_x
    U_delta[:, 1] = cmesh.point_data["U_y_delta"] # U_y
    U_delta_norms = np.linalg.norm(U_delta, axis=1, keepdims=True)
    # add vector fields to point data for quiver plotting; zero z-component since this is 2D
    cmesh.point_data["U_rans"] = U_rans /( U_rans_norms + 1e-8) # normalise for better visualization; add small epsilon to avoid division by zero
    cmesh.point_data["U_pot"] = U_pot /( U_pot_norms + 1e-8)
    cmesh.point_data["U_delta"] = U_delta /( U_delta_norms + 1e-8)

    vector_plots = [
        (0, 0, "U_rans", "U_y_rans", "U_rans_field", (-0.2, 0.2)),
        (0, 1, "U_pot", "U_y_pot", "U_pot_field", (-0.2, 0.2)),
        (0, 2, "U_delta", "U_y_delta", "U_delta_field", (-0.1, 0.1))
    ]
    pl0 = pv.Plotter(shape=(1,3),off_screen=not show_plots)  
    # subsample the mesh for clearer vector plots; too dense and arrows overlap, too sparse and we miss important flow features
    subsampled_mesh = cmesh.extract_subset(
        voi=(0, n_xi-1, 0, n_eta-1, 0, 0), 
        rate=(15, 5, 1) # Adjust these to change arrow density
    )
    # Create specific scalar fields for coloring
    subsampled_mesh.point_data["U_y_rans"] = subsampled_mesh.point_data["U_rans"][:, 1]
    subsampled_mesh.point_data["U_y_pot"] = subsampled_mesh.point_data["U_pot"][:, 1]
    subsampled_mesh.point_data["U_y_delta"] = subsampled_mesh.point_data["U_delta"][:, 1]

    for r, c, orient_field, scalar_field, label, clims in vector_plots:
        arrows = subsampled_mesh.glyph(orient=orient_field, scale=False, factor=0.04)

        pl0.subplot(r, c)
        pl0.add_mesh(cmesh.copy(), scalars = None, style='wireframe', show_edges=True, edge_color='lightgray', line_width=0.4, opacity = 0.3)
        pl0.add_mesh(arrows.copy(), scalars = scalar_field, cmap="RdBu_r", clim = clims, show_edges=False, edge_color='lightgray', line_width=0.4)
        pl0.add_text(label)
        pl0.view_xy()
    pl0.show(auto_close=False)
    if not os.path.exists(f"{path}/physical/"):
        os.makedirs(f"{path}/physical/")
    pl0.screenshot(f"{path}/physical/{name}_vectors_physical_space.png", window_size=[1024, 768]*2)
    pl0.close()  
    
    # Second Plotter for Jacobian Coefficients
    pl2 = pv.Plotter(shape=(2, 3), off_screen=not show_plots)

    plots = [
        (0, 0, "x_xi", "x_xi"),
        (0, 1, "x_eta", "x_eta"),
        (1, 0, "y_xi", "y_xi"),
        (1, 1, "y_eta", "y_eta"),
        (1, 2, "det_J", "det_J")
    ]

    for r, c, field, label in plots:
        pl2.subplot(r, c)
        pl2.add_mesh(cmesh.copy(), scalars=field, cmap="RdBu_r", show_edges=False, edge_color='lightgray', line_width=0.4)
        pl2.add_text(label)
        pl2.view_xy()

    pl2.show(auto_close=False)
    if not os.path.exists(f"{path}/physical/"):
        os.makedirs(f"{path}/physical/")
    pl2.screenshot(f"{path}/physical/{name}_meshcoeffs.png", window_size=[1024, 768]*2)

    pl2.close()  

    pl3 = pv.Plotter(shape=(2, 2), off_screen=not show_plots)

    plots = [
        (0, 0, "nut", "nut"),
        (1, 0, "log_nut_ratio", "log10(nut/nu)"),
        (0, 1, "sdf", "sdf"),
        (1, 1, "exp_sdf", "exp(-k*sdf)")
    ]  
    for r, c, field, label in plots:
        pl3.subplot(r, c)
        pl3.add_mesh(cmesh.copy(), scalars=field, cmap="RdBu_r", show_edges=False,  line_width=0.4)
        pl3.add_text(label)
        pl3.view_xy()

    pl3.show(auto_close=False)
    if not os.path.exists(f"{path}/physical/"):
        os.makedirs(f"{path}/physical/")
    pl3.screenshot(f"{path}/physical/{name}_sdf_nut.png", window_size=[1024, 768]*2)
    pl3.close()      



def plot_latent_space(cmesh,path, name, show_plots=False):

    n_xi, n_eta, nz = cmesh.dimensions
    print(f"Grid dimensions: n_xi={n_xi}, n_eta={n_eta}, nz={nz}")
    xi_coords = np.arange(n_xi)
    eta_coords = np.arange(n_eta)
    XI, ETA, Z = np.meshgrid(xi_coords, eta_coords, [0], indexing='ij')

    latent_grid = pv.StructuredGrid( XI, ETA, Z)

    pl1 = pv.Plotter(shape=(3, 3), off_screen=not show_plots)
    
    # changed from cell to point data for better visualization and to match the input features which are point-based
    for field_name in cmesh.point_data.keys():
        print(f"Transfer field data to latent space for {field_name}...{cmesh.point_data[field_name].shape}")
        field = cmesh.point_data[field_name]
        if field.ndim > 1 and field.shape[1] == 3:
            latent_grid.point_data[field_name] = field # Do not ravel vectors
        else:
            latent_grid.point_data[field_name] = field.ravel(order='F')

    data_plots = [
        (0, 0, "Cp", "Cp", (-1.0, 1.0)),
        (0, 1, "U_x", "U_x", (-1.0, 1.0)),
        (0, 2, "U_y", "U_y", (-0.2, 0.2)),

        (1, 0, "Cp_pot", "Cp_pot", (-1.0, 1.0)),
        (1, 1, "U_x_pot", "U_x_pot", (-1.0, 1.0)),
        (1, 2, "U_y_pot", "U_y_pot", (-0.2, 0.2)),

        (2, 0, "Cp_delta", "Cp_delta", (-0.2, 0.2)),
        (2, 1, "U_x_delta", "U_x_delta", (-0.1, 0.1)),
        (2, 2, "U_y_delta", "U_y_delta", (-0.1, 0.1))
    ]

    for r, c, field_name, label, clims in data_plots:
        pl1.subplot(r, c)
        pl1.add_mesh(latent_grid.copy(), scalars=field_name, cmap="RdBu_r", clim=clims, show_edges=False, edge_color='lightgray', line_width=0.4)
        pl1.add_text(label)
        pl1.view_xy()

    pl1.show(auto_close=False)
    if not os.path.exists(f"{path}/latent/"):
        os.makedirs(f"{path}/latent/")
    pl1.screenshot(f"{path}/latent/{name}_latent_space.png", window_size=[1024, 768]*2)
    pl1.close()

    
    # Second Plotter for Jacobian Coefficients
    pl2 = pv.Plotter(shape=(2, 3), off_screen=not show_plots)

    plots = [
        (0, 0, "x_xi", "x_xi"),
        (0, 1, "x_eta", "x_eta"),
        (1, 0, "y_xi", "y_xi"),
        (1, 1, "y_eta", "y_eta"),
        (1, 2, "det_J", "det_J")
    ]

    for r, c, field, label in plots:
        pl2.subplot(r, c)
        pl2.add_mesh(latent_grid.copy(), scalars=field, cmap="RdBu_r", show_edges=False,  line_width=0.4)
        pl2.add_text(label)
        pl2.view_xy()

    pl2.show(auto_close=False)
    if not os.path.exists(f"{path}/latent/"):
        os.makedirs(f"{path}/latent/")
    pl2.screenshot(f"{path}/latent/{name}_meshcoeffs.png", window_size=[1024, 768]*2)
    pl2.close()    

    # 3rd plotter for SDF and nut

    pl3 = pv.Plotter(shape=(2, 2), off_screen=not show_plots)

    plots = [
        (0, 0, "nut", "nut"),
        (1, 0, "log_nut_ratio", "log10(nut/nu)"),
        (0, 1, "sdf", "sdf"),
        (1, 1, "exp_sdf", "exp(-k*sdf)")
    ]  
    for r, c, field, label in plots:
        pl3.subplot(r, c)
        pl3.add_mesh(latent_grid.copy(), scalars=field, cmap="RdBu_r", show_edges=False,  line_width=0.4)
        pl3.add_text(label)
        pl3.view_xy()
    
    pl3.show(auto_close=False)
    if not os.path.exists(f"{path}/latent/"):
        os.makedirs(f"{path}/latent/")
    pl3.screenshot(f"{path}/latent/{name}_sdf_nut.png", window_size=[1024, 768]*2)
    pl3.close()  

def save_to_pytorch( path, name, archive_dict, MID_FIX="C_mesh"):

    if not os.path.exists(path):
        os.makedirs(path)
    print(f"Preparing to save tensors for {name} with shape {archive_dict['x'].shape} and properties {list(archive_dict.keys())}")

    nc, nx, ny = archive_dict['x'].shape

    print(f"Saving tensors to PyTorch file: {name}_{MID_FIX}_{nx}x{ny}.pt")

    # Save the tensors to a file
    save_path = os.path.join(path, f"{name}_{MID_FIX}_{nx}x{ny}.pt")
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

def process_airfrans_calc_potentialflow(name, AF_ROOT, OF_ROOT, STORAGE_DIR, I0_min, J_MAX, output_type='vts', show_plots=False):
        print(f"\n=== Processing Airfrans simulation: {name} ===")
        print(f"AF_ROOT: {AF_ROOT}  OF_ROOT: {OF_ROOT}  STORAGE_DIR: {STORAGE_DIR}  I0_min: {I0_min}  J_MAX: {J_MAX}  output_type: {output_type}")
        # check if archive exists

        archive_out = f"{STORAGE_DIR}/{name}_C_mesh_512x64.pt"
        if os.path.exists(archive_out):
            print(f" Skipping {name}   Found existing file {archive_out}")
            return


        simulation = af.Simulation(root=AF_ROOT, name=name)

        print(f"\nProcessing simulation: {name}")
        SIM_PATH = f"{OF_ROOT}/{name}"
        VTM_PATH = f"{SIM_PATH}/constant/blockMeshVTK/blockMesh.vtm"
        FOAM_PATH = f"{SIM_PATH}/touch.foam"

        #I0_min = 56
        #J_MAX = 158 # 158 
        cmesh, num_wake_cells = get_cmesh(SIM_PATH, VTM_PATH, FOAM_PATH, I0_min, J_MAX, output_type='vts')
        print(f"Extracted C-mesh: type={type(cmesh).__name__}  pts={cmesh.n_points}  cells={cmesh.n_cells}  "
                f"cell_data={list(cmesh.cell_data.keys())}  point_data={list(cmesh.point_data.keys())}")
        nx, ny, nz = cmesh.dimensions
        X = cmesh.points[:, 0].reshape((nx, ny, nz), order='F').squeeze()
        Y = cmesh.points[:, 1].reshape((nx, ny, nz), order='F').squeeze()

        print(f"number of cells in wake region (block_0): {num_wake_cells}")
        # Wake cells for 
        I_START = num_wake_cells
        I_END = nx-I_START+1

        x_wall = X[I_START:I_END, 0]  # i varies along the wall
        y_wall = Y[I_START:I_END, 0]
        print(f"Extracted wall points: {len(x_wall)}")
        # Airfrans wall points

        af_points = simulation.airfoil.points # (N,3)


        x_wall_af = af_points[:, 0]
        y_wall_af = af_points[:, 1]
        print(f"Extracted wall points from Airfrans: {len(x_wall_af)}")

        if len(x_wall_af) != len(x_wall):
            print(f" ⚠️ Warning: Number of wall points from Airfrans ({len(x_wall_af)}) does not match OpenFOAM ({len(x_wall)}).")
        x_wall_af_max = np.max(x_wall_af)
        x_wall_af_min = np.min(x_wall_af)
        print(f"Airfrans wall x range: [{x_wall_af_min:.3f}, {x_wall_af_max:.3f}]")
        x_wall_max = np.max(x_wall)
        x_wall_min = np.min(x_wall)
        print(f"OpenFOAM wall x range: [{x_wall_min:.3f}, {x_wall_max:.3f}]")
        
        # Need to calculate the SDF /implicit distance from the wall to the first cell centres 
        # add z-axis for PyVista PolyData

        # Compute unsigned distance from each cell centre to the nearest wall
        # node using a KD-tree.  vtkImplicitPolyDataDistance (used by
        # compute_implicit_distance) requires triangulated faces and fails on
        # a bare point-cloud / polyline, so we use scipy instead.
        from scipy.spatial import KDTree
        wall_xy  = np.column_stack((x_wall, y_wall))          # (N_wall, 2)
        tree     = KDTree(wall_xy)
        centers  = cmesh.cell_centers()                        # PolyData
        cc_xy    = centers.points[:, :2]                       # (N_cells, 2)
        sdf_dist, _ = tree.query(cc_xy, workers=-1)            # unsigned dist

        sdf_fixed = np.nan_to_num(sdf_dist, nan=0.0).astype(np.float32)

        print(f"Shape of SDF array: {sdf_fixed.shape}  min: {sdf_fixed.min():.6f}  max: {sdf_fixed.max():.6f}  mean: {sdf_fixed.mean():.6f}")
        print(f" shape of cell data: {cmesh.cell_data['U'].shape}  {cmesh.cell_data['p'].shape}  {cmesh.cell_data['nut'].shape} ")
        cmesh.cell_data['sdf'] = sdf_fixed          
        # decay factor
        k=5.0  # 2 not fast enough
        cmesh.cell_data['exp_sdf'] = np.exp(-k*sdf_fixed)  # add exp(-sdf) as an additional feature to capture near-wall effects with sharper gradients

        tree_p     = KDTree(wall_xy)
        points  = cmesh.points[:,:2]                      # PNode data 
        sdf_point_dist, _ = tree.query(points, workers=-1)            # unsigned dist  
        sdf_point_fixed = np.nan_to_num(sdf_point_dist, nan=0.0).astype(np.float32)
        cmesh.point_data['sdf'] = sdf_point_fixed
        cmesh.point_data['exp_sdf'] = np.exp(-k*sdf_point_fixed)

        print (f"Running panel method with {len(x_wall)} wall points...")
        AOA = simulation.angle_of_attack * 180 / np.pi
        U_inf = simulation.inlet_velocity
        nu = simulation.NU
        rho = simulation.RHO
        reynolds = rho * U_inf / nu
        log_re = np.log(reynolds)   

        print(f"Angle of attack: {AOA:.2f} degrees")
        print(f"Freestream velocity: {U_inf:.2f} m/s")

        print(f" Shape of X: {X.shape}  Y: {Y.shape} ")

        PATH_POTENTIAL = SIM_PATH + "/potential"
        os.makedirs(PATH_POTENTIAL, exist_ok=True)  

        X_c, Y_c, u, v, Cp, cl_pot = run_panel_method(x_wall,y_wall, X, Y,  alpha_aoa=AOA, U_inf=1.0, path=PATH_POTENTIAL)

        # Lift from panel method vs Airfrans forces for verification



        print(f"From OpenFoam Shape of X: {X.shape}  Y: {Y.shape}  ")
        print(f"Cell centers Shape of X_c: {X_c.shape}  Y_c: {Y_c.shape}  u: {u.shape}  v: {v.shape}  Cp: {Cp.shape} ")
        
        # put potential flow solution onto the Openfoam C-mesh 
        cmesh.cell_data['Cp_pot'] = Cp.ravel(order='F')
        cmesh.cell_data['U_x_pot'] = u.ravel(order='F') 
        cmesh.cell_data['U_y_pot'] = v.ravel(order='F') 

        # convert openfoam data 
        cmesh.cell_data['U_x'] = cmesh.cell_data['U'][:, 0] /U_inf
        cmesh.cell_data['U_y'] = cmesh.cell_data['U'][:, 1] /U_inf
        cmesh.cell_data.pop('U')  # remove original vector field to avoid  

        #rho = simulation.RHO

        # p is kinematic pressure (p/rho), so we can directly compute Cp without needing to know the density

        cmesh.cell_data['Cp'] = cmesh.cell_data['p'] / (0.5 *  U_inf**2)  # rename pressure to Cp for consistency

        cmesh.cell_data.pop('p')  # remove original pressure field to avoid confusion
        cmesh.cell_data['Cp_delta'] = cmesh.cell_data['Cp'] - cmesh.cell_data['Cp_pot']
        cmesh.cell_data['U_x_delta'] = cmesh.cell_data['U_x'] - cmesh.cell_data['U_x_pot']
        cmesh.cell_data['U_y_delta'] = cmesh.cell_data['U_y'] - cmesh.cell_data['U_y_pot'] 
        cmesh.cell_data['log_nut_ratio'] =  np.log10(np.clip(cmesh.cell_data['nut'], 1e-12, None) / nu)  # take log(nut/nu) to compress range for ML training
        
        # interpolate cell Cp to points for better visualization and to match input features which are point-based
        nodal_mesh = cmesh.cell_data_to_point_data()

        cmesh.point_data['Cp'] = nodal_mesh['Cp']  
        cmesh.point_data['sdf'] = nodal_mesh['sdf']
        cmesh.point_data['exp_sdf'] = nodal_mesh['exp_sdf']
        cmesh.point_data['x_xi'] = nodal_mesh['x_xi']
        cmesh.point_data['x_eta'] = nodal_mesh['x_eta']
        cmesh.point_data['y_xi'] = nodal_mesh['y_xi']
        cmesh.point_data['y_eta'] = nodal_mesh['y_eta']
        cmesh.point_data['det_J'] = nodal_mesh['det_J']
        cmesh.point_data['U_x_pot'] = nodal_mesh['U_x_pot']       
        cmesh.point_data['U_y_pot'] = nodal_mesh['U_y_pot']       
        cmesh.point_data['Cp_pot'] = nodal_mesh['Cp_pot']    
        cmesh.point_data['U_x'] = nodal_mesh['U_x']
        cmesh.point_data['U_y'] = nodal_mesh['U_y']
        cmesh.point_data['Cp_delta'] = nodal_mesh['Cp_delta']
        cmesh.point_data['U_x_delta'] = nodal_mesh['U_x_delta']
        cmesh.point_data['U_y_delta'] = nodal_mesh['U_y_delta']
        cmesh.point_data['nut'] = nodal_mesh['nut']
        cmesh.point_data['log_nut_ratio'] = nodal_mesh['log_nut_ratio']  
        cmesh.point_data['wallShearStress_x'] = nodal_mesh['wallShearStress'][:, 0]
        cmesh.point_data['wallShearStress_y'] = nodal_mesh['wallShearStress'][:, 1]

        n_xi, n_eta, nz = cmesh.dimensions


        plot_physical_space(cmesh, SIM_PATH, name, show_plots=False)
        plot_latent_space(cmesh, SIM_PATH, name, show_plots=False)

        # Assemble tensors and save to PyTorch format for ML training

        # Calculate aerodynamic coefficients for verification
        ((cd, cdp, cdv), (cl, clp, clv)) = simulation.force_coefficient(compressible=False,reference=False)

        print(f" Airfrans Cl = {cl:.5f}, Cl_pot = {cl_pot:.5f}, Cl_delta = {cl - cl_pot:.5f}")

        #print(f"  Cd: {cd:.5f}, Cdp: {cdp:.5f}, Cdv: {cdv:.5f}, Cl: {cl:.5f}, Clp: {clp:.5f}, Clv: {clv:.5f}")
        # stack the input features and output fields into tensors
        x_data = np.column_stack((cmesh.points[:, 0].ravel(order='F'),
                                    cmesh.points[:, 1].ravel(order='F'),
                                    cmesh.point_data['U_x_pot'],
                                    cmesh.point_data['U_y_pot'],
                                    cmesh.point_data['Cp_pot'],
                                    cmesh.point_data['sdf'],
                                    cmesh.point_data['exp_sdf'],
                                    cmesh.point_data['x_xi'],
                                    cmesh.point_data['x_eta'],
                                    cmesh.point_data['y_xi'],
                                    cmesh.point_data['y_eta'],
                                    cmesh.point_data['det_J']
                                    ))  # shape (n_cells, 2)
        
        print(f"Shape of input feature array x_data: {x_data.shape}  sample: {x_data[0]}")
        
        y_delta = np.column_stack((cmesh.point_data['Cp_delta'],
                                    cmesh.point_data['U_x_delta'],
                                    cmesh.point_data['U_y_delta'],
                                    cmesh.point_data['log_nut_ratio']))
        print(f"Shape of output delta array y_delta: {y_delta.shape}  sample: {y_delta[0]}")
        y_out  = np.column_stack((
                                    cmesh.point_data['U_x'],
                                    cmesh.point_data['U_y'],
                                    cmesh.point_data['Cp'],
                                    cmesh.point_data['nut'],
                                    cmesh.point_data['wallShearStress_x'],  # wall shear stress x-component
                                    cmesh.point_data['wallShearStress_y']  # wall shear stress y-component
                                    ))
        print(f"Shape of output array y_out: {y_out.shape}  sample: {y_out[0]}")

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
            }

        # 7. Package everything

        x_tensor = torch.tensor(x_data, dtype=torch.float32)
        y_delta_tensor = torch.tensor(y_delta, dtype=torch.float32)
        y_out_tensor = torch.tensor(y_out, dtype=torch.float32)

        # reshape x_data to (Channels, 1182, 159) for PyTorch convention (C, H, W)
        x_data_spatial = x_tensor.view(n_eta, n_xi, 12).permute(2, 1, 0)
        y_delta_spatial = y_delta_tensor.view(n_eta, n_xi, 4).permute(2, 1, 0)
        y_out_spatial = y_out_tensor.view(n_eta, n_xi, 6).permute(2, 1, 0)
        
        cp_stagnation = y_out_spatial[2, n_xi//2, 0].item()

        # Raw array check
        cp_raw = cmesh.point_data['Cp']
        max_cp_raw = np.max(cp_raw)
        idx_raw = np.argmax(cp_raw)

        print(f"--- RAW DATA CHECK ---")
        print(f"Global Max Cp: {max_cp_raw:.4f}")
        print(f"Flat index of Max Cp: {idx_raw}")

        # Check coordinates of the max value in PyVista
        # This works regardless of how the array is flattened
        max_pt = cmesh.points[idx_raw]
        print(f"Physical (x, y) of Max Cp: ({max_pt[0]:.4f}, {max_pt[1]:.4f})")

        print(f"Cp at leading edge stagnation point: {cp_stagnation:.4f}")

        if np.isclose(cp_stagnation, 1.0, atol=1e-2):
            print("✅ Logic Check Passed: Cp is ~1.0 at the wall.")
        else:
            print(f"⚠️ Logic Check Failed: Cp is {cp_stagnation}. Check channel index or grid orientation.")

        cp_field = y_out_spatial[2, :, :]
        flat_idx = torch.argmax(cp_field)
        ni, nj = cp_field.shape
        i_max = flat_idx // nj
        j_max = flat_idx % nj

        # Check Cp along the wake cut at a few radial distances
        for i_layer in [0, 20, 50]:
            val_start = y_out_spatial[2, i_layer, 0].item()
            val_end = y_out_spatial[2, -1 - i_layer, 0].item()
            diff = abs(val_start - val_end)
            
            status = "✅ Continuous" if diff < 1e-3 else "⚠️ Discontinuous"
            print(f"Wake Check at i={i_layer}: Start={val_start:.4f}, End={val_end:.4f} | {status}")

        # put tensors and properties into a dictionary for saving
        archive_dict = {
            'x': x_data_spatial,
            'y_delta': y_delta_spatial,
            'y_out': y_out_spatial,
            'props': props
        }
        print(f"Max Cp of {cp_field[i_max, j_max]:.4f} found at i={i_max}, j={j_max}")
        print(f" Save Complete Archive")
        save_to_pytorch( STORAGE_DIR, name, archive_dict)

        # Resample to different grid resolutions for data augmentation and to test interpolation methods

        # Assuming 'x_master' is your (Channels, 1182, 159) nodal tensor
        resolutions = [( 1024, 128), (512, 64), (256, 32) ]  # (ni, nj) pairs for resampling

        for ni_new, nj_new in resolutions:
            # 1. Interpolate (ensure 4D input: [Batch, Channel, H, W])
            # Mode 'bilinear' with align_corners=True preserves the wall (j=0) perfectly

            x_batch = x_data_spatial.unsqueeze(0)
            x_resampled = F.interpolate(
                x_batch, 
                size=(ni_new, nj_new), 
                mode='bilinear', 
                align_corners=True
            ).squeeze(0)
            y_delta_batch = y_delta_spatial.unsqueeze(0)
            y_delta_resampled = F.interpolate(
                y_delta_batch, 
                size=(ni_new, nj_new), 
                mode='bilinear', 
                align_corners=True
            ).squeeze(0)

            y_out_batch = y_out_spatial.unsqueeze(0)
            y_out_resampled = F.interpolate(
                y_out_batch, 
                size=(ni_new, nj_new), 
                mode='bilinear', 
                align_corners=True
            ).squeeze(0)
            
            train_dict ={
                'x': x_resampled,
                'y_delta': y_delta_resampled,
                'y_out': y_out_resampled,
                'props': props
            }

            # ---------------------------------------------------------
            # 2. RECALCULATE JACOBIAN METRICS FOR THE NEW RESOLUTION
            # Extract the newly interpolated physical coordinates (Channels 0 and 1)
            # .cpu().numpy() ensures this works even if tensors are on GPU
            X_new = x_resampled[0].cpu().numpy()
            Y_new = x_resampled[1].cpu().numpy()
            
            # Compute fresh metrics matching the new computational spacing
            x_xi, x_eta, y_xi, y_eta, det_J = compute_nodal_metrics(X_new, Y_new)
            
            # Overwrite the interpolated metrics (Channels 7 through 11)
            # Keep them on the same device as the original tensor
            device = x_resampled.device
            x_resampled[7]  = torch.tensor(x_xi, dtype=torch.float32, device=device)
            x_resampled[8]  = torch.tensor(x_eta, dtype=torch.float32, device=device)
            x_resampled[9]  = torch.tensor(y_xi, dtype=torch.float32, device=device)
            x_resampled[10] = torch.tensor(y_eta, dtype=torch.float32, device=device)
            x_resampled[11] = torch.tensor(det_J, dtype=torch.float32, device=device)

            print(f" Saving at resolution {ni_new} {nj_new}")
            save_to_pytorch( STORAGE_DIR, name, train_dict) 

def load_from_pytorch(STORAGE_DIR, filename):


    print(f"Loading  tensors from PyTorch file: {filename}")

    path = os.path.join(STORAGE_DIR, filename)

    # Load the tensors from a file
    archive_dict = torch.load( path)

    return archive_dict

def resample_airfrans(STORAGE_DIR, resolutions=[(256,64), (512,128), (1024,128)], MID_FIX="C_mesh_V2"):

        # load existing full archive  all full size archives end in 9
        sim_files = [f.name for f in Path(STORAGE_DIR).iterdir() if f.name.endswith("9.pt")]

        for file in sim_files:
            print(f"Found existing archive for : {file}")

            name = str.split(file,'_C')[0]

            archive_dict = load_from_pytorch(STORAGE_DIR, file)

            x_data_spatial = archive_dict['x']
            y_delta_spatial = archive_dict['y_delta']
            y_out_spatial = archive_dict['y_out']
            props = archive_dict['props']
            # Modify some features
            # SDF channel = 5  
            # exp(SDF) = 6
            # Add (nut/nu  + eps)^(1/3) 
            nu = props['nu_mol'].item()
            eps_nu = 0.01
            nut_ratio = y_out_spatial[3:4, :, :] / nu + eps_nu
            nutratio_cuberoot = torch.pow(nut_ratio, 1/3)      
            # 2. Concatenate to your existing 4-channel y_delta
            # y_delta: [ 4, X, Y] -> y_delta_extended: [5, X, Y]
            y_delta_extended = torch.cat([y_delta_spatial, nutratio_cuberoot], dim=0)

            for ni_new, nj_new in resolutions:
                # 1. Interpolate (ensure 4D input: [Batch, Channel, H, W])
                # Mode 'bilinear' with align_corners=True preserves the wall (j=0) perfectly

                x_batch = x_data_spatial.unsqueeze(0)
                x_resampled = F.interpolate(
                    x_batch, 
                    size=(ni_new, nj_new), 
                    mode='bilinear', 
                    align_corners=True
                ).squeeze(0)
                y_delta_batch = y_delta_extended.unsqueeze(0)
                y_delta_resampled = F.interpolate(
                    y_delta_batch, 
                    size=(ni_new, nj_new), 
                    mode='bilinear', 
                    align_corners=True
                ).squeeze(0)

                y_out_batch = y_out_spatial.unsqueeze(0)
                y_out_resampled = F.interpolate(
                    y_out_batch, 
                    size=(ni_new, nj_new), 
                    mode='bilinear', 
                    align_corners=True
                ).squeeze(0)
                
                train_dict ={
                    'x': x_resampled,
                    'y_delta': y_delta_resampled,
                    'y_out': y_out_resampled,
                    'props': props
                }

                # ---------------------------------------------------------
                # 2. RECALCULATE JACOBIAN METRICS FOR THE NEW RESOLUTION
                # Extract the newly interpolated physical coordinates (Channels 0 and 1)
                # .cpu().numpy() ensures this works even if tensors are on GPU
                X_new = x_resampled[0].cpu().numpy()
                Y_new = x_resampled[1].cpu().numpy()
                
                # Compute fresh metrics matching the new computational spacing
                x_xi, x_eta, y_xi, y_eta, det_J = compute_nodal_metrics(X_new, Y_new)
                
                # Overwrite the interpolated metrics (Channels 7 through 11)
                # Keep them on the same device as the original tensor
                device = x_resampled.device
                x_resampled[7]  = torch.tensor(x_xi, dtype=torch.float32, device=device)
                x_resampled[8]  = torch.tensor(x_eta, dtype=torch.float32, device=device)
                x_resampled[9]  = torch.tensor(y_xi, dtype=torch.float32, device=device)
                x_resampled[10] = torch.tensor(y_eta, dtype=torch.float32, device=device)
                x_resampled[11] = torch.tensor(det_J, dtype=torch.float32, device=device)

                print(f" Saving at resolution {ni_new} {nj_new}")
                new_Storage = os.path.join(STORAGE_DIR , MID_FIX)
                os.makedirs(new_Storage, exist_ok=True)
                save_to_pytorch( new_Storage, name, train_dict, MID_FIX) 


def resample_worker_wrapper(arg_tuple):
    name, AF_ROOT, OF_ROOT, STORAGE_DIR, I0_min, J_MAX = arg_tuple
    failed_log = os.path.join(STORAGE_DIR, "failed_simulations.txt")
    try:
        return resample_airfrans(
            STORAGE_DIR=STORAGE_DIR,
            resolutions=[(256,64), (512,128), (1024,128)],
            MID_FIX = "C_mesh_V2",
        )
    except Exception as e:
        import traceback
        error_msg = f"{name}: {type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(f"\n❌ ERROR processing '{name}':\n{error_msg}", flush=True)
        os.makedirs(STORAGE_DIR, exist_ok=True)
        with open(failed_log, "a") as f:
            f.write(f"{name}\t{type(e).__name__}: {e}\n")
        return None

# 1. Move the wrapper OUTSIDE any other function
def global_worker_wrapper(arg_tuple):
    """
    Takes a single tuple of arguments because executor.map 
    is happiest with a single iterable.
    """
    name, AF_ROOT, OF_ROOT, STORAGE_DIR, I0_min, J_MAX = arg_tuple
    failed_log = os.path.join(STORAGE_DIR, "failed_simulations.txt")
    try:
        return process_airfrans_calc_potentialflow(
            STORAGE_DIR=STORAGE_DIR,
            I0_min=I0_min,
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
    OF_ROOT = "/home/timm/Projects/PIML/OF_dataset"      
    STORAGE_DIR ="/home/timm/storage/AF_NO_DATASET/Archive"
    #name = "airFoil2D_SST_31.68_0.424_0.273_4.301_1.0_11.616"
    names = ["airFoil2D_SST_32.494_3.294_4.038_1.959_8.677"]

    sim_dirs = [d.name for d in Path(OF_ROOT).iterdir() if d.is_dir()]
    #sim_dirs = sim_dirs[:15]
    sim_dirs = ["airFoil2D_SST_67.783_-2.041_4.431_3.865_18.25"]

    # clip first 56 cells in i-direction from wake regions for closer match with airfrans training data
    I0_min = 56
    # clip mesh in radial direction to take only the first 158 cells in j-direction to match airfrans training data and avoid large uniform regions far from the wall which can cause ML models to struggle with learning the important near-wall gradients
    J_MAX = 158 # 158 

    # 2. Prepare the list of argument tuples for the map
    # This bundles everything the worker needs into one package per simulation
    job_args = [
        (name, AF_ROOT, OF_ROOT,  STORAGE_DIR, I0_min, J_MAX) 
        for name in sim_dirs
    ]
    
    # If num_workers is None, it uses all available logical cores - 2.

    num_workers=6
    if num_workers is None:
        num_workers = max(1, os.cpu_count() - 2)
    
    print(f"🚀 Starting Multiprocessing with {num_workers} workers...")

    # We use ProcessPoolExecutor for CPU-bound tasks like PyVista sampling
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # uncomment to make new full archives
        #list(tqdm(executor.map(global_worker_wrapper, job_args), total=len(job_args), desc="Processing Airfoils"))
        #   For resampling
        list(tqdm(executor.map(resample_worker_wrapper, job_args), total=len(job_args), desc="Resampling Airfoils"))
        


                




