
from curses import window

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
    print("Extracting subgrids with extents:")
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
    CELL_FIELDS  = ["U", "p"]           # cell-centred → cell_data
    #POINT_FIELDS = ["p"]     # node-interpolated → point_data
    #KEEP = set(CELL_FIELDS + POINT_FIELDS)
    KEEP = set(CELL_FIELDS)
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

    sigmas = panel_foil.geom.source_strength
    gamma = panel_foil.geom.gamma

    return X_c, Y_c, U, V, Cp, cl_pot, sigmas, gamma


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
        arrows = subsampled_mesh.glyph(orient=orient_field, scale=False, factor=0.02)

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


    
def check_source_direction(P, N1, N2, sigma):
    # Convert inputs to numpy arrays
    P = np.array(P)
    N1 = np.array(N1)
    N2 = np.array(N2)
    
    # 1. Geometry setup
    L = np.linalg.norm(N2 - N1)
    phi = np.arctan2(N2[1] - N1[1], N2[0] - N1[0])
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    
    # 2. Translation
    dx, dy = P[0] - N1[0], P[1] - N1[1]
    
    # 3. FIXED Local Coordinate Transformation (Outward-Normal Basis)
    # x_loc: Projection onto the panel tangent
    # y_loc: Projection onto the outward normal (dx*sin - dy*cos)
    x_loc = dx * cos_phi + dy * sin_phi
    y_loc = dx * sin_phi - dy * cos_phi

    # Local V_source (Normal component)
    # If y_loc is negative (inside/near surface), this should push away
    beta_pt_0 = np.arctan2(y_loc, x_loc)
    beta_pt_L = np.arctan2(y_loc, x_loc - L)

    v_src = (sigma / (2 * np.pi)) * (beta_pt_0 - beta_pt_L)

    print("\n--- Source Direction Check ---")
    print(f" Beta at panel start (0): {np.rad2deg(beta_pt_0):.4f} deg, Beta at panel end (L): {np.rad2deg(beta_pt_L):.4f} deg")

    print(f" Check source direction at P={P} relative to panel N1={N1} N2={N2} with sigma={sigma:.4f}")

    
    # 4. Local Analytical Integral
    # v_src is the normal velocity. With sigma > 0 and y_loc > 0, 
    # this should be a positive value (pushing away from the surface).
    v_src_loc = (sigma / (2 * np.pi)) * (beta_pt_0 - beta_pt_L)
    u_src_loc = (sigma / (4 * np.pi)) * np.log((x_loc**2 + y_loc**2) / ((x_loc - L)**2 + y_loc**2))
    
    # 5. Back-Rotation to Global Coordinates
    # Matches: U = u_loc*cos - v_loc*sin | V = u_loc*sin + v_loc*cos
    # Note: Using the standard rotation matrix signs because v_src_loc 
    # is already defined in the "outward" frame.
    U_ind = u_src_loc * cos_phi - v_src_loc * sin_phi
    V_ind = u_src_loc * sin_phi + v_src_loc * cos_phi
    
    print(f"--- Diagnostic for Panel ({N1} to {N2}) ---")
    print(f"  Panel Angle: {np.rad2deg(phi):.2f}°")
    print(f"  Local Point: x={x_loc:.4f}, y={y_loc:.4f} ({'Outside' if y_loc > 0 else 'Inside'})")
    print(f"  Local Vel:   u_tang={u_src_loc:.4f}, v_norm={v_src_loc:.4f}")
    print(f"  Global Ind:  U={U_ind:.4f}, V={V_ind:.4f}")
    
    return U_ind, V_ind

if __name__ == "__main__":

    ROOT_DIR = "/home/timm/Projects/PIML/OF_dataset"

    STORAGE_DIR ="/home/timm/storage/AF_NO_DATASET/Archive"
    #name = "airFoil2D_SST_31.68_0.424_0.273_4.301_1.0_11.616"
    names = ["airFoil2D_SST_32.494_3.294_4.038_1.959_8.677"]
    #names = ["airFoil2D_SST_31.803_7.291_3.243_6.962_0.0_10.641", "airFoil2D_SST_32.058_11.514_6.565_6.266_9.329", "airFoil2D_SST_31.68_0.424_0.273_4.301_1.0_11.616"]
    #names = [d for d in os.listdir(ROOT_DIR) if d.startswith("airFoil2D_SST")]
    for d in names:
        if d.startswith("airFoil2D_SST"):
            AF_ROOT = "/home/timm/Projects/PIML/Dataset"

            simulation = af.Simulation(root=AF_ROOT, name=d)

            print(f"\nProcessing simulation: {d}")
            SIM_PATH = f"{ROOT_DIR}/{d}"
            VTM_PATH = f"{SIM_PATH}/constant/blockMeshVTK/blockMesh.vtm"
            FOAM_PATH = f"{SIM_PATH}/touch.foam"

            I0_min = 56
            J_MAX = 158 # 158 
            
            # load Cmesh from disk for testing
            cmesh = pv.read(f"{SIM_PATH}/testdata/{d}_cmesh_physical_space.vts")

            print(f"\nCombined OpenFOAM mesh: type={type(cmesh).__name__}  "
                    f"pts={cmesh.n_points}  cells={cmesh.n_cells}")
            print(f"Available arrays: {cmesh.array_names}")

            nx, ny, nz = cmesh.dimensions
            X = cmesh.points[:, 0].reshape((nx, ny, nz), order='F').squeeze()
            Y = cmesh.points[:, 1].reshape((nx, ny, nz), order='F').squeeze()


            num_wake_cells = 82  ##need to adjust with case name
            print(f"number of cells in wake region (block_0): {num_wake_cells} need to adjust with case name")
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

            print(f" Rangle of x_wall: [{x_wall.min():.3f}, {x_wall.max():.3f}]  y_wall: [{y_wall.min():.3f}, {y_wall.max():.3f}]")

            X_c, Y_c, u, v, Cp, cl_pot, sigmas , gamma = run_panel_method(x_wall,y_wall, X, Y,  alpha_aoa=AOA, U_inf=1.0, path=PATH_POTENTIAL)

            # Lift from panel method vs Airfrans forces for verification

            print(f"From OpenFoam Shape of X: {X.shape}  Y: {Y.shape}  ")
            print(f"Cell centers Shape of X_c: {X_c.shape}  Y_c: {Y_c.shape}  u: {u.shape}  v: {v.shape}  Cp: {Cp.shape} ")
            
            # put potential flow solution onto the Openfoam C-mesh 
            cmesh.cell_data['Cp_pot'] = Cp.ravel(order='F')
            cmesh.cell_data['U_x_pot'] = u.ravel(order='F') 
            cmesh.cell_data['U_y_pot'] = v.ravel(order='F') 

            #rho = simulation.RHO

            cmesh.cell_data['Cp_delta'] = cmesh.cell_data['Cp'] - cmesh.cell_data['Cp_pot']
            cmesh.cell_data['U_x_delta'] = cmesh.cell_data['U_x'] - cmesh.cell_data['U_x_pot']
            cmesh.cell_data['U_y_delta'] = cmesh.cell_data['U_y'] - cmesh.cell_data['U_y_pot'] 
            
            # interpolate cell Cp to points for better visualization and to match input features which are point-based
            nodal_mesh = cmesh.cell_data_to_point_data()

            cmesh.point_data['Cp'] = nodal_mesh['Cp']  
            cmesh.point_data['U_x_pot'] = nodal_mesh['U_x_pot']       
            cmesh.point_data['U_y_pot'] = nodal_mesh['U_y_pot']       
            cmesh.point_data['Cp_pot'] = nodal_mesh['Cp_pot']    
            cmesh.point_data['U_x'] = nodal_mesh['U_x']
            cmesh.point_data['U_y'] = nodal_mesh['U_y']
            cmesh.point_data['Cp_delta'] = nodal_mesh['Cp_delta']
            cmesh.point_data['U_x_delta'] = nodal_mesh['U_x_delta']
            cmesh.point_data['U_y_delta'] = nodal_mesh['U_y_delta']

            n_xi, n_eta, nz = cmesh.dimensions

            panel_N1 = 616
            panel_N2 = panel_N1 + 1
            print(f"Checking source direction at panel between N1={panel_N1} and N2={panel_N2}...")

            checkpoints  = [(-0.02,0.01), (-0.01,0.02),(-0.02,-0.02)]


            for cp in checkpoints:
                v_src, y_loc =  check_source_direction(cp, (x_wall[panel_N1], y_wall[panel_N1]), (x_wall[panel_N2], y_wall[panel_N2]), sigma=sigmas[panel_N1])
                print(f"Checkpoint {cp}: v_src={v_src:.4f}, y_loc={y_loc:.4f} sigma {sigmas[panel_N1]:.4f}  {'inside' if y_loc < 0 else 'outside'} the surface, expected v_src to push {'away from' if sigmas[panel_N1] > 0 else 'towards'} the surface")
            
            check_source_direction((0.5, 0.0), (x_wall[0], y_wall[0]), (x_wall[-1], y_wall[-1]), sigma=gamma)
            
            plot_physical_space(cmesh, SIM_PATH, d, show_plots=True)

            # Assemble tensors and save to PyTorch format for ML training

            # Calculate aerodynamic coefficients for verification
            ((cd, cdp, cdv), (cl, clp, clv)) = simulation.force_coefficient(compressible=False,reference=False)

            print(f" Airfrans Cl = {cl:.5f}, Cl_pot = {cl_pot:.5f}, Cl_delta = {cl - cl_pot:.5f}")

            # 1. Filter indices for the front half of the airfoil
            front_half_mask = x_wall[:-1] < 0.05
            front_indices = np.where(front_half_mask)[0]

            # 2. Find the largest source strength in that front region
            max_sigma_front_idx = front_indices[np.argmax(np.abs(sigmas[front_indices]))]
            max_sigma_front_val = sigmas[max_sigma_front_idx]

            print(f"Largest source (Front Half) at Panel {max_sigma_front_idx}")
            print(f"X-location: {x_wall[max_sigma_front_idx]:.4f}")
            print(f"Sigma value: {max_sigma_front_val:.6f}")

            # 3. Define the geometry for this panel 
            # C-Mesh ordering is CW and PanelFoil expects CCW, so N1 is the next point and N2 is the current point in the C-mesh ordering
            N1_coord = np.array([x_wall[max_sigma_front_idx +1], y_wall[max_sigma_front_idx + 1]])
            N2_coord = np.array([x_wall[max_sigma_front_idx ], y_wall[max_sigma_front_idx ]])

            # 4. Calculate unit outward normal (Assuming CCW ordering)
            dx, dy = N2_coord - N1_coord
            L = np.linalg.norm([dx, dy])
            nx, ny = -dy/L, dx/L 

            # 5. Create a Test Point 2% chord 'outside' the midpoint
            midpoint = 0.5 * (N1_coord + N2_coord)
            test_p = midpoint - 0.1 * np.array([nx, ny])

            # 6. Run the debug check
            v_src, y_loc = check_source_direction(test_p, N1_coord, N2_coord, max_sigma_front_val)





                




