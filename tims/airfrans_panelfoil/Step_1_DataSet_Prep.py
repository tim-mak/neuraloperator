
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
    subgrids_raw = []
    for bname in BLOCK_ORDER:
        grid = grid_by_name[bname]
        ni, nj, nk = grid.dimensions
        if bname == 'block_0':
            sub = grid.extract_subset([I0_min, ni - 1, 0, min(J_MAX, nj - 1), 0, 1])
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

        # Jacobian metric tensor
        jac = compute_jacobian_metrics(sub)
        for key, arr in jac.items():
            out.cell_data[key] = arr

        sampled_list.append((name, out))

    # ── Concatenate into a single StructuredGrid and save ─────────────────────
    print("\nConcatenating blocks along i-axis...")
    merged = concatenate_blocks(sampled_list, z_ref=0.0)
    print(f"Merged grid: type={type(merged).__name__}  "
          f"pts={merged.n_points}  cells={merged.n_cells}  "    
            f"cell_data={list(merged.cell_data.keys())}  "
            f"point_data={list(merged.point_data.keys())}")

    return merged

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
    panel_foil.compute_lift_coefficient()
    panel_foil.compute_coefficients_from_pressure()

    # Use cell-centre coordinates: centres are half a cell away from the wall
    # so k=0 centres are never on a panel and near-singular behaviour is avoided.
    X_c, Y_c = cell_centers(X, Y)   # shape (n_eta-1, n_xi-1)

    U_out, V_out, *_ = panel_foil.compute_velocity_field(X_c, Y_c)
    U  = U_out
    V  = V_out
    Cp = 1.0 - (U_out**2 + V_out**2) / U_inf**2

    # define a mesh grid
    #nx, ny = 100, 20  # number of points in the x and y directions
    #x_start, x_end = -1.0, 2.0
    #y_start, y_end = -0.3, 0.3
    #Xg, Yg = np.meshgrid(np.linspace(x_start, x_end, nx), np.linspace(y_start, y_end, ny))

    #panel_foil.plot_velocity_field(X, Y, path = f'{path}/Vec_Analytic_velocity_field.png')
    #panel_foil.plot_pressure_field(X, Y, path = f'{path}/Vec_Analytic_pressure_field.png')

    return X_c, Y_c, U, V, Cp


def plot_physical_space(cmesh,path, name, AOA, U_inf, show_plots=False  ):
    
    X = cmesh.points[:, 0]
    Y = cmesh.points[:, 1]

    #n_eta, n_xi = X.shape


    # Build a flat 3-D point array (z = 0)
    Z = np.zeros_like(X)
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    # PyVista StructuredGrid expects (n_xi, n_eta, n_z) order
    #grid = pv.StructuredGrid()
    #grid.points = points
    #grid.dimensions = (n_xi, n_eta, 1)
    #x_xi = cmesh.cell_data['x_xi']
    #x_eta = cmesh.cell_data['x_eta']
    #y_xi = cmesh.cell_data['y_xi']
    #y_eta = cmesh.cell_data['y_eta']
    #J = cmesh.cell_data['det_J']

    pl1 = pv.Plotter(shape=(3, 3), off_screen=not show_plots)

    data_plots = [
        (0, 0, "Cp", "Cp", (-1.0, 1.0)),
        (0, 1, "U_x", "U_x", (-1.0, 1.0)),
        (0, 2, "U_y", "U_y", (-1.0, 1.0)),

        (1, 0, "Cp_pot", "Cp_pot", (-1.0, 1.0)),
        (1, 1, "U_x_pot", "U_x_pot", (-1.0, 1.0)),
        (1, 2, "U_y_pot", "U_y_pot", (-1.0, 1.0)),

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



def plot_latent_space(cmesh,path, name,AOA,U_inf, show_plots=False):

    n_xi, n_eta, nz = cmesh.dimensions
    print(f"Grid dimensions: n_xi={n_xi}, n_eta={n_eta}, nz={nz}")
    xi_coords = np.arange(n_xi)
    eta_coords = np.arange(n_eta)
    XI, ETA, Z = np.meshgrid(xi_coords, eta_coords, [0], indexing='ij')

    latent_grid = pv.StructuredGrid( XI, ETA, Z)

    # J = cmesh.cell_data['det_J']
    # x_xi = cmesh.cell_data['x_xi']
    # x_eta = cmesh.cell_data['x_eta']
    # y_xi = cmesh.cell_data['y_xi']
    # y_eta = cmesh.cell_data['y_eta']

    # U_x = cmesh.cell_data['U_x']
    # U_y = cmesh.cell_data['U_y']
    # Cp = cmesh.cell_data['Cp']
    # Cp_pot = cmesh.cell_data['Cp_pot']
    # U_x_pot = cmesh.cell_data['U_x_pot']
    # U_y_pot = cmesh.cell_data['U_y_pot']
    # U_x_delta = cmesh.cell_data['U_x_delta']
    # U_y_delta = cmesh.cell_data['U_y_delta']


    pl1 = pv.Plotter(shape=(3, 3), off_screen=not show_plots)

    for field_name in cmesh.cell_data.keys():
        print(f"Transfer field data to latent space for {field_name}...{cmesh.cell_data[field_name].shape}")
        field = cmesh.cell_data[field_name]
        latent_grid.cell_data[field_name] = field.ravel(order='F')


    data_plots = [
        (0, 0, "Cp", "Cp", (-1.0, 1.0)),
        (0, 1, "U_x", "U_x", (-1.0, 1.0)),
        (0, 2, "U_y", "U_y", (-1.0, 1.0)),
        
        (1, 0, "Cp_pot", "Cp_pot", (-1.0, 1.0)),
        (1, 1, "U_x_pot", "U_x_pot", (-1.0, 1.0)),
        (1, 2, "U_y_pot", "U_y_pot", (-1.0, 1.0)),

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

def save_to_pytorch(cmesh, path, name):

    import torch

    nx,ny = cmesh.dimensions[:2]

    # Extract cell data and convert to PyTorch tensors
    cell_data = {}
    for key in cmesh.cell_data.keys():
        arr = cmesh.cell_data[key]
        tensor = torch.from_numpy(arr).float()  # Convert to float tensor
        cell_data[key] = tensor

    # Save the tensors to a file
    save_path = os.path.join(path, f"{name}_C_mesh_{nx}x{ny}.pt")
    torch.save(cell_data, save_path)
    print(f"Saved cell data tensors to {save_path}")

if __name__ == "__main__":

    ROOT_DIR = "/home/timm/Projects/PIML/OF_dataset"
    #name = "airFoil2D_SST_31.68_0.424_0.273_4.301_1.0_11.616"


    for d in os.listdir(ROOT_DIR):
        if d.startswith("airFoil2D_SST"):
            print(f"\nProcessing simulation: {d}")
            SIM_PATH = f"{ROOT_DIR}/{d}"
            VTM_PATH = f"{SIM_PATH}/constant/blockMeshVTK/blockMesh.vtm"
            FOAM_PATH = f"{SIM_PATH}/touch.foam"

            I0_min = 58
            J_MAX = 158 # 158 
            cmesh = get_cmesh(SIM_PATH, VTM_PATH, FOAM_PATH, I0_min, J_MAX, output_type='vts')
            nx, ny, nz = cmesh.dimensions
            X = cmesh.points[:, 0].reshape((nx, ny, nz), order='F').squeeze()
            Y = cmesh.points[:, 1].reshape((nx, ny, nz), order='F').squeeze()

            AF_ROOT = "/home/timm/Projects/PIML/Dataset"


            # Wake cells for 
            I_START = 80
            I_END = nx-80+1

            x_wall = X[I_START:I_END, 0]  # i varies along the wall
            y_wall = Y[I_START:I_END, 0]
            print(f"Extracted wall points: {len(x_wall)}")
            simulation = af.Simulation(root=AF_ROOT, name=os.name)

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



            print (f"Running panel method with {len(x_wall)} wall points...")
            AOA = simulation.angle_of_attack * 180 / np.pi
            U_inf = simulation.inlet_velocity
            print(f"Angle of attack: {AOA:.2f} degrees")
            print(f"Freestream velocity: {U_inf:.2f} m/s")

            print(f" Shape of X: {X.shape}  Y: {Y.shape} ")

            PATH_POTENTIAL = SIM_PATH + "/potential"
            os.makedirs(PATH_POTENTIAL, exist_ok=True)  

            X_c, Y_c, u, v, Cp = run_panel_method(x_wall,y_wall, X, Y,  alpha_aoa=AOA, U_inf=1.0, path=PATH_POTENTIAL)
            print(f"From OpenFoam Shape of X: {X.shape}  Y: {Y.shape}  ")
            print(f"Cell centers Shape of X_c: {X_c.shape}  Y_c: {Y_c.shape}  u: {u.shape}  v: {v.shape}  Cp: {Cp.shape} ")
            
            # put potential flow solution onto the Openfoam C-mesh 
            cmesh.cell_data['Cp_pot'] = Cp.ravel(order='F')
            cmesh.cell_data['U_x_pot'] = u.ravel(order='F') # already normalised by U_inf in run_panel_method
            cmesh.cell_data['U_y_pot'] = v.ravel(order='F') # already normalised by U_inf in run_panel_method

            # convert openfoam data 
            cmesh.cell_data['U_x'] = cmesh.cell_data['U'][:, 0] /U_inf
            cmesh.cell_data['U_y'] = cmesh.cell_data['U'][:, 1] /U_inf
            cmesh.cell_data.pop('U')  # remove original vector field to avoid   confusion
            rho = simulation.RHO
            cmesh.cell_data['Cp'] = cmesh.cell_data['p'] / (0.5 * rho * U_inf**2)  # rename pressure to Cp for consistency

            cmesh.cell_data.pop('p')  # remove original pressure field to avoid confusion
            cmesh.cell_data['Cp_delta'] = cmesh.cell_data['Cp'] - cmesh.cell_data['Cp_pot']
            cmesh.cell_data['U_x_delta'] = cmesh.cell_data['U_x'] - cmesh.cell_data['U_x_pot']
            cmesh.cell_data['U_y_delta'] = cmesh.cell_data['U_y'] - cmesh.cell_data['U_y_pot'] 

            plot_physical_space(cmesh, SIM_PATH, d, show_plots=False)
            plot_latent_space(cmesh, SIM_PATH, d, show_plots=False)
            save_to_pytorch(cmesh, SIM_PATH, d)


    # name = "airFoil2D_SST_32.137_12.122_4.854_5.202_9.247"
    # SIM_PATH = f"{ROOT_DIR}/{name}"
    # VTM_PATH = f"{SIM_PATH}/constant/blockMeshVTK/blockMesh.vtm"
    # FOAM_PATH = f"{SIM_PATH}/touch.foam"

    

    # I0_min = 58
    # J_MAX = 158 # 158 
    # cmesh = get_cmesh(SIM_PATH, VTM_PATH, FOAM_PATH, I0_min, J_MAX, output_type='vts')
 

    # # plt.figure(figsize=(10, 4))
    # # plt.subplot(1, 2, 1)
    # # plt.plot(x_wall, y_wall, 'o', markeredgecolor='b', markerfacecolor = 'none' , label='OpenFoam Mesh Surface')
    # # plt.plot(x_wall_af, y_wall_af, 'd',markeredgecolor='r', markerfacecolor = 'none' , label='Airfrans Airfoil')
    # # plt.xlabel('x')
    # # plt.ylabel('y')
    # # plt.title('Airfoil Geometry')
    # # plt.legend()    
    # # plt.show()



    # plot_physical_space(cmesh,AOA=AOA, U_inf=U_inf,path=SIM_PATH, name=name,show_plots=True)

    # plot_latent_space(cmesh,AOA=AOA, U_inf=U_inf,path=SIM_PATH, name=name,show_plots=True)

    # save_to_pytorch(cmesh, path=SIM_PATH, name=name)



