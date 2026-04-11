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

## Read an archived dataset file in pt format and plot the results to check that the data is correct.
def plot_physical_space(cmesh,path, name, show_plots=False, save_plots=False ):
    
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
        (0, 0, "Cp_rans", "Cp_rans", (-1.0, 1.0)),
        (0, 1, "U_x_rans", "U_x_rans", (-1.0, 1.0)),
        (0, 2, "U_y_rans", "U_y_rans", (-0.2, 0.2)),

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
    if save_plots:
        if not os.path.exists(f"{path}/physical/"):
            os.makedirs(f"{path}/physical/")
        pl1.screenshot(f"{path}/physical/{name}_physical_space.png", window_size=[1024, 768]*2)
    pl1.close()  

    U_rans = np.zeros((cmesh.n_points, 3))
    U_rans[:, 0] = cmesh.point_data["U_x_rans"] # U_x
    U_rans[:, 1] = cmesh.point_data["U_y_rans"] # U_y
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
    if save_plots:
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
    if save_plots:
        if not os.path.exists(f"{path}/physical/"):
            os.makedirs(f"{path}/physical/")
        pl2.screenshot(f"{path}/physical/{name}_meshcoeffs.png", window_size=[1024, 768]*2)

    pl2.close()  

    pl3 = pv.Plotter(shape=(2, 2), off_screen=not show_plots)

    plots = [
        (0, 0, "nut_rans", "nut_rans"),
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
    if save_plots:
        if not os.path.exists(f"{path}/physical/"):
            os.makedirs(f"{path}/physical/")
        pl3.screenshot(f"{path}/physical/{name}_sdf_nut.png", window_size=[1024, 768]*2)
    pl3.close()      



def plot_latent_space(cmesh,path, name, show_plots=False, save_plots=False):

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
        (0, 0, "Cp_rans", "Cp_rans", (-1.0, 1.0)),
        (0, 1, "U_x_rans", "U_x_rans", (-1.0, 1.0)),
        (0, 2, "U_y_rans", "U_y_rans", (-0.2, 0.2)),

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
    if save_plots:
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
    if save_plots:
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
    if save_plots:
        if not os.path.exists(f"{path}/latent/"):
            os.makedirs(f"{path}/latent/")
        pl3.screenshot(f"{path}/latent/{name}_sdf_nut.png", window_size=[1024, 768]*2)
    pl3.close()  



def plot_physical_space_channels(cmesh,channels,path, name, show_plots=False  ):
    # This function is currently not used but can be adapted to plot specific channels from the dataset if needed.
    pass

def convert_pt_to_cmesh_pyvista(x, y_delta,y_out):
    # x: (12, n_xi, n_eta)  — input features
    # y: (4,  n_xi, n_eta)  — output targets (delta / correction fields)
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(y_delta, torch.Tensor):
        y_delta = y_delta.detach().cpu().numpy()
    if isinstance(y_out, torch.Tensor):
        y_out = y_out.detach().cpu().numpy()

    n_channels, n_xi, n_eta = x.shape

    # Build the physical-space structured grid from coordinate channels 0 and 1
    X = x[0, :, :, np.newaxis]   # (n_xi, n_eta, 1)
    Y = x[1, :, :, np.newaxis]
    Z = np.zeros_like(X)
    cmesh = pv.StructuredGrid(X, Y, Z)

    # VTK structured-grid points are ordered with the first index varying fastest
    # (Fortran / column-major order), so we ravel with order='F'.
    def flat(arr2d):
        return arr2d.ravel(order='F')

    # ---- fields from x ----
    cmesh.point_data["U_x_pot"]    = flat(x[2])
    cmesh.point_data["U_y_pot"]    = flat(x[3])
    cmesh.point_data["Cp_pot"]     = flat(x[4])
    cmesh.point_data["sdf"]        = flat(x[5])
    cmesh.point_data["exp_sdf"]    = flat(x[6])
    cmesh.point_data["x_xi"]       = flat(x[7])
    cmesh.point_data["x_eta"]      = flat(x[8])
    cmesh.point_data["y_xi"]       = flat(x[9])
    cmesh.point_data["y_eta"]      = flat(x[10])
    cmesh.point_data["det_J"]      = flat(x[11])

    # ---- fields from y (delta/correction) ----
    cmesh.point_data["Cp_delta"]      = flat(y_delta[0])
    cmesh.point_data["U_x_delta"]     = flat(y_delta[1])
    cmesh.point_data["U_y_delta"]     = flat(y_delta[2])
    cmesh.point_data["log_nut_ratio"] = flat(y_delta[3])

    # ---- derived RANS fields: rans = potential + delta ----
    cmesh.point_data["U_x_rans"] = flat(y_out[0])
    cmesh.point_data["U_y_rans"] = flat( y_out[1])
    cmesh.point_data["p_rans"]  = flat( y_out[2])
    cmesh.point_data["nut_rans"] = flat( y_out[3])
    #cmesh.point_data["wall_shear_stress_x_rans"] = flat( y_out[4])
    #cmesh.point_data["wall_shear_stress_y_rans"] = flat( y_out[5])
    return cmesh

def physical_to_latent_mapping(cmesh, show_plots=False):

    n_xi ,n_eta, nz = cmesh.dimensions
    print(f"Grid dimensions: n_xi={n_xi}, n_eta={n_eta}, nz={nz}")
    xi_coords = np.arange(n_xi)
    eta_coords = np.arange(n_eta)
    XI, ETA, Z = np.meshgrid(xi_coords, eta_coords, [0], indexing='ij')
    latent_grid = pv.StructuredGrid( XI, ETA, Z)    
    # 1. Compute 'True' Physical Gradient in VTK space
    # This uses the physical XYZ coordinates of your C-mesh
    grad_mesh = cmesh.compute_derivative(scalars="Cp_pot", gradient="grad_Cp_physical")
    true_grad = grad_mesh.point_data["grad_Cp_physical"].reshape(n_xi, n_eta, 3, order='F')
    # 2. Compute Computational Gradients (Latent Space)
    # In latent space, the grid is uniform (d_xi = 1, d_eta = 1)
    # We reshape back to 2D to use numpy.gradient
    cp_latent = cmesh.point_data["Cp_pot"].reshape(n_xi, n_eta, order='F')
    dCp_dxi, dCp_deta = np.gradient(cp_latent, axis=(0, 1))

    # 3. Reconstruct Physical Gradient using your stored Jacobians
    # Pull metrics back to 2D (Fortran order to match cmesh.point_data)
    x_xi  = cmesh.point_data["x_xi"].reshape(n_xi, n_eta, order='F')
    x_eta = cmesh.point_data["x_eta"].reshape(n_xi, n_eta, order='F')
    y_xi  = cmesh.point_data["y_xi"].reshape(n_xi, n_eta, order='F')
    y_eta = cmesh.point_data["y_eta"].reshape(n_xi, n_eta, order='F')
    det_J = cmesh.point_data["det_J"].reshape(n_xi, n_eta, order='F')

    # Inverse Jacobian Transformation
    inv_det = 1.0 / (det_J + 1e-10)
    recon_grad_x = inv_det * ( y_eta * dCp_dxi - y_xi * dCp_deta)
    recon_grad_y = inv_det * (-x_eta * dCp_dxi + x_xi * dCp_deta)

    # 4. Compare
    error_x = np.abs(true_grad[:,:,0] - recon_grad_x)
    print(f"Mean Gradient Error X: {np.mean(error_x):.2e}")
    error_y = np.abs(true_grad[:,:,1] - recon_grad_y)
    print(f"Mean Gradient Error Y: {np.mean(error_y):.2e}")

    # 4. Attach to Latent Grid for Visualization
    # Use .ravel(order='F') to ensure VTK-compatibility
    latent_grid.point_data["True_Grad_X"] = true_grad[:,:,0].ravel(order='F')
    latent_grid.point_data["Recon_Grad_X"] = recon_grad_x.ravel(order='F')
    latent_grid.point_data["Error_X"] = np.abs(true_grad[:,:,0] - recon_grad_x).ravel(order='F')
    
    latent_grid.point_data["True_Grad_Y"] = true_grad[:,:,1].ravel(order='F')
    latent_grid.point_data["Recon_Grad_Y"] = recon_grad_y.ravel(order='F')
    latent_grid.point_data["Error_Y"] = np.abs(true_grad[:,:,1] - recon_grad_y).ravel(order='F')

    plotter = pv.Plotter(shape=(1, 2), off_screen=not show_plots)

    # Subplot 1: The reconstructed X-gradient on the latent grid
    plotter.subplot(0, 0)
    plotter.add_mesh(latent_grid, scalars="Recon_Grad_X", cmap="RdBu_r")
    plotter.add_text("Latent Space: Reconstructed dCp/dx")

    # Subplot 2: The Absolute Error
    plotter.subplot(0, 1)
    plotter.add_mesh(latent_grid, scalars="Error_X", cmap="viridis")
    plotter.add_text("Latent Space: Gradient Reconstruction Error")

    plotter.show()

    return latent_grid

import numpy as np
from scipy.integrate import cumulative_trapezoid

def test_coordinate_reconstruction(cmesh,show_plots=False):
    ni, nj, _ = cmesh.dimensions
    print(f"Testing Coordinate Reconstruction for n_xi={ni}, n_eta={nj}...")

    # 1. Extract True Coordinates and Metrics (Using order='F'!)
    X_true = cmesh.points[:, 0].reshape((ni, nj), order='F')
    Y_true = cmesh.points[:, 1].reshape((ni, nj), order='F')

    x_xi  = cmesh.point_data["x_xi"].reshape((ni, nj), order='F')
    x_eta = cmesh.point_data["x_eta"].reshape((ni, nj), order='F')
    y_xi  = cmesh.point_data["y_xi"].reshape((ni, nj), order='F')
    y_eta = cmesh.point_data["y_eta"].reshape((ni, nj), order='F')

    # 2. Initialize Reconstructed Arrays
    X_recon = np.zeros((ni, nj))
    Y_recon = np.zeros((ni, nj))

    # 3. Integrate along the bottom boundary (eta = 0, changing xi)
    X_recon[:, 0] = X_true[0, 0] + cumulative_trapezoid(x_xi[:, 0], dx=1, initial=0)
    Y_recon[:, 0] = Y_true[0, 0] + cumulative_trapezoid(y_xi[:, 0], dx=1, initial=0)

    # 4. Integrate outward (changing eta) for every xi slice
    for i in range(ni):
        X_recon[i, :] = X_recon[i, 0] + cumulative_trapezoid(x_eta[i, :], dx=1, initial=0)
        Y_recon[i, :] = Y_recon[i, 0] + cumulative_trapezoid(y_eta[i, :], dx=1, initial=0)

    # 5. Calculate Absolute Errors
    error_X = np.abs(X_true - X_recon)
    error_Y = np.abs(Y_true - Y_recon)

    print(f"Mean Coordinate Error X: {np.mean(error_X):.4e}")
    print(f"Max Coordinate Error X:  {np.max(error_X):.4e}")
    print(f"Mean Coordinate Error Y: {np.mean(error_Y):.4e}")
    print(f"Max Coordinate Error Y:  {np.max(error_Y):.4e}")

    # Attach to cmesh for plotting if desired
    cmesh.point_data["Error_Coord_X"] = error_X.ravel(order='F')
    cmesh.point_data["Error_Coord_Y"] = error_Y.ravel(order='F')

    plotter = pv.Plotter(shape=(1, 2), off_screen=not show_plots)

    # Subplot 1: The reconstructed X-gradient on the latent grid
    plotter.subplot(0, 0)
    plotter.add_mesh(cmesh, scalars="Error_Coord_X", cmap="RdBu_r")
    plotter.add_text("Latent Space: Coordinate Reconstruction Error X")

    # Subplot 2: The Absolute Error
    plotter.subplot(0, 1)
    plotter.add_mesh(cmesh, scalars="Error_Coord_Y", cmap="RdBu_r")
    plotter.add_text("Latent Space: Coordinate Reconstruction Error Y")

    plotter.show()

    return error_X, error_Y

if __name__ == "__main__":

    # Load the dataset
    STORAGE_DIR ="/home/timm/storage/AF_NO_DATASET/Archive"

    names = ["airFoil2D_SST_33.32_11.069_6.369_2.254_12.837"]


    resolutions = [(512, 64), (256, 32)]
    resolutions = [ (1194, 159)]

    # x channels
    # 0  = x
    # 1  = y
    # 2  = U_x_pot
    # 3  = U_y_pot
    # 4  = Cp_pot
    # 5  = sdf
    # 6  = exp_sdf
    # 7  = x_xi
    # 8  = x_eta
    # 9  = y_xi
    # 10 = y_eta
    # 11 = det_J

    # y channels
    # 0 = Cp_dela
    # 1 = U_x_delta
    # 2 = U_y_delta
    # 3 = log_nut_ratio

    # y_out channels
    # 0 = U_x_rans
    # 1 = U_y_rans
    # 2 = Cp_rans
    # 3 = nut_rans
    # 4 = wall_shear_stress_x_rans
    # 5 = wall_shear_stress_y_rans



    for name in names:
        print(f"Processing case: {name}")
        for res in resolutions:
            data_file = f"{STORAGE_DIR}/{name}_C_mesh_{res[0]}x{res[1]}.pt"
            raw_data = torch.load(data_file)
            print(f"Loaded data for {name} with keys: {raw_data.keys()}")
            x = raw_data["x"] # Input features (latent space)
            print(f"Input features shape: {x.shape}")
            y_delta = raw_data["y_delta"] # Output targets (delta/correction fields)
            print(f"Output targets shape: {y_delta.shape}")
            y_out = raw_data["y_out"] # Output targets (delta/correction fields)
            print(f"Output targets shape: {y_out.shape}")        
        #cmesh = dataset[name]["cmesh"]

        cmesh = convert_pt_to_cmesh_pyvista(x, y_delta, y_out)

        plot_physical_space(cmesh, STORAGE_DIR, name, show_plots=True, save_plots=False)
        physical_to_latent_mapping(cmesh, show_plots=True)
        test_coordinate_reconstruction(cmesh, show_plots=True)
        #plot_latent_space(cmesh, STORAGE_DIR, name, show_plots=False)

