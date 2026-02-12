import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from foil_utils import sort_airfoil_to_sellig, get_smoothed_airfoil_spline, generate_aligned_o_grid, debug_mesh_connectivity
import airfrans as af


import numpy as np

import matplotlib.pyplot as plt
import numpy as np

import pyvista as pv
import numpy as np

def plot_u_velocity_pv(grid_coords, u_phys, v_phys, phi_xi__xi_x, phi_eta__eta_x):
    # Create the base StructuredGrid
    grid = pv.StructuredGrid(grid_coords[:,:,0], grid_coords[:,:,1], grid_coords[:,:,2])
    
    # Attach u_velocity and the vector field
    grid.point_data["u_vel"] = u_phys.flatten(order='F')
    grid.point_data["v_vel"] = v_phys.flatten(order='F')
    
    vel_vectors = np.zeros((grid.n_points, 3))
    vel_vectors[:, 0] = u_phys.flatten(order='F')
    vel_vectors[:, 1] = v_phys.flatten(order='F')
    grid.point_data["Velocity"] = vel_vectors
    grid.point_data["phi_xi__xi_x"] = phi_xi__xi_x.flatten(order='F')
    grid.point_data["phi_eta__eta_x"] = phi_eta__eta_x.flatten(order='F')

    plotter = pv.Plotter(window_size=[1000, 800])
    plotter.set_background("white")

    # Specialized meshes for scalars and streamlines
    grid_u = grid.copy()
    grid_u.set_active_scalars("phi_xi__xi_x")
    grid_vel = grid.copy()
    grid_vel.set_active_vectors("Velocity")
    
    # 1. Plot u-velocity (Look for symmetry and far-field V_inf)
    # At alpha=0, u should be ~1.0 everywhere except near the foil
    plotter.add_mesh(grid_u, scalars="phi_xi__xi_x",
                      cmap="viridis", 
                      clim=[0.0, 1.5], # Anticipated range for V_inf=1.0
                      lighting=False, show_edges=False)
    
    # 2. Add Streamlines to see the path vs the field
    streamline_seeds = pv.Line(pointa=(-3.5, -2.0, 0), pointb=(-3.5, 2.0, 0), resolution=10)
    streamlines = grid_vel.streamlines_from_source(streamline_seeds, 
                                                   vectors="Velocity", 
                                                   integration_direction='both', 
                                                   initial_step_length=0.01,
                                                   max_steps=1000,
                                                   max_time=15)

    if streamlines.n_points > 0:
        plotter.add_mesh(streamlines, color="black", line_width=2)
    
    plotter.view_xy()
    plotter.show()

def plot_flowfield_pv(grid_coords, u_phys, v_phys, cp_2d):
    # Create the base StructuredGrid
    grid = pv.StructuredGrid(grid_coords[:,:,0], grid_coords[:,:,1], grid_coords[:,:,2])
    
    # CRITICAL: Attach data to the base grid FIRST
    grid.point_data["Cp"] = cp_2d.flatten(order='F')
    
    # Combine u and v for the velocity vectors
    vel_vectors = np.zeros((grid.n_points, 3))
    vel_vectors[:, 0] = u_phys.flatten(order='F')
    vel_vectors[:, 1] = v_phys.flatten(order='F')
    grid.point_data["Velocity"] = vel_vectors

    plotter = pv.Plotter(window_size=[1000, 800])
    plotter.set_background("white")

    # Now create the copies for specialized plotting
    grid_cp = grid.copy()
    grid_cp.set_active_scalars("Cp")
    
    grid_vel = grid.copy()
    grid_vel.set_active_vectors("Velocity")
    
    # 1. Add the Pressure Map
    plotter.add_mesh(grid_cp, scalars="Cp",
                      cmap="RdBu_r", 
                      clim=[-2.0, 1.0], # Range for chord=4
                      lighting=False, show_edges=False)
    
    # 2. Add Streamlines (Seed points upstream at x=-3.5)
    streamline_seeds = pv.Line(pointa=(-3.5, -2.0, 0), pointb=(-3.5, 2.0, 0), resolution=30)
    streamlines = grid_vel.streamlines_from_source(streamline_seeds, 
                                                   vectors="Velocity", 
                                                   integration_direction='both', 
                                                   initial_step_length=0.01,
                                                   max_steps=1000,
                                                max_time=15)

    if streamlines.n_points > 0:
        plotter.add_mesh(streamlines, color="black", line_width=2)
    
    plotter.view_xy()
    plotter.show()


def plot_flowfield_pv2(grid_coords, u_phys, v_phys, cp_2d):
    # Create the base StructuredGrid
    grid = pv.StructuredGrid(grid_coords[:,:,0], grid_coords[:,:,1], grid_coords[:,:,2])
    
    # CRITICAL: Attach data to the base grid FIRST
    grid.point_data["Cp"] = cp_2d.flatten(order='F')
    
    # Combine u and v for the velocity vectors
    vel_vectors = np.zeros((grid.n_points, 3))
    vel_vectors[:, 0] = u_phys.flatten(order='F')
    vel_vectors[:, 1] = v_phys.flatten(order='F')
    grid.point_data["Velocity"] = vel_vectors
    
    # Debug: Check velocity field
    vel_mag = np.sqrt(vel_vectors[:, 0]**2 + vel_vectors[:, 1]**2)
    print(f"DEBUG Streamlines: vel_mag range: {vel_mag.min():.4f} to {vel_mag.max():.4f}")
    print(f"DEBUG Streamlines: grid x range: {grid_coords[:,:,0].min():.2f} to {grid_coords[:,:,0].max():.2f}")
    print(f"DEBUG Streamlines: grid y range: {grid_coords[:,:,1].min():.2f} to {grid_coords[:,:,1].max():.2f}")

    plotter = pv.Plotter(window_size=[1200, 900])
    plotter.set_background("white")

    # Now create the copies for specialized plotting
    grid_cp = grid.copy()
    grid_cp.set_active_scalars("Cp")
    
    # 1. Add the Pressure Map
    plotter.add_mesh(grid_cp, scalars="Cp",
                      cmap="RdBu_r", 
                      clim=[-2.0, 1.0],
                      lighting=False, show_edges=False)
    
    # 2. Add Streamlines
    # Resample velocity onto a uniform rectilinear grid for reliable streamline integration
    x_min, x_max = grid_coords[:,:,0].min(), grid_coords[:,:,0].max()
    y_min, y_max = grid_coords[:,:,1].min(), grid_coords[:,:,1].max()
    
    # Create a uniform grid that covers the domain
    nx_resample, ny_resample = 200, 200
    x_uniform = np.linspace(x_min * 0.95, x_max * 0.95, nx_resample)
    y_uniform = np.linspace(y_min * 0.95, y_max * 0.95, ny_resample)
    z_uniform = np.array([0.0])
    
    uniform_grid = pv.RectilinearGrid(x_uniform, y_uniform, z_uniform)
    
    # Interpolate the structured grid data onto the uniform grid
    resampled = uniform_grid.sample(grid)
    
    # Seed points well upstream of the airfoil
    seed_x = x_min * 0.5  # 50% of the way to the left boundary
    n_seeds = 25
    seed_y = np.linspace(y_min * 0.5, y_max * 0.5, n_seeds)
    
    print(f"DEBUG Streamlines: seed x = {seed_x:.2f}, seed y range: [{seed_y.min():.2f}, {seed_y.max():.2f}]")
    
    streamline_seeds = pv.Line(
        pointa=(seed_x, seed_y.min(), 0), 
        pointb=(seed_x, seed_y.max(), 0), 
        resolution=n_seeds
    )
    
    try:
        streamlines = resampled.streamlines_from_source(
            streamline_seeds, 
            vectors="Velocity", 
            integration_direction='both', 
            initial_step_length=0.05,
            max_steps=2000,
            max_time=50.0
        )
        
        if streamlines.n_points > 0:
            plotter.add_mesh(streamlines, color="black", line_width=1.5)
            print(f"DEBUG Streamlines: Generated {streamlines.n_lines} streamlines with {streamlines.n_points} points")
        else:
            print("WARNING: No streamline points generated!")
    except Exception as e:
        print(f"WARNING: Streamline generation failed: {e}")
    
    plotter.view_xy()
    plotter.show()

def solve_potential_flow_final(grid_coords, V_inf=1.0, alpha_deg=0.0):
    ni, nj = grid_coords.shape[0], grid_coords.shape[1]
    alpha = np.radians(alpha_deg)
    
    # 1. Setup Canonical Polar Coordinates (Circle World)
    theta = np.linspace(0, 2*np.pi, ni)
    r = np.linspace(1.0, 8.0, nj) # Ensure r_max matches your grid span
    T, R = np.meshgrid(theta, r, indexing='ij')
    z = R * np.exp(1j * T)

    # 2. Potential Flow Physics (Kutta Condition)
    gamma = 4 * np.pi * V_inf * 1.0 * np.sin(alpha) 
    dw_dz = V_inf * (np.exp(-1j * alpha) - (1.0**2 * np.exp(1j * alpha)) / z**2) + \
            (1j * gamma) / (2 * np.pi * z)
    
    # 3. Call the v4 Transformation
    # This function handles the sign flips and xi/eta basis mapping internally.
    results = finalized_metrics_v4(grid_coords, dw_dz, T, R)
    
    # 4. Final pressure calculation using correctly projected velocities
    v_mag_sq = results['u']**2 + results['v']**2
    cp_2d = 1.0 - (v_mag_sq / V_inf**2)
    
    return {
        'x': results['u'],   # Horizontal velocity channel for GINO
        'y': results['v'],   # Vertical velocity channel for GINO
        'cp': cp_2d,         # Pressure prior
        'error': results['val_xi'] + results['val_eta'] # Identity check
    }

def solve_and_transform_to_foil_v2(grid_coords, metrics, V_inf=1.0, alpha_deg=2.983):
    ni, nj = grid_coords.shape[0], grid_coords.shape[1]
    alpha = np.radians(alpha_deg)
    
    # 1. Setup Canonical Polar Coordinates matching the grid topology
    # xi (axis 0) wraps around the airfoil -> maps to theta [0, 2pi]
    # eta (axis 1) goes away from surface -> maps to r [1, r_max]
    theta_1d = np.linspace(0, 2*np.pi, ni, endpoint=False)  # periodic, no duplicate
    r_max = 8.0  # should match your grid generation R_far
    r_1d = np.linspace(1.0, r_max, nj)
    
    T, R = np.meshgrid(theta_1d, r_1d, indexing='ij')  # Shape: (ni, nj)
    z = R * np.exp(1j * T)
    
    # 2. Potential Flow on the Unit Circle
    gamma = 4 * np.pi * V_inf * 1.0 * np.sin(alpha)  # Kutta condition
    
    # Complex potential derivative: dw/dz
    dw_dz = V_inf * (np.exp(-1j * alpha) - (1.0**2 * np.exp(1j * alpha)) / z**2) + \
            (1j * gamma) / (2 * np.pi * z)
    
    # 3. Canonical Cartesian velocities in CIRCLE space
    u_circle = np.real(dw_dz)
    v_circle = -np.imag(dw_dz)
    
    # 4. Contravariant velocities in logical space
    # These are the velocities expressed in the (xi, eta) coordinate directions
    # Using the INVERSE of the physical-to-canonical mapping:
    #   U^xi  = (d_xi/d_xc) * u_circle + (d_xi/d_yc) * v_circle
    #   U^eta = (d_eta/d_xc) * u_circle + (d_eta/d_yc) * v_circle
    #
    # Where (xc, yc) are the canonical circle cartesian coords.
    # Since canonical space is just (r*cos(theta), r*sin(theta)):
    #   d_xc/d_xi = d(r*cos(theta))/d_xi = -R*sin(T) * d_theta/d_xi
    #   d_yc/d_xi = d(r*sin(theta))/d_xi =  R*cos(T) * d_theta/d_xi
    #   d_xc/d_eta = cos(T) * dr/d_eta
    #   d_yc/d_eta = sin(T) * dr/d_eta
    
    d_theta = 2 * np.pi / ni      # theta step per xi index
    d_r = (r_max - 1.0) / (nj - 1)  # r step per eta index
    
    # Forward derivatives of canonical coords w.r.t. logical coords
    dxc_dxi  = -R * np.sin(T) * d_theta
    dyc_dxi  =  R * np.cos(T) * d_theta
    dxc_deta =  np.cos(T) * d_r
    dyc_deta =  np.sin(T) * d_r
    
    # Jacobian of canonical mapping
    J_canon = dxc_dxi * dyc_deta - dxc_deta * dyc_dxi
    inv_J_canon = 1.0 / np.clip(np.abs(J_canon), 1e-12, None)
    
    # Inverse: logical coords w.r.t. canonical cartesian
    dxi_dxc  =  dyc_deta * inv_J_canon
    dxi_dyc  = -dxc_deta * inv_J_canon
    deta_dxc = -dyc_dxi  * inv_J_canon
    deta_dyc =  dxc_dxi  * inv_J_canon
    
    # Contravariant velocities in logical space
    U_xi  = dxi_dxc  * u_circle + dxi_dyc  * v_circle
    U_eta = deta_dxc * u_circle + deta_dyc * v_circle
    
    # 5. Transform from logical space to physical airfoil space
    # Using the FORWARD physical grid metrics:
    #   u_phys = U^xi * dx/d_xi + U^eta * dx/d_eta
    #   v_phys = U^xi * dy/d_xi + U^eta * dy/d_eta
    
    # Need to normalize metrics to per-unit-logical-step
    d_xi_phys = 1.0 / (ni - 1)
    d_eta_phys = 1.0 / (nj - 1)
    
    x_xi  = metrics['x_xi']  * d_xi_phys   # These are already per-unit from finalized_metrics_v3
    y_xi  = metrics['y_xi']  * d_xi_phys
    x_eta = metrics['x_eta'] * d_eta_phys
    y_eta = metrics['y_eta'] * d_eta_phys
    
    u_phys = U_xi * x_xi + U_eta * x_eta
    v_phys = U_xi * y_xi + U_eta * y_eta
    
    # 6. Pressure coefficient
    v_mag_sq = u_phys**2 + v_phys**2
    cp_2d = 1.0 - (v_mag_sq / V_inf**2)
    
    # Debug output
    print(f"DEBUG: Canonical u range: {u_circle.min():.4f} to {u_circle.max():.4f}")
    print(f"DEBUG: Canonical v range: {v_circle.min():.4f} to {v_circle.max():.4f}")
    print(f"DEBUG: U_xi range: {U_xi.min():.4f} to {U_xi.max():.4f}")
    print(f"DEBUG: U_eta range: {U_eta.min():.4f} to {U_eta.max():.4f}")
    print(f"DEBUG: J_canon range: {J_canon.min():.6f} to {J_canon.max():.6f}")
    print(f"DEBUG: Physical u range: {u_phys.min():.4f} to {u_phys.max():.4f}")
    print(f"DEBUG: Physical v range: {v_phys.min():.4f} to {v_phys.max():.4f}")
    print(f"DEBUG: Cp range: {cp_2d.min():.4f} to {cp_2d.max():.4f}")
    
    return u_phys, v_phys, cp_2d


def solve_and_transform_to_foil(grid_coords, metrics, V_inf=1.0, alpha_deg=2.983):
    ni, nj = grid_coords.shape[0], grid_coords.shape[1]
    alpha = np.radians(alpha_deg)
    
    # 1. Setup Canonical Polar Coordinates (Circle World)
    # R=1 maps to chord=4. r_max should match your grid extrusion.
    theta = np.linspace(0, 2*np.pi, ni)
    r = np.linspace(1.0, 5.0, nj) # Example r_max=5
    T, R = np.meshgrid(theta, r, indexing='ij')
    z = R * np.exp(1j * T)

    # 2. Potential Flow Physics (Kutta Condition Locked)
    # Gamma derived from TE stagnation at theta=0
    gamma = 4 * np.pi * V_inf * 1.0 * np.sin(alpha) 
    
    # Complex Velocity (dw/dz) in Circle Space
    dw_dz = V_inf * (np.exp(-1j * alpha) - (1.0**2 * np.exp(1j * alpha)) / z**2) + \
            (1j * gamma) / (2 * np.pi * z)
    
    # R min and max
    r_max = R.max()
    r_min = R.min()

    
    # 1. Circle Cartesian Velocities (Orthogonal)
    u_circle = np.real(dw_dz)
    v_circle = -np.imag(dw_dz)

    # 2. Project onto Logical Basis (phi_xi = tangential, phi_eta = radial)
    # T is your local polar angle grid.
    # This aligns the orthogonal circle flow with the wrap-around grid lines.
    phi_xi_raw = -u_circle * np.sin(T) + v_circle * np.cos(T)
    phi_eta_raw = u_circle * np.cos(T) + v_circle * np.sin(T)

    # 3. Apply Domain Scaling
    phi_xi = phi_xi_raw * (2 * np.pi)
    phi_eta = phi_eta_raw * (r_max - r_min) 

    # Need to flip x_xi and y_xi to match the correct direction along the foil

    x_xi_base = metrics['x_eta'] 
    y_xi_base = metrics['y_eta']
    x_eta_base = metrics['x_xi']
    y_eta_base = metrics['y_xi']

    # 2. The Orientation: Correct for Clockwise Upper-Surface indexing
    # This aligns the wrap-around direction with the potential flow
    x_xi_true = -x_xi_base
    y_xi_true = -y_xi_base

    # 4. Use Validated Metrics to get Physical Cartesian (u, v)
    u_phys = phi_xi * x_xi_true + phi_eta * x_eta_base
    v_phys = phi_xi * y_xi_true + phi_eta * y_eta_base
    phi_xi__xi_x = phi_xi * x_xi_true
    phi_eta__eta_x = phi_eta * x_eta_base

    v_mag_sq = u_phys**2 + v_phys**2
    cp_2d = 1.0 - (v_mag_sq / V_inf**2)    

        # Debug output
    print(f"DEBUG: Canonical u range: {u_circle.min():.4f} to {u_circle.max():.4f}")
    print(f"DEBUG: Canonical v range: {v_circle.min():.4f} to {v_circle.max():.4f}")
    print(f"DEBUG: Phi_xi range: {phi_xi.min():.4f} to {phi_xi.max():.4f}")
    print(f"DEBUG: Phi_eta range: {phi_eta.min():.4f} to {phi_eta.max():.4f}")
    print(f"DEBUG: Physical u range: {u_phys.min():.4f} to {u_phys.max():.4f}")
    print(f"DEBUG: Physical v range: {v_phys.min():.4f} to {v_phys.max():.4f}")
    print(f"DEBUG: Cp range: {cp_2d.min():.4f} to {cp_2d.max():.4f}")
    print(f"DEBUG: Phi_xi__xi_x range: {phi_xi__xi_x.min():.4f} to {phi_xi__xi_x.max():.4f}")
    print(f"DEBUG: Phi_eta__eta_x range: {phi_eta__eta_x.min():.4f} to {phi_eta__eta_x.max():.4f}")
    return u_phys, v_phys, cp_2d, phi_xi__xi_x, phi_eta__eta_x


def plot_grid_metrics(grid_coords, metrics):
    # Extract X, Y for the physical mesh background
    X = grid_coords[:,:,0]
    Y = grid_coords[:,:,1]
    
    # We will plot the 4 components of the transformation matrix and J
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    # Flip xi derivatives to match physical orientation    
    # The 'Golden State' alignment
    metrics['x_xi_true'] = metrics['x_eta'] # This is actually the wrap-around stretch
    metrics['x_eta_true'] = metrics['x_xi'] # This is actually the radial expansion

    metrics['y_xi_true'] = metrics['y_eta'] 
    metrics['y_eta_true'] = metrics['y_xi']
    # List of metrics to plot
    plot_data = [
        (metrics['x_xi_true'], 'x_xi (Stretch along Foil)'),
        (metrics['x_eta_true'], 'x_eta (Stretch away from Foil)'),
        (metrics['J'], 'Jacobian (Local Area)'),
        (metrics['y_xi_true'], 'y_xi (Slope along Foil)'),
        (metrics['y_eta_true'], 'y_eta (Slope away from Foil)'),
        (metrics['error'], 'Identity Error (Should be ~0)')
    ]
    
    for i, (data, title) in enumerate(plot_data):
        ax = axes[i]
        # Use a diverging colormap for derivatives, sequential for J/Error
        cmap = 'RdBu_r' if 'xi' in title or 'eta' in title else 'viridis'
        
        im = ax.pcolormesh(X, Y, data, cmap=cmap, shading='auto')
        plt.colorbar(im, ax=ax)
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.set_xlim([-4, 6]) # Focus on the airfoil region
        ax.set_ylim([-3, 3])

    plt.tight_layout()
    plt.show()


def finalized_metrics_v4(grid_coords, dw_dz, T, R):
    """
    Combines high-precision metrics with physical basis alignment 
    to solve the 'orbiting' and 'butterfly' errors.
    """
    ni, nj = grid_coords.shape[0], grid_coords.shape[1]
    
    # 1. Normalize Logical Steps
    d_xi = 1.0 / (ni - 1)
    d_eta = 1.0 / (nj - 1)
    
    # 2. Compute Physical Derivatives (Meters per Logical-Unit)
    # Mapping Axis 0 -> xi (Wrap), Axis 1 -> eta (Radial)
    x_xi = np.gradient(grid_coords[:,:,0], axis=0) / d_xi
    y_xi = np.gradient(grid_coords[:,:,1], axis=0) / d_xi
    x_eta = np.gradient(grid_coords[:,:,0], axis=1) / d_eta
    y_eta = np.gradient(grid_coords[:,:,1], axis=1) / d_eta
    
    # 3. Handle Handedness (Corrects for Clockwise Upper-Surface Indexing)
    # As seen in image_2c0e19.png, we must align grid direction with potential flow.
    J = x_xi * y_eta - x_eta * y_xi
    if np.mean(J) < 0:
        x_xi, y_xi = -x_xi, -y_xi # Align grid to anti-clockwise flow
        J = -J
    
    # 4. Inverse Metrics (The 'Pull-Back' to Circle Space)
    inv_J = 1.0 / np.clip(J, 1e-12, None)
    xi_x =  y_eta * inv_J
    xi_y = -x_eta * inv_J
    eta_x = -y_xi * inv_J
    eta_y =  x_xi * inv_J

    # 5. Potential Flow Projections (Corrected Basis)
    # u_circle and v_circle from dw_dz are Cartesian in the canonical plane.
    u_circle = np.real(dw_dz)
    v_circle = -np.imag(dw_dz)

    # Transform to Logical Basis (Projecting circle-cartesian to grid-tangential/radial)
    # This aligns the orthogonal flow with the wrap-around C-grid.
    phi_xi_raw = -u_circle * np.sin(T) + v_circle * np.cos(T)
    phi_eta_raw = u_circle * np.cos(T) + v_circle * np.sin(T)

    # Scale to match metric normalization
    phi_xi = phi_xi_raw * (2 * np.pi)
    phi_eta = phi_eta_raw * (R.max() - R.min())

    # 6. Final Physical Velocity Projection
    # This solves the 'butterfly' by using the correct basis mapping.
    u_phys = (phi_xi * xi_x + phi_eta * eta_x)
    v_phys = (phi_xi * xi_y + phi_eta * eta_y)
    
    # 7. Validation metrics
    val_xi = xi_x * x_xi + xi_y * y_xi
    val_eta = eta_x * x_eta + eta_y * y_eta

    return {
        'u': u_phys, 'v': v_phys, 
        'val_xi': val_xi, 'val_eta': val_eta,
        'J': J
    }


def finalized_metrics_v3(grid_coords):
    ni, nj = grid_coords.shape[0], grid_coords.shape[1]
    
    # 1. Normalize the Logical Steps (Meters per Unit-Logical-Length)
    # This clears the 560.0 magnitude error and the 10^15 far-field explosion.
    d_xi = 1.0 / (ni - 1)
    d_eta = 1.0 / (nj - 1)
    
    # 2. Compute Derivatives with Normalized Steps
    x_xi = np.gradient(grid_coords[:,:,0], axis=0) / d_xi
    y_xi = np.gradient(grid_coords[:,:,1], axis=0) / d_xi
    x_eta = np.gradient(grid_coords[:,:,0], axis=1) / d_eta
    y_eta = np.gradient(grid_coords[:,:,1], axis=1) / d_eta
    
    # 3. Jacobian with Direction Correction
    # We ensure J is positive to maintain a right-handed coordinate system.
    J = x_xi * y_eta - x_eta * y_xi
    if np.mean(J) < 0:
        J = -J
        # Realign derivatives to match the positive Jacobian orientation
        x_xi, y_xi = -x_xi, -y_xi 

    # 4. Inverse Metrics (The 'Pull-Back' to Circle Space)
    inv_J = 1.0 / np.clip(J, 1e-12, None)
    xi_x =  y_eta * inv_J
    xi_y = -x_eta * inv_J
    eta_x = -y_xi * inv_J
    eta_y =  x_xi * inv_J
    
    # 5. Dual-Direction Identity Check
    # These should both be exactly 1.0 everywhere.
    val_xi = xi_x * x_xi + xi_y * y_xi
    val_eta = eta_x * x_eta + eta_y * y_eta
    
    # Total Error should now be 0.0, not 4.0
    identity_error = np.abs(val_xi - 1.0) + np.abs(val_eta - 1.0)
    
    metrics = {
        'J': J, 'inv_J': inv_J,
        'x_xi': x_xi, 'y_xi': y_xi, 'x_eta': x_eta, 'y_eta': y_eta,
        'xi_x': xi_x, 'xi_y': xi_y, 'eta_x': eta_x, 'eta_y': eta_y,
        'error': identity_error
    }

    print(f"Jacobian stats: min {J.min():.2e}, max {J.max():.2e}, mean {J.mean():.2e}")
    print(f"val_xi stats: min {val_xi.min():.2e}, max {val_xi.max():.2e}, mean {val_xi.mean():.2e}")
    print(f"val_eta stats: min {val_eta.min():.2e}, max {val_eta.max():.2e}, mean {val_eta.mean():.2e}")
    
    return metrics, val_xi, val_eta

def validate_metrics(metrics):
    J = metrics['J']
    # Use a safety floor to avoid the 10^15 errors near singularities
    inv_J = 1.0 / np.clip(J, 1e-12, None)
    
    # Inverse Metrics: mapping from physical (x,y) back to logical (xi, eta)
    xi_x =  metrics['y_eta'] * inv_J
    xi_y = -metrics['x_eta'] * inv_J
    eta_x = -metrics['y_xi'] * inv_J
    eta_y =  metrics['x_xi'] * inv_J
    
    # Check for Identity: d_xi/dx * dx/d_xi + d_xi/dy * dy/d_xi should equal 1.0
    val_xi = xi_x * metrics['x_xi'] + xi_y * metrics['y_xi']
    val_eta = eta_x * metrics['x_eta'] + eta_y * metrics['y_eta']
    
    # This error should be near zero (e.g., 1e-12)
    error_map = np.abs(val_xi - 1.0) + np.abs(val_eta - 1.0)
    
    print(f"Max Identity Error: {np.max(error_map):.2e}")
    print(f"Mean Identity Error: {np.mean(error_map):.2e}")
    
    return error_map

import numpy as np

def calculate_and_validate_metrics(grid_coords):
    # grid_coords shape: (512, 40, 3)
    ni, nj = grid_coords.shape[0], grid_coords.shape[1]
    
    # 1. Standardize Computational Step (Unit Length)
    # This aligns the derivatives so they are independent of grid density.
    d_xi = 1.0 / (ni - 1)
    d_eta = 1.0 / (nj - 1)
    
    # 2. Forward Metrics (Meters per Unit-Logical-Step)
    # axis 0 is xi (around airfoil), axis 1 is eta (away from airfoil)
    x_xi = np.gradient(grid_coords[:,:,0], axis=0) / d_xi
    x_eta = np.gradient(grid_coords[:,:,0], axis=1) / d_eta
    y_xi = np.gradient(grid_coords[:,:,1], axis=0) / d_xi
    y_eta = np.gradient(grid_coords[:,:,1], axis=1) / d_eta
    
    # 3. Jacobian (Area per Unit-Logical-Area)
    J = x_xi * y_eta - x_eta * y_xi
    
    # 4. Identity Pull-back (The 'True' Validation)
    # These represent (d_xi/dx, d_xi/dy, etc.)
    inv_J = 1.0 / np.clip(np.abs(J), 1e-12, None)
    xi_x, xi_y = y_eta * inv_J, -x_eta * inv_J
    eta_x, eta_y = -y_xi * inv_J, x_xi * inv_J
    
    # Validation: These should equal 1.0 everywhere
    val_xi = xi_x * x_xi + xi_y * y_xi
    val_eta = eta_x * x_eta + eta_y * y_eta
    
    error_map = np.abs(val_xi - 1.0) + np.abs(val_eta - 1.0)
    
    metrics = {'J': J, 'x_xi': x_xi, 'x_eta': x_eta, 'y_xi': y_xi, 'y_eta': y_eta}
    return metrics, error_map

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

        xoffset = -0.5  # Shift airfoil to the right by 1.0 units to avoid singularity at origin
        foil_points_spline = foil_points_spline + np.array([xoffset, 0.0])  # Shift airfoil to the right by xoffset

        foil_points_spline = foil_points_spline*4.0

        print(f" Range of foil points after scaling: x [{foil_points_spline[:,0].min():.2f}, {foil_points_spline[:,0].max():.2f}], y [{foil_points_spline[:,1].min():.2f}, {foil_points_spline[:,1].max():.2f}]")


        
        # 2. Create Mapped Grid


        grid_coords = generate_aligned_o_grid(foil_points_spline, n_layers=40, expansion_ratio=1.02, R_far=8.0)

        print( f"Grid shape: {grid_coords.shape}, x range: [{grid_coords[:,:,0].min():.2f}, {grid_coords[:,:,0].max():.2f}], y range: [{grid_coords[:,:,1].min():.2f}, {grid_coords[:,:,1].max():.2f}]")
        debug_mesh_connectivity(grid_coords, foil_points_spline)    


        # 3. Compute Metrics & Validate 
        metrics, val_xi,val_eta = finalized_metrics_v3(grid_coords)

        plot_grid_metrics(grid_coords, metrics)

        results = solve_potential_flow_final(grid_coords, V_inf=1.0, alpha_deg=0)

        plot_flowfield_pv2(grid_coords, results['x'], results['y'], results['cp'])

        #plot_u_velocity_pv(grid_coords, results['x'], results['y'], results['error'], results['error'])


        # Plot the first few points of the inner boundary (the airfoil)
        # grid_coords shape is (512, 40, 3) -> [xi, eta, xyz]
        plt.figure(figsize=(8,4))
        plt.scatter(grid_coords[:50, 0, 0], grid_coords[:50, 0, 1], c=np.arange(50), cmap='viridis')
        plt.colorbar(label='Index (xi)')
        plt.title("Checking xi direction: Yellow is the direction of increasing index")
        plt.axis('equal')
        plt.show()



if __name__ == "__main__":

    PATH_TO_DATASET = "/home/timm/Projects/PIML/Dataset"
    PATH_TO_OUTPUT = "/home/timm/Projects/PIML/Dataset_Cyl_Pot_FNO_X5Y4"
    GRID_SIZE = (64, 64)
    REGENERATE = True  # Set to True to regenerate all pt files
    xlen = float(6.0)   # domain length to be sampled  (-xlen/2, xlen/2)
    ylen = float(3.0)   # domain height to be sampled  (-ylen/2, ylen/2)
    xoffset = float(1.0)  # x-offset to be sampled

    convert(PATH_TO_DATASET, PATH_TO_OUTPUT, xlen, ylen, xoffset, GRID_SIZE)