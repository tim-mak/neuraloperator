import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import pyvista as pv
from foil_utils import sort_airfoil_to_sellig
import airfrans as af
from scipy.optimize import minimize

def plot_foil(x, y, zeta_real=None,zeta_imag=None, title="Airfoil"):
    """Plot the airfoil coordinates"""
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, marker='o')
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")

    if zeta_real is not None and zeta_imag is not None:
        plt.plot(zeta_real, zeta_imag, marker='x', label='Joukowski Transform', color='r')
        plt.legend()
    plt.axis('equal')
    plt.grid()
    plt.show()

def plot_grid(x,y,zeta_real,zeta_imag):
    """Plot the generated grid points"""
    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(16, 4))
    ax1.scatter(x, y, s=5)
    ax1.set_title("Generated O-grid")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.axis('equal')
    ax1.grid()
    ax2.scatter(zeta_real, zeta_imag, s=5, color='r')
    ax2.set_title("Joukowski Transform")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.axis('equal')
    ax2.grid()
    plt.show()

def joukowski_transform(x, y, c=1.0):
    """Apply the Joukowski transform to the given coordinates"""
    
    z = (x) + 1j * y
    
    # Apply Joukowski transform
    zeta = z + c**2 / z
    
    # Rotate by angle of attack
    #zeta_rotated = zeta * np.exp(-1j * alpha_rad)
    
    return zeta.real, zeta.imag

def karman_trefftz_transform(x, y, c, tau_deg):
    """Maps circle plane z to physical plane zeta with finite TE angle."""
    z = (x) + 1j * y

    tau = np.radians(tau_deg)
    n = 2.0 - tau / np.pi
    
    # Kármán-Trefftz mapping equation
    # ((z-c)/(z+c))^n
    num = z - c
    den = z + c
    
    # Avoid division by zero at the trailing edge focus
    ratio = num / np.where(den == 0, 1e-12, den)
    term = ratio**n
    
    zeta = n * c * (1 + term) / np.where(1 - term == 0, 1e-12, 1 - term)
    return zeta.real,zeta.imag

def generate_circle_points(radius, offset, num_points): 
    """Generate points on a circle for reference"""
    theta = np.linspace(0, 2*np.pi, num_points, endpoint=True)
    x_circle = radius * np.cos(theta) + offset[0]
    y_circle = radius * np.sin(theta) + offset[1]
    return x_circle, y_circle

def generate_o_grid(r_min,r_max,offset, n_radial,n_circumferential):
    """Generate an O-grid around the airfoil"""
    r = np.linspace(r_min, r_max, n_radial)
    theta = np.linspace(0, 2*np.pi, n_circumferential, endpoint=True)
    R, Theta = np.meshgrid(r, theta)
    x = R * np.cos(Theta) + offset[0]
    y = R * np.sin(Theta) + offset[1]
    return x, y

def plot_quiver(x, y, u, v):
    """Plot velocity vectors as quiver plot"""
    plt.figure(figsize=(8, 4))
    plt.quiver(x, y, u, v, scale=1.1, scale_units='xy')
    plt.title("Velocity Field")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')
    plt.grid()
    plt.show()


def plot_streamlines(x, y, w):
    """Plot streamlines using the stream function"""
    plt.figure(figsize=(8, 4))
    plt.contour(x, y, w.imag, levels=20, cmap='viridis')
    plt.title("Streamlines (Stream Function Contours)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')
    plt.grid()
    plt.show()

def pyvista_plot(x, y, u, v, cp):
    """Plot velocity field and pressure using PyVista"""
    grid = pv.StructuredGrid(x, y, np.zeros_like(x))
    grid["Velocity"] = np.stack((u.flatten(order='F'), v.flatten(order='F'), np.zeros_like(u.flatten(order='F'))), axis=-1)
    grid["u"] = u.flatten(order='F')
    grid["v"] = v.flatten(order='F')
    grid["cp"] = cp.flatten(order='F')

    plotter = pv.Plotter()

    grid_cp = grid.copy()
    grid_cp.set_active_scalars("cp")

    grid_vel = grid.copy()
    grid_vel.set_active_scalars("Velocity")

    
    # Add Pressure Contour first
    plotter.add_mesh(grid_cp, scalars="cp", cmap="RdBu_r", clim=[-2, 1.0], lighting=False, show_edges=False)

    # Add Streamlines using velocity mesh
    streamline_seeds = pv.Line(pointa=(-0.0,-2.0,0), pointb=(-0.0,2.0,0), resolution=30)
    streamlines = grid_vel.streamlines_from_source(streamline_seeds,
                                                vectors="Velocity", 
                                                integrator_type=45,
                                                initial_step_length=0.001,
                                                max_step_length=0.01,
                                                max_steps=5000,
                                                integration_direction='both',
                                                  max_time=20)
    plotter.add_mesh(streamlines, color="black", line_width=2)  

    plotter.view_xy()
    plotter.show()

def pyvista_plot_streamline(x, y, u, v, w):
    """Plot velocity field and pressure using PyVista"""
    grid = pv.StructuredGrid(x, y, np.zeros_like(x))
    grid["Velocity"] = np.stack((u.flatten(order='F'), v.flatten(order='F'), np.zeros_like(u.flatten(order='F'))), axis=-1)
    grid["u"] = u.flatten(order='F')
    grid["v"] = v.flatten(order='F')
    grid["s"] = w.imag.flatten(order='F')

    plotter = pv.Plotter()

    grid_cp = grid.copy()
    grid_cp.set_active_scalars("s")

    grid_vel = grid.copy()
    grid_vel.set_active_scalars("Velocity")

    grid_contours = grid.copy()

    contours = grid.contour(scalars="s", isosurfaces=80)
    # Add Pressure Contour first
    plotter.add_mesh(grid_cp, scalars="s", cmap="RdBu_r", clim=[-2, 2.0], lighting=False, show_edges=False)
    plotter.add_mesh(contours, color="black", line_width=2, label="Streamline Contours")


    plotter.view_xy()
    plotter.show()

def calculate_potential_flow(x, y, offset, v_mag,c=1.0, alpha_deg=0):
    """Calculate potential flow around a cylinder using Joukowski transform"""
    z = x + 1j * y
    # Shift and coordinates to account for  offset
    zd = z - (offset[0] + 1j * offset[1])
    
    # Calculate velocity components in the z-plane
    alpha_rad = np.deg2rad(alpha_deg)
    
    # Calculate the TE Beta for the trailing edge of the airfoil
    R = np.abs(c -complex(offset[0], offset[1]))   
    
    beta = np.arcsin(offset[1] / R)  # Angle from center to TE
    print(f"Calculated TE beta angle: {np.rad2deg(beta):.2f} degrees")
    alpha_eff = alpha_rad + beta  # Effective angle of attack at the TE)

    print(f"Effective angle of attack at TE (alpha_eff): {np.rad2deg(alpha_eff):.2f} degrees")

    # circulation gamma for lift (Kutta condition)
    gamma = 4 * np.pi * c * v_mag * np.sin(alpha_eff)
    # Stream function

    w = v_mag * (zd * np.exp(-1j * alpha_rad) + (c**2 * np.exp(1j * alpha_rad)) / zd) + 1j * (gamma / (2 * np.pi)) * np.log(zd / c)

    # Compute velocity in z-plane
    dw_dz = v_mag * (np.exp(-1j * alpha_rad) - (c**2 * np.exp(1j * alpha_rad)) / zd**2   + (1j*gamma) / (2 * np.pi * zd))


    z_complex = np.conj(dw_dz)
    uz = z_complex.real
    vz = z_complex.imag 
    cpz = 1.0 - (np.abs(z_complex) / v_mag)**2

    #u = u*np.cos(alpha_rad) + v*np.sin(alpha_rad)
    #v = -u*np.sin(alpha_rad) + v*np.cos(alpha_rad)
    return uz, vz, cpz, w


def calculate_potential_flow_joukowski(x, y, offset, v_mag,c=1.0, alpha_deg=0):
    """Calculate potential flow around a cylinder using Joukowski transform"""
    z = x + 1j * y
    # Shift and coordinates to account for  offset
    zd = z - (offset[0] + 1j * offset[1])


    # Calculate velocity components in the z-plane
    alpha_rad = np.deg2rad(alpha_deg)

    # Calculate the TE Beta for the trailing edge of the airfoil
    R = np.abs(c -complex(offset[0], offset[1]))   
    
    beta = np.arcsin(offset[1] / R)  # Angle from center to TE
    print(f"Calculated TE beta angle: {np.rad2deg(beta):.2f} degrees")
    alpha_eff = alpha_rad + beta  # Effective angle of attack at the TE)

    print(f"Effective angle of attack at TE (alpha_eff): {np.rad2deg(alpha_eff):.2f} degrees")

    # circulation gamma for lift (Kutta condition)
    gamma = 4 * np.pi * c * v_mag * np.sin(alpha_eff)
    # Stream function

    w = v_mag * (zd * np.exp(-1j * alpha_eff) + (c**2 * np.exp(1j * alpha_eff)) / zd) + 1j * (gamma / (2 * np.pi)) * np.log(zd / c)

    # Compute velocity in z-plane
    dw_dz = v_mag * (np.exp(-1j * alpha_eff) - (c**2 * np.exp(1j * alpha_eff)) / zd**2   + (1j*gamma) / (2 * np.pi * zd))

    # Transform velocity to physical plane using Joukowski transform
    dzeta_dz = 1 - (c**2) / zd**2
    dw_dzeta = dw_dz / dzeta_dz

    z_complex = np.conj(dw_dzeta)
    uz = z_complex.real
    vz = z_complex.imag 
    cpz = 1.0 - (np.abs(z_complex) / v_mag)**2

    return uz, vz, cpz, w

def calculate_trailing_edge_angle(offset, c=1.0):
        # Calculate the TE Beta for the trailing edge of the airfoil
    R = np.abs(c -complex(offset[0], offset[1]))   
    
    beta = np.arcsin(offset[1] / R)  # Angle from center to TE
    print(f"Calculated TE beta angle: {np.rad2deg(beta):.2f} degrees")
    return np.rad2deg(beta)

def calculate_potential_flow_karman_trefftz(x, y, offset, v_mag,c=1.0, alpha_deg=0, alpha_te_deg=0  ):
    """Calculate potential flow around a cylinder using Karman-Trefftz transform"""

    tau = np.rad2deg(alpha_te_deg)  # TE angle in degrees

    n = 2.0 - tau / np.pi # Karman-Trefftz exponent

    z = x + 1j * y
    # Shift and coordinates to account for  offset
    zd = z - (offset[0] + 1j * offset[1])


    # Calculate velocity components in the z-plane
    alpha_rad = np.deg2rad(alpha_deg)


    beta_deg = calculate_trailing_edge_angle(offset, c)

    R = np.abs(c -complex(offset[0], offset[1]))

    print(f"Calculated TE beta angle: {beta_deg:.2f} degrees")
    alpha_eff = alpha_rad + np.deg2rad(beta_deg)  # Effective angle of attack at the TE)

    print(f"Effective angle of attack at TE (alpha_eff): {np.rad2deg(alpha_eff):.2f} degrees")

    # circulation gamma for lift (Kutta condition)
    gamma = 4 * np.pi * c * v_mag * np.sin(alpha_eff)
    # Stream function

    w = v_mag * (zd * np.exp(-1j * alpha_eff) + (c**2 * np.exp(1j * alpha_eff)) / zd) + 1j * (gamma / (2 * np.pi)) * np.log(zd / c)
    
    dw_dz = v_mag * (np.exp(-1j * alpha_rad) - (R**2 * np.exp(1j * alpha_rad)) / zd**2) + (1j * gamma) / (2 * np.pi * zd)
    # 2. Intermediate terms to prevent redundant calculation
    # We use z and the focal points +/- c
    z_minus_c = zd - c
    z_plus_c  = zd + c
    # Transform velocity to physical plane using Karman-Trefftz transform
    ratio = z_minus_c / np.where(z_plus_c == 0, 1e-12, z_plus_c)
    term = ratio**n
    # 3. The Derivative Formula
    # dzeta/dz = (4 * n^2 * c^2 / (z^2 - c^2)) * (term / (1 - term)^2)
    numerator = 4.0 * (n**2) * (c**2) * term
    denominator = (z**2 - c**2) * (1.0 - term)**2
    
    # Safety clip for the denominator to prevent division by zero at focal points
    dzeta_dz = numerator / np.where(np.abs(denominator) < 1e-15, 1e-15, denominator)
    
    # dw_dzeta = dw/dz * dz/dzeta = dw/dz / (dzeta/dz)
    dw_dzeta = dw_dz / dzeta_dz
    z_complex = np.conj(dw_dzeta)
    uz = z_complex.real
    vz = z_complex.imag 
    cpz = 1.0 - (np.abs(z_complex) / v_mag)**2

    return uz, vz, cpz, w

def fit_joukowski(target_x, target_y):
    # Initial guess: small thickness, zero camber
    # p[0] = mu (thickness), p[1] = eta (camber)
    initial_guess = [-0.1, 0.05]
    
    def objective(p):
        mu, eta = p[0], p[1]
        c = 1.0
        z0 = complex(mu, eta)
        R = np.abs(c - z0)
        
        # Generate Joukowski coordinates
        theta = np.linspace(0, 2*np.pi, len(target_x))
        z = z0 + R * np.exp(1j * theta)
        zeta = z + c**2 / z
        
        # Calculate distance error to target points
        dist = np.sqrt((zeta.real - target_x)**2 + (zeta.imag - target_y)**2)
        
        # Penalty for 'exploding' or self-intersection
        # mu should stay negative and eta shouldn't be extreme
        penalty = 0
        if mu >= -1e-3: penalty += 1e5
        
        return np.mean(dist) + penalty

    res = minimize(objective, initial_guess, method='Nelder-Mead')

    R = np.abs(1.0 - complex(res.x[0], res.x[1]))
    print(f"Fitted parameters: mu={res.x[0]:.4f}, eta={res.x[1]:.4f}, R={R:.4f}")
    return res.x, R # Returns [best_mu, best_eta], R



def fit_karman_trefftz(target_x, target_y):
    # p[0]: mu (thickness), p[1]: eta (camber), p[2]: tau_deg (TE angle)
    initial_guess = [-0.1, 0.05, 10.0]
    c = 1.0 # Focal point remains fixed
    
    def objective(p):
        mu, eta, tau_deg = p
        z0 = complex(mu, eta)
        R = np.abs(c - z0) # Radius pinned to TE
        
        # Generate Circle Plane coordinates
        theta = np.linspace(0, 2*np.pi, len(target_x))
        z = z0 + R * np.exp(1j * theta)
        
        # Transform to Kármán-Trefftz plane
        zeta_real, zeta_imag = karman_trefftz_transform(z.real, z.imag, c, tau_deg)
        
        # Error calculation (MSE)
        error = np.mean(np.sqrt((zeta_real - target_x)**2 + (zeta_imag - target_y)**2))
        
        # Hard Constraints to prevent "Exploding"
        penalty = 0
        if mu >= -0.001: penalty += 1e5   # Must have thickness
        if tau_deg < 0.1 or tau_deg > 25: penalty += 1e5 # Sensible TE angles
        
        return error + penalty

    res = minimize(objective, initial_guess, method='Nelder-Mead', tol=1e-6)
    R = np.abs(1.0 - complex(res.x[0], res.x[1]))
    print(f"Fitted parameters: mu={res.x[0]:.4f}, eta={res.x[1]:.4f}, tau_deg={res.x[2]:.2f}, R={R:.4f}")

    return res.x , R # Returns [best_mu, best_eta, best_tau]

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
        x_foil = sorted_foil_points[:, 0]
        y_foil = sorted_foil_points[:, 1]
        # Get TE point for reference
        i_te = np.argmax(x_foil)  # Assuming TE is at maximum x coordinate
        i_le = np.argmin(x_foil)  # Assuming LE is at minimum x coordinate
        x_te = x_foil[i_te] # Assuming TE is at minimum x coordinate]
        y_te = y_foil[i_te]
        # get the LE point for reference
        x_le = x_foil[i_le]  # Assuming LE is at maximum x coordinate
        y_le = y_foil[i_le]
        x_offset = (x_le + x_te) / 2   # Center the transform around the midpoint of LE and TE, then apply offset
        print(f"LE point before transform: ({x_le:.6f}, {y_le:.6f})")
        print(f"TE point before transform: ({x_te:.6f}, {y_te:.6f})")
        x_foil = x_foil - x_offset  # Shift coordinates to center around midpoint of LE and TE
        print(f"x_offset: {x_offset:.6f}")
        scale = 4.0 / (x_te - x_le)  # Scale to ensure the transformed airfoil has a chord length of 2

        x_foil = x_foil * scale
        y_foil = y_foil * scale

        print(f"LE point after transform: ({x_foil[i_le]:.6f}, {y_foil[i_le]:.6f})")
        print(f"TE point after transform: ({x_foil[i_te]:.6f}, {y_foil[i_te]:.6f})")


        (best_mu, best_eta), R = fit_joukowski(x_foil, y_foil)

        print(f"Fitted Joukowski parameters: mu={best_mu}, eta={best_eta}, R={R}")


        offset = [best_mu, best_eta]
        x_circle, y_circle = generate_circle_points(radius=R, offset=offset, num_points=len(x_foil))
        zeta_real, zeta_imag = joukowski_transform(x_circle, y_circle, c=1.0)
        #zeta_real, zeta_imag, x_offset = joukowski_transform(x_foil, y_foil, c=1.0)
        plot_foil(x_foil, y_foil, zeta_real, zeta_imag, title=f"Airfoil and Joukowski Transform (AoA={aoa_deg}°)")


        tau_te_deg = 5 # Use the same function to get TE angle for Karman-Trefftz fit

        (best_mu, best_eta, best_tau), R = fit_karman_trefftz( x_foil, y_foil)


        print(f"Fitted Karman-Trefftz parameters: mu={best_mu}, eta={best_eta}, tau={best_tau}, R={R}")

        offset = [best_mu, best_eta]
        
        x_circle, y_circle = generate_circle_points(radius=R, offset=offset, num_points=len(x_foil))

        beta_deg = calculate_trailing_edge_angle(offset, c=1.0)

        zeta_real, zeta_imag = karman_trefftz_transform(x_circle, y_circle, c=1.0, tau_deg=beta_deg)

        plot_foil(x_foil, y_foil, zeta_real, zeta_imag, title=f"Airfoil and Karman-Trefftz Transform (AoA={aoa_deg}°)")


        # 2. Generate O-grid around the airfoil 
        n_radial, n_circumferential = (100,145)

        x_grid, y_grid = generate_o_grid(r_min=R, r_max=5.0, offset=offset, n_radial=n_radial, n_circumferential=n_circumferential)

        #zeta_grid_real, zeta_grid_imag = joukowski_transform(x_grid, y_grid, c=1.0)
        zeta_grid_real, zeta_grid_imag = karman_trefftz_transform(x_grid, y_grid, c=1.0, tau_deg=beta_deg)


        #plot_grid(x_grid, y_grid, zeta_grid_real, zeta_grid_imag)

        # 3. Calculate potential flow around the airfoil using Joukowski transform
        v_inf = 1.0
        
        u_grid, v_grid, cp_grid, w_grid = calculate_potential_flow(x_grid, y_grid, offset, v_inf, c=1.0, alpha_deg=aoa_deg)
        print(f"U shape: {u_grid.shape},  V shape: {v_grid.shape}, Psi shape: {w_grid.shape}")
        print(f"X : {x_grid.shape},  Y : {y_grid.shape}")
        plot_streamlines(x_grid, y_grid, w=w_grid)

        pyvista_plot(x_grid, y_grid, u_grid, v_grid, cp=cp_grid)

        u_foil, v_foil, cp_foil, w_foil = calculate_potential_flow_karman_trefftz(x_grid, y_grid, offset, v_inf, c=1.0, alpha_deg=aoa_deg)

        print(f"U shape: {u_foil.shape},  V shape: {v_foil.shape}, Psi shape: {w_foil.shape}")
        print(f"Zeta real: {zeta_grid_real.shape},  Zeta imag: {zeta_grid_imag.shape}")
        plot_streamlines(zeta_grid_real, zeta_grid_imag, w=w_foil)

        pyvista_plot(zeta_grid_real, zeta_grid_imag, u_foil, v_foil, cp=cp_foil)

        pyvista_plot_streamline(zeta_grid_real, zeta_grid_imag, u_foil, v_foil, w_foil)





if __name__ == "__main__":

    PATH_TO_DATASET = "/home/timm/Projects/PIML/Dataset"
    PATH_TO_OUTPUT = "/home/timm/Projects/PIML/Dataset_Cyl_Pot_FNO_X5Y4"
    GRID_SIZE = (64, 64)
    REGENERATE = True  # Set to True to regenerate all pt files
    xlen = float(6.0)   # domain length to be sampled  (-xlen/2, xlen/2)
    ylen = float(3.0)   # domain height to be sampled  (-ylen/2, ylen/2)
    xoffset = float(1.0)  # x-offset to be sampled

    convert(PATH_TO_DATASET, PATH_TO_OUTPUT, xlen, ylen, xoffset, GRID_SIZE)