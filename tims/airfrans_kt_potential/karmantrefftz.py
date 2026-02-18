from multiprocessing.util import debug
from matplotlib.pylab import gamma
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import airfrans as af
from airfoil_utils import BSplineFoil
from scipy.interpolate import griddata, interp1d, splprep, splev
import pyvista as pv

from scipy.spatial import cKDTree
import numpy as np

class KarmanTrefftzAirfoil:
    def __init__(self, mux=0.1, muy=-0.08, tau_deg=0):
        self.mux = mux  # Maximum camber
        self.muy = muy  # Location of maximum camber
        self.k = 2 - tau_deg/np.pi # TE position
        self.b = 1.0
        self.tau_deg =tau_deg  # Trailing edge angle in degrees (0 for cusped, >0 for blunt TE)
        self.R = abs(complex(mux, muy) - self.k)  # Radius of the circle in zeta-plane

        print(f"Initialized Karman-Trefftz Airfoil with mux={mux}, muy={muy},TE deg{tau_deg}°, k={self.k}, R={self.R:.4f}")

    def set_params(self, mux, muy, tau_deg=0):
        """Set new parameters for the Karman-Trefftz airfoil"""
        self.mux = mux
        self.muy = muy
        self.tau_deg = tau_deg
        self.k = 2 - tau_deg/np.pi  # Update k based on new tau
        self.R = abs(complex(mux, muy) - self.k)
        print(f"Updated Karman-Trefftz Airfoil parameters: mux={mux}, muy={muy}, tau={tau_deg}°, k={self.k}, R={self.R:.4f}")


    def apply_kalman_transform(self, zeta):
        """Apply the Karman-Trefftz transformation to the complex coordinate zeta
        given in the zeta-plane, returning the physical coordinate in the z-plane.
        tau_deg: Trailing edge angle in degrees (0 for cusped, >0 for blunt TE)
        """
        # Step 1: Center of circle in zeta-plane
        mu = self.mux + 1j * self.muy

        # Radius from mu to TE (b) is 1.0 in the z-plane
        R = abs(mu - self.b)

        # Trailing edge angle adjustment (tau) modifies the exponent n
        tau_rad = np.deg2rad(self.tau_deg)

        # k slightly less than 2 give a fixed trailing edge, k=2 is a cusped TE, k<2 gives a blunt TE
        self.k = 2.0 - tau_rad / np.pi  # Adjust k based on TE angle

        print(f"Using k={self.k:.4f} for tau={self.tau_deg} degrees")
        num =(zeta + self.b)**self.k + ( zeta -self.b)**self.k
        den = (zeta + self.b)**self.k - ( zeta - self.b)**self.k

        ratio = num / np.where(den == 0, 1e-10, den)  # Avoid division by zero
        z = self.b * self.k*ratio

        return z

    def apply_joukowski_transform(self, zeta): 
        """Apply the Joukowski transformation to the complex coordinate zeta"""

        return zeta + self.k**2 / zeta 



    def generate_circle_points(self, num_points=101):

        """Generate points on the unit circle in the zeta-plane"""

        self.R = abs(complex(self.mux, self.muy) - self.k)  # Radius of the circle in zeta-plane
        if self.R <= 1.0:
            print(f"Warning: R={self.R:.4f} is not greater than 1.0. Consider adjusting mux and muy for a valid Karman-Trefftz airfoil.")

        theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        zeta = self.R * np.exp(1j * theta)

        # Points on the unit circle 
        zeta.real = zeta.real + self.mux 
        # Shift to center at (mux, muy)
        zeta.imag = zeta.imag + self.muy


        return zeta
    
    def get_trailing_edge_theta(self):
        zeta_ctr = self.mux + 1j*self.muy
        zeta_te = np.angle(self.b - zeta_ctr, deg=False)
        
        print(f" TE theta {np.rad2deg(zeta_te)} deg ")

        return zeta_te

    def get_leading_edge_theta(self):
        zeta_ctr = self.mux + 1j * self.muy
        # Vector from center to the LE singularity (-b)
        # This is the correct definition for "Point E" in the Blom report
        zeta_le_angle = np.angle(-self.b - zeta_ctr, deg=False)
        
        # Wrap to 0-2pi or -pi to pi range consistently
        zeta_le_angle = np.mod(zeta_le_angle, 2 * np.pi)
        
        print(f" LE theta (Point E): {np.rad2deg(zeta_le_angle):.4f} deg ")
        return zeta_le_angle
    
    def generate_circle_points_at_theta(self,theta):
        """ Generate points on the unit circle from 0-pi in zeta-plane"""   
        # Radius of the circle in zeta-plane
        self.R = abs(complex(self.mux, self.muy) - self.b) 
        if self.R <= 1.0:
            print(f"Warning: R={self.R:.4f} is not greater than 1.0. Consider adjusting mux and muy for a valid Karman-Trefftz airfoil.")

        #Generate points at origin
        zeta = self.R * np.exp(1j * theta)
        # shift to center
        zeta +=complex(self.mux,self.muy)

        return zeta    
    
    def get_kt_normals(self, theta):
        """
        Calculates analytical normal vectors for K-T profile.
        theta: array of angles in radians.
        """
        zeta = self.mux + 1j*self.muy + self.R * np.exp(1j * theta)
        
        # Components for the derivative
        tp = (zeta + self.b)**(self.k - 1)
        tm = (zeta - self.b)**(self.k - 1)
        
        # Power terms for the denominator
        tpk = (zeta + self.b)**self.k
        tmk = (zeta - self.b)**self.k
        
        # The derivative dz/dzeta
        num = 4 * (self.k**2) * (self.b**2) * tp * tm
        den = (tpk - tmk)**2
        
        dz_dzeta = num / den
        
        # Radial vector in zeta plane
        radial_zeta = np.exp(1j * theta)
        
        # Normal vector in z plane (unnormalized)
        n_z = dz_dzeta * radial_zeta
        
        # Return as normalized x and y components
        nx = n_z.real / np.abs(n_z)
        ny = n_z.imag / np.abs(n_z)
        
        return np.column_stack([nx, ny])

    def plot_airfoil(self, num_points=101): 
        """Generate and plot the airfoil shape""" 

        zeta = self.generate_circle_points( num_points)

        z = self.apply_kalman_transform(zeta)

        fig,(ax1,ax2) = plt.subplots(1,2, figsize=(12,6))
        ax1.plot(zeta.real, zeta.imag, 'b-', label='Unit Circle in Zeta-plane') 
        ax1.axis('equal')
        ax1.set_title(f'Unit Circle in Zeta-plane (mux={self.mux}, muy={self.muy}, tau={self.tau_deg}°)') 
        ax1.set_xlabel('x')
        ax1.set_ylabel('y') 
        ax1.grid() 
        ax1.legend()
        ax2.plot(z.real, z.imag, 'b-', label='Karman-Trefftz Airfoil') 
        ax2.axis('equal')
        ax2.set_title(f'Karman-Trefftz Airfoil (mux={self.mux}, muy={self.muy}, tau={self.tau_deg}°)') 
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.grid()
        ax2.legend()
        plt.show()
    
    def calculate_chord_length(self, z_raw):
            # 2. Find the raw K-T span
            kt_min_x, kt_max_x = np.min(z_raw.real), np.max(z_raw.real)
            kt_raw_chord = kt_max_x - kt_min_x
            
            return kt_raw_chord,kt_min_x,kt_max_x
    def set_fitted_scale(self, scale):
        self.fitted_scale = scale

    def set_fitted_chord(self, chord):
        self.fitted_chord = chord
    def set_fitted_te(self, te):
        self.fitted_te = te

    def find_best_fit(self, x_foil, y_foil, num_points=101):
        """Find the best fit Karman-Trefftz parameters for the given airfoil coordinates"""
        from scipy.optimize import minimize

        def objective(params):
            mux, muy , tau_deg= params

            self.set_params(mux, muy, tau_deg)
            zeta = self.generate_circle_points( num_points)

            z_raw = self.apply_kalman_transform(zeta)

            #calculate chord length and scale to match the original airfoil chord
            kt_raw_chord, kt_min_x, kt_max_x = self.calculate_chord_length(z_raw)

            # 3. Calculate Target Chord
            target_min_x, target_max_x = np.min(x_foil), np.max(x_foil)
            target_chord = target_max_x - target_min_x
            
            # 4. SCALE THE K-T FOIL TO MATCH TARGET CHORD
            # This removes chord length as a variable the optimizer has to "find"
            scale_to_target = target_chord / kt_raw_chord
            z_final = (z_raw - kt_max_x) * scale_to_target + target_max_x
            print(f" Scaling K-T foil by {scale_to_target:.4f} to match target chord length {target_chord:.4f}")
            # Interpolate z to the x_foil points and compute distance
            self.set_fitted_scale(scale_to_target)
            self.set_fitted_chord(kt_raw_chord*scale_to_target)
            self.set_fitted_te(kt_max_x)

            # get interpolated points on the airfoil for upper and lower surface
            z_upper = z_final[zeta.imag >= 0]
            z_lower = z_final[zeta.imag < 0]

            le_index = np.argmin(x_foil)
            y_upper = y_foil[:le_index]
            y_lower = y_foil[le_index:]

            # Need to sort upper surface point so that x is ascending
            x_foil_upper_sorted_indices = np.argsort(x_foil[:le_index])
            z_upper_sorted_indices = np.argsort(z_upper.real)

            z_upper_interp = np.interp(x_foil[x_foil_upper_sorted_indices], z_upper.real[z_upper_sorted_indices], z_upper.imag[z_upper_sorted_indices])
            z_upper_interp = np.flip(z_upper_interp)  # Flip to match the order of y_upper
            z_lower_interp = np.interp(x_foil[le_index:], z_lower.real, z_lower.imag)

            y_dist_upper = np.sqrt((z_upper_interp - y_upper)**2 )
            y_dist_lower = np.sqrt((z_lower_interp - y_lower)**2 )

            y_dist_mean = np.mean(np.concatenate([y_dist_upper, y_dist_lower]))

            debug = False
            if debug:
                plt.figure(figsize=(10, 6))
                plt.plot(x_foil[:le_index], z_upper_interp, 'bo-', label='Karman-Trefftz- Upper')
                plt.plot(x_foil[:le_index], y_upper, 'ro-', label='Original Airfoil - Upper', markersize=4)
                plt.plot(x_foil[le_index:], z_lower_interp, 'g-', label='Karman-Trefftz - Lower')
                plt.plot(x_foil[le_index:], y_lower , 'mo-', label='Original Airfoil - Lower', markersize=4)
                        
                plt.axis('equal')
                plt.title(f'Fit with mux={mux:.4f}, muy={muy:.4f}, tau={tau_deg:.2f}°')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.grid()
                plt.legend()
                plt.show()
                     

            print(f"Testing params: mux={mux:.4f}, muy={muy:.4f}, tau={tau_deg:.2f}°, y_dist_mean={y_dist_mean:.2f}, scale to target={scale_to_target:.4f}, kt_raw_chord={kt_raw_chord:.4f}, target_chord={target_chord:.4f}")
            return y_dist_mean
        
        tau_deg = 10.0  # Initial guess for trailing edge angle
        initial_guess = [-0.08, 0.02, tau_deg]
        result = minimize(objective, initial_guess, bounds=[(-0.5, -0.01), (-0.2, 0.2), (0, 20)], method='Nelder-Mead')
        best_mux, best_muy,best_tau_deg = result.x
        print(f"Best fit parameters: mux={best_mux:.4f}, muy={best_muy:.4f} tau={best_tau_deg:.2f}°")
        self.mux = best_mux
        self.muy = best_muy
        self.k = 2 - best_tau_deg / 180.0
        self.R = abs(complex(self.mux, self.muy) - self.k)
        self.tau_deg = best_tau_deg

        self.set_params(best_mux, best_muy, best_tau_deg)
        return best_mux, best_muy, best_tau_deg
    
    def generate_circle_grid(self,  n_radial=50 , n_theta=50, dr_foil = 0.01,dr_ratio=1.1):
        """Generate a grid in the zeta-plane that can be transformed to the physical plane"""

        # Inner radius is R to ensure we are outside the singularity at the center of the circle in the zeta-plane
        r_min = self.R 

        # get TE and LE theta for the circle grid
        theta_te = self.get_trailing_edge_theta()

        theta = np.linspace(theta_te, theta_te + 2*np.pi, n_theta, endpoint=True)
        theta = np.unwrap(theta)  # Unwrap to ensure continuity

        r = self.generate_radial_dist(r_min, dr_foil, dr_ratio, n_radial)


        R, Theta = np.meshgrid(r, theta)
        zeta = R * np.exp(1j * Theta) + complex(self.mux, self.muy)  # Shift to center at (mux, muy)
        return zeta
        
    def generate_radial_dist(self, r_min, dr_foil, dr_ratio, n_radial):
        """
        Generates radial points starting at r_min, with a strictly 
        controlled first cell height and geometric stretch ratio.
        """
        # 1. Create an array of exponents: [0, 1, 2, ..., n_layers-1]
        exponents = np.arange(n_radial - 1)
        
        # 2. Calculate the exact thickness of every individual cell layer
        # dr_array = [dr, dr*r, dr*r^2, dr*r^3 ...]
        dr_array = dr_foil * (dr_ratio ** exponents)
        
        # 3. Cumulatively sum the thicknesses to get the absolute node positions
        # We prepend a 0 because the first node sits exactly on the airfoil surface
        relative_positions = np.concatenate(([0.0], np.cumsum(dr_array)))
        
        # 4. Shift the entire array out to the surface radius
        r = r_min + relative_positions
        
        return r
    
    def generate_conformal_grid(self,  n_radial=50, n_theta=50, dr_foil = 0.01,dr_ratio=1.1):
        """Generate a conformal grid in the zeta-plane that can be transformed to the physical plane"""
        
        
        # Generate points on the airfoil surface (zeta-plane)
        zeta_surface = self.generate_circle_grid(n_radial=n_radial, n_theta=n_theta, dr_foil = dr_foil,dr_ratio=dr_ratio)
        
        # Apply Karman-Trefftz transformation to get physical coordinates
        z_grid = self.apply_kalman_transform(zeta_surface)

        return z_grid, zeta_surface
    
    def calculate_potential_flow(self, zeta,  v_mag,  alpha_deg=0  ):
        """Calculate potential flow around a cylinder using Karman-Trefftz transform"""

        n = self.k

        # Calculate velocity components in the z-plane
        alpha_rad = np.deg2rad(alpha_deg)

        theta_te =  self.get_trailing_edge_theta()

        # calculate z_prime to centre doublet term
        zeta_prime = zeta - complex(self.mux, self.muy)

        print(f"Calculated TE beta angle: {theta_te:.2f} degrees")
        alpha_eff = alpha_rad + np.deg2rad(theta_te)  # Effective angle of attack at the TE)

        print(f"Effective angle of attack at TE (alpha_eff): {np.rad2deg(alpha_eff):.2f} degrees")

        # CIRCULATION FIX: Use self.R instead of self.b
        gamma = 4 * np.pi * self.R * v_mag * np.sin(alpha_eff)

        # STREAM FUNCTION FIX: Use alpha_rad for freestream, self.R**2 for doublet
        w = v_mag * (zeta_prime * np.exp(-1j * alpha_rad) + (self.R**2 * np.exp(1j * alpha_rad)) / zeta_prime) + 1j * (gamma / (2 * np.pi)) * np.log(zeta_prime / self.R)
        
        # COMPLEX VELOCITY FIX: Note your variable 'dw_dz' is mathematically dW/dzeta'
        dw_dzeta = v_mag * (np.exp(-1j * alpha_rad) - (self.R**2 * np.exp(1j * alpha_rad)) / zeta_prime**2) + (1j * gamma) / (2 * np.pi * zeta_prime)

        # 2. Intermediate terms 
        # Correct: We use the unshifted zeta and the focal points +/- b
        z_minus_b = zeta - self.b
        z_plus_b  = zeta + self.b
        
        ratio = z_minus_b / np.where(z_plus_b == 0, 1e-12, z_plus_b)
        term = ratio**n
        
        # 3. The Derivative Formula
        # Note: mathematically this evaluates to dz/dzeta, not dzeta/dz
        numerator = 4.0 * (n**2) * (self.b**2) * term
        denominator = (zeta**2 - self.b**2) * (1.0 - term)**2
        
        # Safety clip
        dz_dzeta = numerator / np.where(np.abs(denominator) < 1e-15, 1e-15, denominator)
        
        # Physical Velocity: dW/dz = (dW/dzeta) / (dz/dzeta)
        dw_dz = dw_dzeta / dz_dzeta
        
        z_complex = np.conj(dw_dz)
        uz = z_complex.real
        vz = z_complex.imag 
        cpz = 1.0 - (np.abs(z_complex) / v_mag)**2

        return uz, vz, cpz, w



    def pyvista_mesh(self, zeta):
        """Create a PyVista mesh from the given coordinates"""

        grid = pv.StructuredGrid(zeta.real, zeta.imag, np.zeros_like(zeta.real))
        plotter = pv.Plotter()
        plotter.add_mesh(grid, color="lightblue", show_edges=True)
        plotter.view_xy()
        plotter.show()  
    
    def pyvista_overlay_mesh(self, zeta1, zeta2):
        """Create a PyVista mesh from the given coordinates"""

        grid1 = pv.StructuredGrid(zeta1.real, zeta1.imag, np.zeros_like(zeta1.real))
        grid2 = pv.StructuredGrid(zeta2.real, zeta2.imag, np.zeros_like(zeta2.real))
        plotter = pv.Plotter()
        plotter.add_mesh(grid1, style='wireframe', color="blue", opacity=1.0,line_width=2)
        plotter.add_mesh(grid2, style='wireframe', color="red", opacity=1.0, line_width=2)
        plotter.view_xy()
        plotter.show()  

    def algebraic_grid_blend(self, zeta_kt, z_cst_snap, decay_power=2.0):
        """
        Blends the snapped wall deformation into the interior K-T grid algebraically.
        zeta_kt: The original, perfect analytical K-T grid (complex array, shape N_xi, N_eta)
        z_cst_snap: The snapped surface coordinates (complex array, shape N_xi)
        """
        N_xi, N_eta = zeta_kt.shape
        zeta_new = zeta_kt.copy()
        
        # 1. Calculate the exact deformation vector at the wall (j=0)
        # This is the difference between your new CST boundary and the old K-T boundary
        wall_deformation = z_cst_snap - zeta_kt[:, 0]
        
        # 2. Create a decay curve from 1.0 (at the wall) to 0.0 (at the farfield)
        # j_indices runs from 0 to N_eta - 1
        j_indices = np.arange(N_eta)
        
        # decay_power controls how fast the deformation vanishes. 
        # 1.0 = linear decay. 2.0 or 3.0 = deformation mostly stays near the foil.
        decay = (1.0 - j_indices / (N_eta - 1)) ** decay_power
        
        # 3. Apply the decaying deformation to every point in the grid
        # Reshaping decay to (1, N_eta) allows NumPy to broadcast it across all xi angles
        zeta_new = zeta_kt + wall_deformation[:, np.newaxis] * decay[np.newaxis, :]
        
        return zeta_new


    def calculate_grid_sdf(self,X_grid, Y_grid):
        """
        Calculates the SDF using the grid's own surface nodes.
        Guarantees SDF = 0.0 at j=0.
        """
        # 1. Extract the grid's surface points (j=0)
        surface_points = np.column_stack([X_grid[:, 0], Y_grid[:, 0]])
        
        # 2. Build the KD-Tree from the grid surface
        tree = cKDTree(surface_points)
        
        # 3. Flatten the full grid to query it
        grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
        
        # 4. Query the tree
        distances, _ = tree.query(grid_points, k=1)
        
        # 5. Reshape back to (N_xi, N_eta)
        sdf_array = distances.reshape(X_grid.shape)
        
        return sdf_array

def plot_geometry_comparison(x_foil, y_foil, x2_foil, y2_foil):

    """Plot the original airfoil coordinates and the best fit Karman-Trefftz airfoil for comparison"""

    plt.figure(figsize=(20, 16))
    plt.plot(x_foil, y_foil, 'ro-', label='Original Airfoil', markersize=4)
    plt.plot(x2_foil, y2_foil , 'b-', label='Best Fit Karman-Trefftz Airfoil')
    plt.axis('equal')
    plt.title('Geometry Comparison')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.legend()
    plt.show()

def pyvista_plot(x, y, u, v, cp, w, scalar_min=-1.0, scalar_max=1.0):
    """Plot velocity field and pressure using PyVista"""
    grid = pv.StructuredGrid(x, y, np.zeros_like(x))
    grid["Velocity"] = np.stack((u.flatten(order='F'), v.flatten(order='F'), np.zeros_like(u.flatten(order='F'))), axis=-1)
    grid["u"] = u.flatten(order='F')
    grid["v"] = v.flatten(order='F')
    grid["cp"] = cp.flatten(order='F')
    grid["w"] = w.flatten(order='F')
    grid["phi"] = w.real.flatten(order='F')
    grid["s"] = w.imag.flatten(order='F')

    plotter = pv.Plotter()

    grid_cp = grid.copy()
    grid_cp.set_active_scalars("cp")

    grid_vel = grid.copy()
    grid_vel.set_active_scalars("Velocity")

    grid_s = grid.copy()
    grid_s.set_active_scalars("s")


    # Add Pressure Contour first
    
    # Add Pressure Contour first
    plotter.add_mesh(grid_cp, scalars="cp", cmap="RdBu_r", clim=[scalar_min, scalar_max], lighting=False, edge_opacity=0.5, show_edges=True)

    contours = grid_s.contour(scalars="s", isosurfaces=80)
    plotter.add_mesh(contours, color="black", line_width=2, label="Streamline Contours")


    plotter.view_xy()
    plotter.show()

def pyvista_plot_airfrans_scalar(grid, scalar_name="Scalar", scalar_min=-1.0, scalar_max=1.0):

    # 1. Attach the calculated array back to the PyVista grid
    # .flatten() ensures it matches the (n_points,) shape PyVista expects
    #grid.point_data[scalar_name] = scalar_field.flatten()


    # 2. Initialize the plotter
    plotter = pv.Plotter()

    # 3. Add the grid to the plotter
    # 'coolwarm' or 'RdBu_r' are excellent colormaps for pressure (Cp) 
    # because they clearly highlight the stagnation (high) and suction (low) peaks.
    plotter.add_mesh(
        grid, 
        scalars=scalar_name, 
        cmap='coolwarm',     
        show_edges=True, 
        edge_opacity=0.5,  
        clim =[scalar_min, scalar_max],
        scalar_bar_args={'title': f'{scalar_name}'}
    )

    # 4. Set the camera to look straight down the Z-axis for 2D flow
    plotter.view_xy()

    # 5. Render the plot
    plotter.show()

def convert(dataset_root, output_folder, xlen, ylen, xoffset, grid_size):

    archive_dir = Path(output_folder) / "Archive"
    training_dir = Path(output_folder) / "TrainingX5Y4"
    Path(archive_dir).mkdir(parents=True, exist_ok=True)
    Path(training_dir).mkdir(parents=True, exist_ok=True)


    sim_dirs = [str(d) for d in Path(dataset_root).iterdir() if d.is_dir()]
    #sim_dirs[:5]  # Process only the first 5 simulations for testing
    #for sim_dir in tqdm(sim_dirs[:2], desc="Processing Simulations"):
    for sim_dir in sim_dirs[19:20]:  # Process only the 5th simulation for testing

        name = Path(sim_dir).name

        simulation = af.Simulation(root=dataset_root, name=name)
        # 1. Extract Metadata from Name/Sim
        parts = name.split('_')
        v_mag_inf = float(parts[2]) 
        aoa_deg = float(parts[3])
        aoa_rad = np.deg2rad(aoa_deg)
        foil = simulation.airfoil

        # Inlet Components
        u_inf = v_mag_inf * np.cos(aoa_rad)
        v_inf = v_mag_inf * np.sin(aoa_rad)
        nu_mol = simulation.NU
        rho = simulation.RHO
        reynolds = (v_mag_inf * 1.0 / nu_mol)  # assuming chord=1.0
        log_re = np.log10(reynolds)


        points_on_foil = 201

        # Fit CST airfoil to get smooth coordinates and handle any issues with the original points
        bspline_airfoil = BSplineFoil(foil.points[:, 0], foil.points[:, 1])
        bspline_airfoil.fit_spline(smoothing=0.00)
        
        debug = False
        if debug:
            bspline_airfoil.plot_fit(points_on_foil)  # Optional: visualize the fitted B-spline airfoil
            bspline_airfoil.get_chord()
            z_te = bspline_airfoil.eval_spline(1.0)  # TE point at theta=1.0

            print(f"TE point from bspline fit: ({z_te[0]:.6f}, {z_te[1]:.6f})")
            exit()
        thetas = np.linspace(0, 2*np.pi, points_on_foil, endpoint=False)

        x_foil, y_foil = bspline_airfoil.eval_spline_theta(np.rad2deg(thetas))



        x_min_orig = np.min(x_foil)
        x_max_orig = np.max(x_foil)
        x_chord_orig = x_max_orig - x_min_orig

        print(f"Original airfoil coordinates for {name}: Max y-coordinate: {np.max(y_foil):.4f}, Min y-coordinate: {np.min(y_foil):.4f}")

        if np.max(y_foil) > 0.99 or np.min(y_foil) < -0.99:   
            print(f"Warning: Airfoil {name} appears to be wrong. Max y-coordinate: {np.max(y_foil):.4f}, Min y-coordinate: {np.min(y_foil):.4f}. Consider rescaling for better fitting.")
            bspline_airfoil.plot_fit()
        # Scale the coordinates to match expected conformal mapping domain space [-2,2] for chord    
        # Get TE point for reference
        i_te = np.argmax(x_foil)  # Assuming TE is at maximum x coordinate
        i_le = np.argmin(x_foil)  # Assuming LE is at minimum x coordinate
        x_te = x_foil[i_te] # Assuming TE is at minimum x coordinate]
        y_te = y_foil[i_te]
        # get the LE point for reference
        x_le = x_foil[i_le]  # Assuming LE is at maximum x coordinate
        y_le = y_foil[i_le]
        x_offset = (x_le + x_te) / 2   # Center the transform around the midpoint of LE and TE, then apply offset
        print(f"LE point before transform: ({x_le:.6f}, {y_le:.6f}) TE point before transform: ({x_te:.6f}, {y_te:.6f})")
        x_foil = x_foil - x_offset  # Shift coordinates to center around midpoint of LE and TE
        print(f"x_offset: {x_offset:.6f}")

      
        scale = 4.0 / (x_te - x_le)  # Initial Scale to ensure the transformed airfoil has a chord length of 2

        x_foil = x_foil * scale
        y_foil = y_foil * scale

        print(f"LE point after transform: ({x_foil[i_le]:.6f}, {y_foil[i_le]:.6f}) TE point after transform: ({x_foil[i_te]:.6f}, {y_foil[i_te]:.6f}) ")
        print(f"Name : {name}, V_inf: {v_mag_inf}, AoA: {aoa_deg} degrees")

        kt_foil = KarmanTrefftzAirfoil(mux=-0.08, muy=0.08, tau_deg=10.0)


        best_mux, best_muy, best_tau_deg = kt_foil.find_best_fit(x_foil, y_foil, num_points=points_on_foil)
        print(f"Best fit Karman-Trefftz parameters for {name}: mux={best_mux:.4f}, muy={best_muy:.4f}, tau={best_tau_deg:.2f}° ")

        scale_to_target = kt_foil.fitted_scale # Example value, replace with actual target scale if needed
        te_kt = kt_foil.fitted_te

        # adjust x_foil and y_foil to match the scale and TE position of the best fit K-T foil
        z_kt_foil = kt_foil.apply_kalman_transform(kt_foil.generate_circle_points(num_points=points_on_foil))
        x_kt_foil = z_kt_foil.real
        y_kt_foil = z_kt_foil.imag


        kt_chord = ( np.max(x_kt_foil) - np.min(x_kt_foil) )
        target_chord = np.max(x_foil) - np.min(x_foil)
        scale_to_kt = kt_chord / target_chord

        te_kt = np.max(x_kt_foil)  # TE position of the K-T foil after scaling
        te_target = np.max(x_foil)  # TE position of the original foil after scaling    
        #update x_foil to match the scale and TE position of the best fit K-T foil
        x_foil = (x_foil - te_target) * scale_to_kt  + te_kt  # 

        print(f" K-T fitted chord: {kt_chord:.4f}, Target chord: {target_chord:.4f},  Delta chord {kt_chord - target_chord:.4f}  Scale to target: {scale_to_target:.4f} ")
        
        


        plot_geometry_comparison(x_foil, y_foil, x_kt_foil, y_kt_foil)

        z_grid, zeta_surface= kt_foil.generate_conformal_grid(n_radial=50, n_theta=points_on_foil,  dr_foil = 0.01, dr_ratio=1.1)
   

        # Theta TE 

        # coordinates of k-t foil for upper surface [0-180]
        theta_te = kt_foil.get_trailing_edge_theta()
        #theta_te_rad = np.deg2rad(theta_te)
        #
        #print(f"Theat_clean {theta_clean}")
        z_te = kt_foil.apply_kalman_transform(kt_foil.generate_circle_points_at_theta(theta_te))
        print(f" Z TE {z_te}")
        
        theta_le = kt_foil.get_leading_edge_theta()
        print(f"Theta LE {theta_le}  expecting around {np.pi} rad or 180 deg ")
        z_le = kt_foil.apply_kalman_transform(kt_foil.generate_circle_points_at_theta(theta_le))
        print(f" Z LE {z_le}")


        # Angles theta from TE back to TE for complete surface
        theta = np.linspace(theta_te, theta_te + 2*np.pi, points_on_foil, endpoint= True)
        #theta_upper = theta_upper + theta_te
        theta = np.unwrap(theta)  # Unwrap to ensure continuity      
        print(f" Range of theta from {(theta[0])}  to {(theta[-1])}")

        z_kt_foil = kt_foil.apply_kalman_transform(kt_foil.generate_circle_points_at_theta(theta))
        
        # Create B-spline representation of the CST airfoil for snapping
        print(f" KT singularity point k  in z-plane: {(kt_foil.k )}")

        x_te = np.max(z_kt_foil.real ) # TE point from K-T foil
        x_le  = np.min(z_kt_foil.real)  # LE point from K-T foil

        # Set the reference points for the B-spline transform to match the TE and LE of the K-T foil,
        #  ensuring better alignment during snapping to K-T foil in the physical potential flow space
        bspline_airfoil.set_transform_reference(x_te=x_te, x_le=x_le)


        # Get t-values of points closest to the upper surface of the K-T foil to initialize snapping
        _,t_init = bspline_airfoil.snap_kt_to_bspline(z_kt_foil,  t_init=0.0,t_delta=0.05)
        print(f"Initial t values for snapping: {t_init}")

        # Calculate normals for the upper surface of the K-T foil from angles theta in zeta plane
        kt_normals = kt_foil.get_kt_normals(theta)
        # Get points on the B-spline surface closest to the upper surface of the K-T foil by snapping along the normals of K-T foil
        # dont snap the first and last point to avoid issues with the trailing edge singularity and the leading edge point
        
        z_cst_snap = z_kt_foil.copy()
        t_snap = np.zeros_like(t_init)
        z_cst_snap[1:-1], t_snap[1:-1] = bspline_airfoil.snap_kt_to_bspline_along_normal(z_kt_foil[1:-1], kt_normals[1:-1], t_init=t_init[1:-1], t_delta=1/points_on_foil, t_min=0.0, t_max=1.0)
        
        #x_cst_target, y_cst_target = bspline_airfoil.eval_spline_at_t(np.linspace(0,1.0, fit_points, endpoint= True))

        print(f"Upper Shape of x_upper: {z_kt_foil.real.shape}, Upper Shape of y_upper_target: {z_kt_foil.imag.shape}")
        print(f"Upper Shape of y_upper_target: {z_cst_snap.shape}, Lower Shape of y_lower_target: {z_cst_snap.shape}")

        #y_target = np.concatenate([y_upper_target, y_lower_target])

        #print(f"Shape of y_ideal: {y_target.shape}, Shape of y_kt_foil: {y_kt_foil.shape}")
        print(f"Shape of zeta_g: {z_grid.shape} ")
        plt.figure(figsize=(10, 6))
        plt.plot(z_cst_snap.real, z_cst_snap.imag, 'bo-', label='Target  Surface')
        plt.plot(z_kt_foil.real, z_kt_foil.imag, 'go-', label='KT Airfoil ', markersize=4)
        plt.quiver(z_cst_snap.real, z_cst_snap.imag, kt_normals[:, 0], kt_normals[:, 1], 
               color='blue', 
               angles='xy', 
               scale_units='xy', 
               scale=5, 
               width=0.003,
               label='K-T Normals on Target Upper')
        plt.axis('equal')

        plt.legend()

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Comparison of Target and KT Airfoil Surfaces')
        plt.grid(True)
        plt.show()

        # update the K-T grid to snap to the CST surface

        z_grid_smooth = kt_foil.algebraic_grid_blend(z_grid, z_cst_snap, decay_power=1.0)

        kt_foil.pyvista_overlay_mesh(z_grid, z_grid_smooth)

        uz, vz, cpz, w =  kt_foil.calculate_potential_flow(zeta_surface, v_mag=v_mag_inf, alpha_deg=aoa_deg) 
        
        X_grid = z_grid_smooth.real
        Y_grid = z_grid_smooth.imag
        pyvista_plot(X_grid, Y_grid, uz, vz, cpz, w)
        x_snap_min =np.min(z_cst_snap.real)
        x_snap_max = np.max(z_cst_snap.real)
        x_snap_chord = x_snap_max - x_snap_min
        print(f" Snap range in x: {x_snap_min:.4f} to {x_snap_max:.4f} Chord length at snap: {x_snap_chord:.4f} ")

        # scale and move to airfrans coordinate     
        scale_to_af = x_chord_orig /  x_snap_chord
        X_grid = (X_grid - x_snap_min) * scale_to_af + x_min_orig
        Y_grid = (Y_grid ) * scale_to_af
        z_grid_smooth=(z_grid_smooth -x_snap_min) * scale_to_af + x_min_orig
        print(f"Original airfoil coordinates for {name}: Max x-coordinate: {np.max(x_foil):.4f}, Min x-coordinate: {np.min(x_foil):.4f}")
        print(f"Apply scaling to airfrans coordinates with scale factor {scale_to_af:.4f} and x offset {x_min_orig:.4f}")

        # calculate the SDF
        sdf = kt_foil.calculate_grid_sdf(X_grid, Y_grid)
        pyvista_plot(X_grid, Y_grid, uz, vz, sdf, w, scalar_min=0.0, scalar_max=np.max(sdf))


        # get the airfrans results for comparison


            # 2. Grid Sampling
        mesh = simulation.internal




        xmin, xmax = (-xlen/2 + xoffset, xlen/2 + xoffset)
        ymin, ymax = (-ylen/2, ylen/2)

        x_range = np.linspace(xmin, xmax, grid_size[0])
        y_range = np.linspace(ymin, ymax, grid_size[1])
        z_plane =mesh.center[2]
        z_coords = np.full_like(z_grid_smooth.real, z_plane)
        
        grid = pv.StructuredGrid(z_grid_smooth.real, z_grid_smooth.imag, z_coords)
        print("AirfRANS Mesh Bounds (Xmin, Xmax, Ymin, Ymax, Zmin, Zmax):")
        print(mesh.bounds)

        print("\nMy K-T Grid Bounds:")
        print(grid.bounds)

        sampled = grid.sample(mesh)

        # 3. Raw Field Extraction
        sdf_raw = sampled.point_data['implicit_distance'].reshape(z_grid_smooth.shape)
        u_raw = sampled.point_data['U'][:, 0].reshape(z_grid_smooth.shape)
        v_raw = sampled.point_data['U'][:, 1].reshape(z_grid_smooth.shape)
        p_raw = sampled.point_data['p'].reshape(z_grid_smooth.shape)
        nut_raw = sampled.point_data['nut'].reshape(z_grid_smooth.shape)
        #vtk_valid =sampled.point_data['vtkValidPointMask'].reshape(z_grid_smooth.shape)

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
        cp_cfd = p_raw / (0.5 * rho * v_mag_inf**2)

        cp_pot  = cpz.flatten(order='F')
        cp_rans = cp_cfd.flatten(order='C')
        cp_diff = cp_rans - cp_pot

        sampled.point_data['cp_pot'] = cp_pot
        sampled.point_data['cp_rans'] = cp_rans
        sampled.point_data['cp_diff'] = cp_diff

        
        # Log Turbulent Viscosity Ratio (log10(nut/nu_mol))
        # We use a floor of 1e-10 to prevent log(0)
        log_nut_ratio = np.log10(np.clip((nut_raw + 1e-10) / nu_mol, a_min=1e-10, a_max=None))

        # check nu?_mol > 0
        if nu_mol <= 0:
            raise ValueError(f"Invalid molecular viscosity (nu_mol): {nu_mol}. Must be positive.")
        
        log_nut_ratio = np.log10(np.clip(nut_raw, 1e-12, None) / nu_mol)

        # 5. Visualization for Debugging
        pyvista_plot_airfrans_scalar(sampled, scalar_name="cp_rans", scalar_min=-1.0, scalar_max=1.0)
        pyvista_plot_airfrans_scalar(sampled, scalar_name="cp_pot", scalar_min=-1.0, scalar_max=1.0)
        pyvista_plot_airfrans_scalar(sampled, scalar_name="cp_diff", scalar_min=-1.0, scalar_max=1.0)
        pyvista_plot_airfrans_scalar(sampled, scalar_name="vtkValidPointMask", scalar_min=0, scalar_max=1.0)






if __name__ == "__main__":
    # Example usage
    #airfoil = KarmanTrefftzAirfoil(mux=-0.08, muy=0.08)
    #airfoil.plot_airfoil(num_points=200, tau_deg=10.0)




    PATH_TO_DATASET = "/home/timm/Projects/PIML/Dataset"
    PATH_TO_OUTPUT = "/home/timm/Projects/PIML/Dataset_Cyl_Pot_FNO_X5Y4"
    GRID_SIZE = (64, 64)
    REGENERATE = True  # Set to True to regenerate all pt files
    xlen = float(6.0)   # domain length to be sampled  (-xlen/2, xlen/2)
    ylen = float(3.0)   # domain height to be sampled  (-ylen/2, ylen/2)
    xoffset = float(1.0)  # x-offset to be sampled

    convert(PATH_TO_DATASET, PATH_TO_OUTPUT, xlen, ylen, xoffset, GRID_SIZE)