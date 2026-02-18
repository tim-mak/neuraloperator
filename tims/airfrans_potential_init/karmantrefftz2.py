from matplotlib.pylab import gamma
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import airfrans as af
from foil_utils import CSTAirfoil
from scipy.interpolate import griddata, interp1d, splprep, splev
import pyvista as pv

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
    
    def generate_circle_grid(self,  n_radial=50 , n_theta=50, far_field_radius=5.0):
        """Generate a grid in the zeta-plane that can be transformed to the physical plane"""

        # Inner radius is R to ensure we are outside the singularity at the center of the circle in the zeta-plane
        r_min = self.R 

        # get TE and LE theta for the circle grid
        theta_te = self.get_trailing_edge_theta()

        r = np.linspace(r_min, far_field_radius, n_radial)  
        theta = np.linspace(theta_te, theta_te + 2*np.pi, n_theta, endpoint=True)
        theta = np.unwrap(theta)  # Unwrap to ensure continuity

        R, Theta = np.meshgrid(r, theta)
        zeta = R * np.exp(1j * Theta) + complex(self.mux, self.muy)  # Shift to center at (mux, muy)
        return zeta
    
    def generate_conformal_grid(self,  n_radial=50, n_theta=50, far_field_radius=5.0):
        """Generate a conformal grid in the zeta-plane that can be transformed to the physical plane"""
        
        
        # Generate points on the airfoil surface (zeta-plane)
        zeta_surface = self.generate_circle_grid(n_radial=n_radial, n_theta=n_theta, far_field_radius=far_field_radius)
        
        # Apply Karman-Trefftz transformation to get physical coordinates
        z_grid = self.apply_kalman_transform(zeta_surface)

        return z_grid

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

    def compute_source_terms(self, X, Y):
        """
        Computes the P and Q source terms from an initial K-T grid using second-order central differences.
        Assumes X, Y are shape (N_xi, N_eta).
        """
        # 1st derivatives
        X_xi = 0.5 * (X[2:, 1:-1] - X[:-2, 1:-1])
        X_eta = 0.5 * (X[1:-1, 2:] - X[1:-1, :-2])
        Y_xi = 0.5 * (Y[2:, 1:-1] - Y[:-2, 1:-1])
        Y_eta = 0.5 * (Y[1:-1, 2:] - Y[1:-1, :-2])

        # 2nd derivatives
        X_xixi = X[2:, 1:-1] - 2*X[1:-1, 1:-1] + X[:-2, 1:-1]
        Y_xixi = Y[2:, 1:-1] - 2*Y[1:-1, 1:-1] + Y[:-2, 1:-1]

        X_etaeta = X[1:-1, 2:] - 2*X[1:-1, 1:-1] + X[1:-1, :-2]
        Y_etaeta = Y[1:-1, 2:] - 2*Y[1:-1, 1:-1] + Y[1:-1, :-2]

        # Source terms (Hilgenstock / Thomas-Middlecoff formulation)
        # 1e-10 prevents division by zero in highly orthogonal regions
        P = -(X_xi * X_xixi + Y_xi * Y_xixi) / (X_xi**2 + Y_xi**2 + 1e-10)
        Q = -(X_eta * X_etaeta + Y_eta * Y_etaeta) / (X_eta**2 + Y_eta**2 + 1e-10)
        print(f"Computed source terms P and Q with shapes {P.shape} and {Q.shape}")

        return P, Q
    
    def smooth_elliptic_grid(self, X, Y, P, Q, iterations=50):
        """
         Relaxes internal grid points using Poisson equations to preserve spacing.
        Assumes shape (N_xi, N_eta) where:
        X[:, 0] is the airfoil surface
        X[:, -1] is the far-field boundary
        X[0, :] and X[-1, :] meet at the trailing edge wake cut
        P and Q are the source terms computed from the initial K-T grid, shape (N_xi-2, N_eta-2) for internal points.
        """

        for it in range(iterations):
            X_old = X.copy()
            Y_old = Y.copy()
            
            # 1. First Derivatives
            X_xi = 0.5 * (X[2:, 1:-1] - X[:-2, 1:-1])
            X_eta = 0.5 * (X[1:-1, 2:] - X[1:-1, :-2])
            Y_xi = 0.5 * (Y[2:, 1:-1] - Y[:-2, 1:-1])
            Y_eta = 0.5 * (Y[1:-1, 2:] - Y[1:-1, :-2])
            
            # 2. Metric coefficients
            alpha = X_eta**2 + Y_eta**2
            beta = X_xi * X_eta + Y_xi * Y_eta
            gamma = X_xi**2 + Y_xi**2
            
            # 3. Cross derivatives
            X_xiet = 0.25 * (X[2:, 2:] - X[2:, :-2] - X[:-2, 2:] + X[:-2, :-2])
            Y_xiet = 0.25 * (Y[2:, 2:] - Y[2:, :-2] - Y[:-2, 2:] + Y[:-2, :-2])
            # 2. UPDATE ONLY FROM j=2 ONWARD
            # 4. Poisson Update (Notice the addition of P * X_xi and Q * X_eta)
            X[1:-1, 2:-1] = (alpha[:, 1:] * (X[2:, 2:-1] + X[:-2, 2:-1] + P[:, 1:] * X_xi[:, 1:]) - 
                         2 * beta[:, 1:] * X_xiet[:, 1:] + 
                         gamma[:, 1:] * (X[1:-1, 3:] + X[1:-1, 1:-2] + Q[:, 1:] * X_eta[:, 1:])) / (2 * (alpha[:, 1:] + gamma[:, 1:]))
                         
            Y[1:-1, 2:-1] = (alpha[:, 1:] * (Y[2:, 2:-1] + Y[:-2, 2:-1] + P[:, 1:] * Y_xi[:, 1:]) - 
                         2 * beta[:, 1:] * Y_xiet[:, 1:] + 
                         gamma[:, 1:] * (Y[1:-1, 3:] + Y[1:-1, 1:-2] + Q[:, 1:] * Y_eta[:, 1:])) / (2 * (alpha[:, 1:] + gamma[:, 1:]))
            
            # 5. Handle the Periodic Wake Cut
            X[0, 1:-1] = X[-1, 1:-1] = 0.5 * (X[1, 1:-1] + X[-2, 1:-1])
            Y[0, 1:-1] = Y[-1, 1:-1] = 0.5 * (Y[1, 1:-1] + Y[-2, 1:-1])
            
        return X, Y


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


        points_on_foil = 101

        # Fit CST airfoil to get smooth coordinates and handle any issues with the original points
        cst_airfoil = CSTAirfoil(foil.points[:, 0], foil.points[:, 1], n_points= points_on_foil, n_order=14)
        cst_airfoil.plot_fit()
        coords = cst_airfoil.get_coords_sellig_format()


        x_foil = coords[:, 0]
        y_foil = coords[:, 1]

        if np.max(y_foil) > 0.99 or np.min(y_foil) < -0.99:   
            print(f"Warning: Airfoil {name} appears to be wrong. Max y-coordinate: {np.max(y_foil):.4f}, Min y-coordinate: {np.min(y_foil):.4f}. Consider rescaling for better fitting.")
            cst_airfoil.plot_fit()
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

        zeta_g = kt_foil.generate_conformal_grid(n_radial=50, n_theta=points_on_foil, far_field_radius=5.0)

    
        kt_foil.pyvista_mesh(zeta_g)

        fit_points = 101

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
        theta = np.linspace(theta_te, theta_te + 2*np.pi, fit_points, endpoint= True)
        #theta_upper = theta_upper + theta_te
        theta = np.unwrap(theta)  # Unwrap to ensure continuity      

        #theta_upper_le = theta_upper[-1]
        #print(f" Theta upper LE {theta_upper_le} expecting around {theta_le} rad or 180 deg ")
        #theta_upper = (theta_upper + 2*np.pi) % (2 * np.pi) - np.pi

        print(f" Range of theta from {(theta[0])}  to {(theta[-1])}")
        z_kt_foil = kt_foil.apply_kalman_transform(kt_foil.generate_circle_points_at_theta(theta))
        
        # Create B-spline representation of the CST airfoil for snapping
        print(f" KT singularity point k  in z-plane: {(kt_foil.k )}")

        x_te = np.max(z_kt_foil.real ) # TE point from K-T foil
        x_le  = np.min(z_kt_foil.real)  # LE point from K-T foil

        cst_airfoil.create_splinerep( x_te= x_te, x_le= x_le)


        # Get t-values of points closest to the upper surface of the K-T foil to initialize snapping
        _,t_init = cst_airfoil.snap_kt_to_bspline(z_kt_foil,  t_init=0.0,t_delta=0.05)

        # Calculate normals for the upper surface of the K-T foil from angles theta in zeta plane
        kt_normals = kt_foil.get_kt_normals(theta)
        # Get points on the CST surface closest to the upper surface of the K-T foil by snapping along the normals of K-T foil
        # dont snap the first and last point to avoid issues with the trailing edge singularity and the leading edge point
        
        z_cst_snap = z_kt_foil.copy()
        t_snap = np.zeros_like(t_init)
        z_cst_snap[1:-1], t_snap[1:-1] = cst_airfoil.snap_kt_to_bspline_along_normal(z_kt_foil[1:-1], kt_normals[1:-1], t_init=t_init[1:-1], t_delta=1/fit_points, t_min=0.0, t_max=1.0)
        
        #x_cst_target, y_cst_target = cst_airfoil.eval_spline_at_t(np.linspace(0,1.0, fit_points, endpoint= True))

        print(f"Upper Shape of x_upper: {z_kt_foil.real.shape}, Upper Shape of y_upper_target: {z_kt_foil.imag.shape}")
        print(f"Upper Shape of y_upper_target: {z_cst_snap.shape}, Lower Shape of y_lower_target: {z_cst_snap.shape}")

        #y_target = np.concatenate([y_upper_target, y_lower_target])

        #print(f"Shape of y_ideal: {y_target.shape}, Shape of y_kt_foil: {y_kt_foil.shape}")
        print(f"Shape of zeta_g: {zeta_g.shape} ")
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

        plt.legend()

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Comparison of Target and KT Airfoil Surfaces')
        plt.grid(True)
        plt.show()

        # update the K-T grid to snap to the CST surface

        print(f" Shape of z_cst_snap: {z_cst_snap.shape}, Shape of z_kt_foil: {z_kt_foil.shape} ")
        zeta_g_snapped = zeta_g.copy()


        # lock the height of the first layer of points above the surface to prevent issues with the elliptic solver near the surface
        # 1. Snap the surface (j=0) 
        #zeta_g_snapped = zeta_g.copy()
        #zeta_g_snapped.real[1:-2, 0] = z_cst_snap[1:-2].real
        #zeta_g_snapped.imag[1:-2, 0] = z_cst_snap[1:-2].imag

        # 2. Extract the exact normal vector and distance of the original 1st cell
        #dx_wall = zeta_g.real[:, 1] - zeta_g.real[:, 0]
        #dy_wall = zeta_g.imag[:, 1] - zeta_g.imag[:, 0]

        # 3. Apply this exact offset to the newly snapped surface
        #zeta_g_snapped.real[:, 1] = zeta_g_snapped.real[:, 0] + dx_wall
        #zeta_g_snapped.imag[:, 1] = zeta_g_snapped.imag[:, 0] + dy_wall

        #kt_foil.pyvista_mesh(zeta_g_snapped)

        #zeta_g_smooth = zeta_g_snapped.copy()

        # calculate source terms P and Q from the initial K-T grid
        #P, Q = kt_foil.compute_source_terms(zeta_g.real, zeta_g.imag)
        
        #print(f"Computed source terms P and Q with shapes {P.shape} and {Q.shape}")

        #P = P*0

        #Q = Q*0

        #zeta_g_smooth.real, zeta_g_smooth.imag = kt_foil.smooth_elliptic_grid(zeta_g_smooth.real, zeta_g_smooth.imag, P, Q, iterations=1000)
        
        #kt_foil.pyvista_mesh(zeta_g_smooth)
        zeta_g_smooth = kt_foil.algebraic_grid_blend(zeta_g, z_cst_snap, decay_power=1.0)

        kt_foil.pyvista_overlay_mesh(zeta_g, zeta_g_snapped)


        kt_foil.pyvista_overlay_mesh(zeta_g, zeta_g_smooth)



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