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
    

    def find_best_fit(self, x_foil, y_foil, num_points=101):
        """Find the best fit Karman-Trefftz parameters for the given airfoil coordinates"""
        from scipy.optimize import minimize

        def objective(params):
            mux, muy , tau_deg= params

            self.set_params(mux, muy, tau_deg)
            zeta = self.generate_circle_points( num_points)

            z = self.apply_kalman_transform(zeta)
            # Interpolate z to the x_foil points and compute distance

            # get interpolated points on the airfoil for upper and lower surface
            z_upper = z[zeta.imag >= 0]
            z_lower = z[zeta.imag < 0]

            le_index = np.argmin(x_foil)
            y_upper = y_foil[:le_index]
            y_lower = y_foil[le_index:]

            #print(f" ")
            #print(f" z_upper values: {z_upper[:5]} ... {z_upper[:-5]}")
            #print(f" All")
            #print(f" z_upper values: min {min(z_upper.imag)} ... max {max(z_upper.imag)} ")
            #print(f" z_lower values: min {min(z_lower.imag)} ... max {max(z_lower.imag)} ")

            # Need to sort upper surface point so that x is ascending
            x_foil_upper_sorted_indices = np.argsort(x_foil[:le_index])
            z_upper_sorted_indices = np.argsort(z_upper.real)

            z_upper_interp = np.interp(x_foil[x_foil_upper_sorted_indices], z_upper.real[z_upper_sorted_indices], z_upper.imag[z_upper_sorted_indices])
            z_upper_interp = np.flip(z_upper_interp)  # Flip to match the order of y_upper
            z_lower_interp = np.interp(x_foil[le_index:], z_lower.real, z_lower.imag)
            
            #print(f" x_upper values: {x_foil[:5]} ... {x_foil[-5:]}")

            #print(f" z_upper_interp  values: min {min(z_upper_interp)} ... max {max(z_upper_interp)}")
            #print(f" z_lower_interp  values: min {min(z_lower_interp)} ... max {max(z_lower_interp)}")


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
                     

            print("Testing params: mux={:.4f}, muy={:.4f}, tau={:.2f}°, y_dist_mean ={:.2f}".format(mux, muy, tau_deg, y_dist_mean))
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
    
    def generate_circle_grid(self, n_radial=50,n_theta=50, far_field_radius=5.0):
        """Generate a grid in the zeta-plane that can be transformed to the physical plane"""
        theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=True)

        # Inner radius is R to ensure we are outside the singularity at the center of the circle in the zeta-plane
        r_min = self.R 

        r = np.linspace(r_min, far_field_radius, n_radial)  
        R, Theta = np.meshgrid(r, theta)
        zeta = R * np.exp(1j * Theta) + complex(self.mux, self.muy)  # Shift to center at (mux, muy)
        return zeta
    
    def generate_conformal_grid(self, n_radial=50, n_theta=50, far_field_radius=5.0):
        """Generate a conformal grid in the zeta-plane that can be transformed to the physical plane"""
        # Generate points on the airfoil surface (zeta-plane)
        zeta_surface = self.generate_circle_grid(n_radial=n_radial, n_theta=n_theta, far_field_radius=far_field_radius)
        
        # Generate points in the far-field (circular grid in zeta-plane)
        theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)


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

        # Fit CST airfoil to get smooth coordinates and handle any issues with the original points
        cst_airfoil = CSTAirfoil(foil.points[:, 0], foil.points[:, 1], n_points= 101, n_order=14)
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

      
        scale = 4.0 / (x_te - x_le)  # Scale to ensure the transformed airfoil has a chord length of 2

        x_foil = x_foil * scale
        y_foil = y_foil * scale

        print(f"LE point after transform: ({x_foil[i_le]:.6f}, {y_foil[i_le]:.6f}) TE point after transform: ({x_foil[i_te]:.6f}, {y_foil[i_te]:.6f}) ")
        print(f"Name : {name}, V_inf: {v_mag_inf}, AoA: {aoa_deg} degrees")

        kt_foil = KarmanTrefftzAirfoil(mux=-0.08, muy=0.08, tau_deg=10.0)

        n_points =101

        best_mux, best_muy, best_tau_deg = kt_foil.find_best_fit(x_foil, y_foil, num_points=n_points)
        print(f"Best fit Karman-Trefftz parameters for {name}: mux={best_mux:.4f}, muy={best_muy:.4f}, tau={best_tau_deg:.2f}°")
        
        z = kt_foil.apply_kalman_transform(kt_foil.generate_circle_points(num_points=n_points))
        
        
        x_kt_foil = z.real
        y_kt_foil = z.imag

        plot_geometry_comparison(x_foil, y_foil, x_kt_foil, y_kt_foil)

        zeta_g = kt_foil.generate_conformal_grid(n_radial=50, n_theta=n_points, far_field_radius=5.0)

    
        kt_foil.pyvista_mesh(zeta_g)

        fit_points = 51

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



        theta_upper = np.linspace(0, np.pi, fit_points, endpoint= True)
        theta_upper = theta_upper + theta_te
        theta_upper = np.unwrap(theta_upper)  # Unwrap to ensure continuity

        theta_upper_le = theta_upper[-1]
        print(f" Theta upper LE {theta_upper_le} expecting around {theta_le} rad or 180 deg ")
        #theta_upper = (theta_upper + 2*np.pi) % (2 * np.pi) - np.pi

        print(f" Range of theta for upper from {(theta_upper[0])}  to {(theta_upper[-1])}")


        z_upper = kt_foil.apply_kalman_transform(kt_foil.generate_circle_points_at_theta(theta_upper))
        print(f" Leading edge point from upper surface: {z_upper[-1]} Trailing edge point from upper surface: {z_upper[0]}")
        
        z_le_alt = (z_upper[-1])
        print(f" Z-plane LE from upper surface: ({z_le_alt.real:.6f}, {z_le_alt.imag:.6f})")       
        
        print(f" KT singularity point k  in z-plane: {(kt_foil.k )}")
        cst_airfoil.create_splinerep( x_te= kt_foil.k, x_le= -kt_foil.k)
        # test spline eval
        eval_x, eval_y = cst_airfoil.eval_spline_at_t(0.5055)
        print(f" Spline eval at t=0.5: ({eval_x:.6f}, {eval_y:.6f})")

        t_le = cst_airfoil.find_t_closest_xy(z_le_alt.real, z_le_alt.imag, t_min=0.4, t_max=0.6)

        x_le,y_le = cst_airfoil.eval_spline_at_t(t_le)
        print(f" Closest point on CST to LE: ({x_le:.6f}, {y_le:.6f}) at t={t_le:.4f}")

        # Get t-values of points closest to the upper surface of the K-T foil to initialize snapping
        _,t_upper_init = cst_airfoil.snap_kt_to_bspline(z_upper,  t_init=0.0,t_delta=0.05)

        # Calculate normals for the upper surface of the K-T foil from angles theta in zeta plane
        kt_normal_upper = kt_foil.get_kt_normals(theta_upper)
        # Get points on the CST surface closest to the upper surface of the K-T foil by snapping along the normals of K-T foil
        kt_closest_upper, t_upper = cst_airfoil.snap_kt_to_bspline_along_normal(z_upper, kt_normal_upper, t_init=t_upper_init, t_delta=0.05)


        print(f" T-dist TE: {t_upper} expecting around 0.0 to {t_le} ")
        
        theta_lower = np.linspace(np.pi, 2*np.pi, fit_points, endpoint= True)
        theta_lower = theta_lower + theta_te 
        theta_lower = np.unwrap(theta_lower)  # Unwrap to ensure continuity

        # coordinates of k-t foil for lower surface [180-360]
        z_lower = kt_foil.apply_kalman_transform(kt_foil.generate_circle_points_at_theta(theta_lower))
        _,t_lower_init = cst_airfoil.snap_kt_to_bspline(z_lower,  t_init=t_le,t_delta=0.05, t_min=t_le, t_max=1.0)

        kt_normal_lower = kt_foil.get_kt_normals(theta_lower)
        kt_closest_lower, t_lower = cst_airfoil.snap_kt_to_bspline_along_normal(z_lower, kt_normal_lower, t_init=t_lower_init,t_delta=0.05)

        

        #le_idx = np.argmin(x_kt_foil)

        x_kt_foil_upper = z_upper.real
        x_kt_foil_lower = z_lower.real

        y_kt_foil_upper = z_upper.imag
        y_kt_foil_lower = z_lower.imag

        x_kt_foil_upper_min = np.min(x_kt_foil_upper)
        x_kt_foil_upper_max = np.max(x_kt_foil_upper)
        x_kt_foil_lower_min = np.min(x_kt_foil_lower)
        x_kt_foil_lower_max = np.max(x_kt_foil_lower)



        x_upper_target, y_upper_target = cst_airfoil.eval_spline_at_t(np.linspace(0,t_le, fit_points, endpoint= True))
        x_lower_target, y_lower_target = cst_airfoil.eval_spline_at_t(np.linspace(t_le, 1, fit_points, endpoint= True))

        print(f"Upper Shape of x_upper: {x_kt_foil_upper.shape}, Upper Shape of y_upper_target: {y_upper_target.shape}")
        print(f"Upper Shape of y_upper_target: {y_upper_target.shape}, Lower Shape of y_lower_target: {y_lower_target.shape}")

        #y_target = np.concatenate([y_upper_target, y_lower_target])

        #print(f"Shape of y_ideal: {y_target.shape}, Shape of y_kt_foil: {y_kt_foil.shape}")
        print(f"Shape of zeta_g: {zeta_g.shape} ")
        plt.figure(figsize=(10, 6))
        plt.plot(kt_closest_upper.real, kt_closest_upper.imag, 'bo-', label='Target Upper Surface')
        plt.plot(x_kt_foil_upper, y_kt_foil_upper, 'go-', label='KT Airfoil - Upper', markersize=4)
        plt.quiver(kt_closest_upper.real, kt_closest_upper.imag, kt_normal_upper[:, 0], kt_normal_upper[:, 1], 
               color='blue', 
               angles='xy', 
               scale_units='xy', 
               scale=5, 
               width=0.003,
               label='K-T Normals on Target Upper')
        plt.plot(kt_closest_lower.real, kt_closest_lower.imag, 'ms-', label='Target Lower Surface')
        plt.plot(x_kt_foil_lower, y_kt_foil_lower, 'rs-', label='KT Airfoil - Lower', markersize=4)
        plt.quiver(kt_closest_lower.real, kt_closest_lower.imag, kt_normal_lower[:, 0], kt_normal_lower[:, 1], 
               color='magenta', 
               angles='xy', 
               scale_units='xy', 
               scale=5, 
               width=0.003,
               label='K-T Normals on Target Lower')
        plt.legend()

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Comparison of Target and KT Airfoil Surfaces')
        plt.grid(True)
        plt.show()

        # update the K-T grid to snap to the CST surface
        zeta_g_snapped = zeta_g.copy()
        zeta_g_snapped[1:-1, 0] = x_kt_foil[1:-1] + 1j * y_upper_target[1:-1]

        kt_foil.pyvista_mesh(zeta_g_snapped)




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