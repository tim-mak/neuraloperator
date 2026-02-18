from scipy.interpolate import splprep,splev
import numpy as np  
from scipy.optimize import minimize_scalar


class BSplineFoil():

    
    def __init__(self, x, y):
        """
        Initializes the B-spline representation of the airfoil.
        
        Parameters:
        - x: Array of x-coordinates of the airfoil points.
        - y: Array of y-coordinates of the airfoil points.
        """

        x,y = self.sort_points(x,y)
        self.x_raw = x
        self.y_raw = y

        self.fit_spline(smoothing=0.01)


    def sort_points(self,x,y):
        """
        Sorts the airfoil points in a consistent order (e.g., from trailing edge to trailing edge).
        This is important for fitting the B-spline correctly.
        """
        # Compute the centroid of the points
        centroid_x = np.mean(x)
        centroid_y = np.mean(y)
        
        # Compute angles from the centroid to each point
        angles = np.arctan2(y - centroid_y, x - centroid_x)

        # 3. Identify the Trailing Edge (typically the maximum X coordinate)
        te_idx = np.argmax(x)
        te_angle = angles[te_idx]

        shifted_angles = (angles - te_angle) % (2 * np.pi)

        
        # 5. Sort by the new continuous angle
        sort_idx = np.argsort(shifted_angles)
        
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]
        
        return x_sorted, y_sorted
    
    def eval_spline_theta(self,theta, derivative=0):
        """
        Evaluates the B-spline at given theta values (0 to 360 degrees).
        
        Parameters:
        - theta: Array of angles (in degrees) where the spline is evaluated.
        - derivative: Derivative order (0 for position, 1 for first derivative, etc.)
        
        Returns:
        - x_eval, y_eval: Arrays of x and y coordinates of the spline at the specified theta values.
        """
        # Convert theta to parameter t (0 to 1)
        t_values = np.array(theta) / 360.0
        return self.eval_spline(t_values, derivative=derivative)
    

    
    def fit_spline(self,smoothing =0.0):

        if not (np.isclose(self.x_raw[0], self.x_raw[-1]) and np.isclose(self.y_raw[0], self.y_raw[-1])):
            self.x_raw = np.append(self.x_raw, self.x_raw[0])
            self.y_raw = np.append(self.y_raw, self.y_raw[0])
        
        # 3. Fit the Parametric B-Spline
        # per=1 enforces periodic boundary conditions (a perfectly smooth closed loop)
        # s is the smoothing factor (0.0 forces the spline exactly through every point)
        fit_spline, u_param = splprep([self.x_raw, self.y_raw], s=smoothing, per=1)

        self.spline = fit_spline
        self.u_param = u_param

    def eval_spline(self, t_values, derivative=0):
        """
        Evaluates the B-spline at given parameter values.
        
        Parameters:
        - t_values: Array of B-spline parameter values where the spline is evaluated.
        - derivative: Derivative order (0 for position, 1 for first derivative, etc.)
        
        Returns:
        - x_eval, y_eval: Arrays of x and y coordinates of the spline at the specified parameter values.
        """
        x_eval, y_eval = splev(t_values, self.spline, der=derivative)
        return np.array(x_eval), np.array(y_eval)

        
    def compute_normals(self, t_values):
        """
        Computes the normal vectors at given B-spline parameter values.
        
        Parameters:
        - t_values: Array of B-spline parameter values where normals are computed.
        
        Returns:
        - normals: Array of normal vectors at the specified parameter values.
        """
        # Compute derivatives
        dx_dt, dy_dt = self.eval_spline(t_values, derivative=1)
        
        # Compute norms of the tangent vectors
        norms = np.sqrt(dx_dt**2 + dy_dt**2)
        
        # Avoid division by zero
        norms[norms == 0] = 1e-8
        
        # Compute normals (rotate tangent by 90 degrees)
        normals = np.column_stack((-dy_dt / norms, dx_dt / norms))

        return normals

    def compute_tangents(self, t_values):
        """
        Computes the tangent vectors at given B-spline parameter values.
        
        Parameters:
        - t_values: Array of B-spline parameter values where tangents are computed.
        
        Returns:
        - tangents: Array of tangent vectors at the specified parameter values.
        """
        # Compute derivatives
        dx_dt, dy_dt = self.eval_spline(t_values, derivative=1)
        
        # Compute norms of the tangent vectors
        norms = np.sqrt(dx_dt**2 + dy_dt**2)
        
        # Avoid division by zero
        norms[norms == 0] = 1e-8
        
        # Compute tangents (normalize the derivative)
        tangents = np.column_stack((dx_dt / norms, dy_dt / norms))
        
        return tangents
    
    def get_chord_te_le(self):
        """
        Computes the chord length of the airfoil (distance between leading and trailing edge).
        
        Returns:
        - chord_length: The chord length of the airfoil.
        """
        # Assuming the first point is the trailing edge and the point with maximum x is the leading edge
        x_te = np.max(self.x_raw)
        i_te = np.argmax(self.x_raw)
        y_te = self.y_raw[i_te]

        x_le = np.min(self.x_raw)
        i_le = np.argmin(self.x_raw)
        y_le = self.y_raw[i_le]

        te_point = np.array([x_te, y_te])
        le_idx = i_le
        le_point = np.array([x_le, y_le])
        
        chord_length = x_te - x_le
        print(f"Chord length: {chord_length:.4f} TE: {te_point}  i_te: {i_te} LE: {le_point}  i_le: {i_le}")
        return chord_length, te_point, le_point
    
    def set_transform_reference(self, x_te, x_le):

        target_chord = x_te - x_le
        # Get current TE and chord from the B-spline fit
        current_chord, current_te, current_le = self.get_chord_te_le()
        # Compute scaling factor to match the target chord length
        scale_factor = target_chord / current_chord
        # Compute translation to align the trailing edge
        translation = complex(x_te - current_te[0], 0.0)
        self.scale_factor = scale_factor
        self.translation = translation

        print(f"Set transform reference: target_chord={target_chord:.4f}, current_chord={current_chord:.4f}, scale_factor={scale_factor:.4f}, translation={translation}")

    
    def find_t_closest_xy(self, x, y, t_min, t_max):
        """
        Finds the B-spline parameter t that matches the 
        x,y coordinates by minimizing the distance between the spline point P(t) and the target (x, y).
        """
        # Objective: Minimize the Euclidean distance between 
        # the B-spline point P(t) and the target (x, y)
        if not hasattr(self, 'spline'):
            raise ValueError("Spline representation not created. Call create_splinerep() first.")
               
        def objective(t):
            px, py = splev(t, self.spline)
            return np.sqrt((px - x)**2 + (py - y)**2)

        # Search in the middle of the loop (t ~ 0.5)
        res = minimize_scalar(objective, bounds=(t_min, t_max), method='bounded')
        #print(f"Closest point on spline to ({x:.4f}, {y:.4f}) is at t={res.x:.4f} with distance {res.fun:.2e}")
        return res.x
    
    def snap_kt_to_bspline(self, kt_points,  t_init=0.0, t_delta=0.02, t_min=0, t_max=1.0):
        """
        Snaps K-T guide points to the B-spline along their normals.
        """
        snapped_points = []
        t_values = []

        t_guess = t_init  # Start with the first point's initial guess

        scale_factor = getattr(self, 'scale_factor', 1.0)
        chord,te,le = self.get_chord_te_le()
        x_te = te[0]  # TE point from B-spline fit

        # scale the kt_points down  to match the B-spline reference frame
        kt_x_te  = np.max(kt_points.real)  # TE point from K-T foil
        kt_points_transformed = (kt_points - kt_x_te) / scale_factor + x_te

        for i in range(len(kt_points_transformed)):
            # Objective: Find t that minimizes distance between B-spline P(t) 
            t_low = max(t_min, t_guess - t_delta)
            t_high = min(t_max, t_guess + t_delta)
            
            t_sol = self.find_t_closest_xy(kt_points_transformed[i].real, kt_points_transformed[i].imag, t_low, t_high)
            
            # Evaluate the spline at the found t
            px, py = splev(t_sol, self.spline)
            
            # scale back to original reference frame
            px = (px - x_te) * scale_factor + kt_x_te
            py = (py ) * scale_factor
            snapped_points.append(complex(px, py))
            t_values.append(t_sol)
            
            # Update guess for next point (points are ordered)
            t_guess = t_sol
        
        return np.array(snapped_points), t_values

    def snap_kt_to_bspline_along_normal(self, kt_points, kt_normals, t_init=0.0, t_delta=0.02, t_min=0, t_max=1.0):
        """
        Snaps K-T guide points to the B-spline by finding the intersection
        of the K-T normal ray and the B-spline curve.
        """
        snapped_points = []
        t_values = []
        # Start the search near the previous t to speed up convergence
        if isinstance(t_init, list) or isinstance(t_init, np.ndarray):
            # if a list of initial t values is provided, use them for each point
            t_guess = t_init  # Start with the first point's initial guess
        else:
            # If a single value is provided, use it for all points
            t_guess = np.ones_like(kt_points) * t_init  

        scale_factor = getattr(self, 'scale_factor', 1.0)
        chord,te,le = self.get_chord_te_le()
        x_te = te[0]  # TE point from B-spline fit

        # scale the kt_points down  to match the B-spline reference frame
        kt_x_te  = np.max(kt_points.real)  # TE point from K-T foil

        print(f"Scaling K-T points with scale_factor={1/scale_factor:.4f} and TE={x_te}")
        kt_points_transformed = (kt_points - kt_x_te) / scale_factor + x_te    
        
        for i in range(len(kt_points_transformed)):
            p_kt = kt_points_transformed[i]
            n_kt = kt_normals[i] # Expecting [nx, ny]

            # Define the objective: Minimize the perpendicular distance 
            # from the spline point P(t) to the ray (p_kt + alpha * n_kt)
            def objective(t):
                px, py = splev(t, self.spline)
                # 2D Cross product magnitude: |(P - P_kt) x n_kt|
                dist_to_ray = abs((px - p_kt.real) * n_kt[1] - (py - p_kt.imag) * n_kt[0])
                
                # Add a small penalty for Euclidean distance to ensure we pick 
                # the intersection closest to the airfoil, not one on the far side
                euclidean_dist = np.sqrt((px - p_kt.real)**2 + (py - p_kt.imag)**2)
                return dist_to_ray + 0.01 * euclidean_dist

            # Local window search to maintain ordering and surface integrity
            t_low = max(t_min, t_guess[i] - t_delta)
            t_high = min(t_max, t_guess[i] + t_delta)
            
            # Use a more robust optimizer if brentq/minimize_scalar struggles
            res = minimize_scalar(objective, bounds=(t_low, t_high), method='bounded')
            
            t_sol = res.x
            px, py = splev(t_sol, self.spline)

            # scale back to original K-T reference frame
            px = (px - x_te) * scale_factor + kt_x_te
            py = (py ) * scale_factor

            print(f"Up Scaling K-T points with scale_factor={scale_factor:.4f} and TE={x_te}")

            
            snapped_points.append(complex(px, py))
            t_values.append(t_sol)
            
            
        return np.array(snapped_points), t_values
    
    def plot_fit(self, n_points=100):
        import matplotlib.pyplot as plt
        thetas = np.linspace(0, 360.0, n_points, endpoint=True)

        x_foil, y_foil = self.eval_spline_theta(thetas)

        plt.figure(figsize=(8, 4))
        plt.plot(self.x_raw, self.y_raw, 'ro', label='Original Points')
        plt.plot(x_foil, y_foil, 'b-', label='B-Spline Fit')
        plt.axis('equal')
        plt.title('B-Spline Fit to Airfoil')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid()
        plt.show()