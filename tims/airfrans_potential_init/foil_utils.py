import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from scipy.interpolate import CubicSpline, splprep, splev
from scipy.optimize import minimize_scalar

import numpy as np
from scipy.optimize import lsq_linear
from scipy.special import comb

from scipy.interpolate import CubicSpline

class CSTAirfoil:
    def __init__(self, x_coords, y_coords,  n_points = 100,n_order=8):
        """Initialize CST Airfoil with given coordinates."""


        x_coords, y_coords = self.align_and_scale(x_coords, y_coords)
        x_coords_resample , yu_new,  yl_new = self.resample_airfoil(x_coords, y_coords, n_points=n_points)
        self.x_coords, self.y_upper = self.align_and_scale(x_coords_resample, yu_new)
        self.x_coords, self.y_lower = self.align_and_scale(x_coords_resample, yl_new)

        x_min, x_max = np.min(self.x_coords), np.max(self.x_coords)
        self.psi = (self.x_coords - x_min) / (x_max - x_min)
        self.psi = np.clip(self.psi, 0.0, 1.0)
        self.y_upper = self.y_upper
        self.y_lower = self.y_lower
        self.n_order = n_order


        best_upper_params = self.optimize_airfoil_parameterization( self.y_upper)

        best_lower_params = self.optimize_airfoil_parameterization( self.y_lower)


        self.params_upper = best_upper_params
        self.params_lower = best_lower_params

    def resample_airfoil(self,x, y, n_points=101):
        """Clean and resample airfoil coordinates using cosine spacing."""
        # 1. Normalize
        x = (x - x.min()) / (x.max() - x.min())


        #  2. Split into upper and lower using piecewise linear reference
        xu_raw, yu_raw, xl_raw, yl_raw = self.piecewise_reference_split(x, y)

        
        # Ensure LE (0,0) and TE (1,0) are included in both to avoid gaps
        # This prevents the 'ValueError' during spline interpolation later
        def force_endpoints(xs, ys):
            if 0.0 not in xs:
                xs, ys = np.append(xs, 0.0), np.append(ys, 0.0)
            if 1.0 not in xs:
                xs, ys = np.append(xs, 1.0), np.append(ys, 0.0)
            return xs, ys
        
        xu_raw, yu_raw = force_endpoints(xu_raw, yu_raw)
        xl_raw, yl_raw = force_endpoints(xl_raw, yl_raw)
        

        def make_strictly_monotonic(x_s, y_s):
            """Ensures x is strictly increasing by removing duplicates and sorting."""
            # Find unique x values and their first occurrences
            _, unique_indices = np.unique(x_s, return_index=True)
            # Sort indices to maintain general geometric order
            unique_indices = np.sort(unique_indices)
            x_clean = x_s[unique_indices]
            y_clean = y_s[unique_indices]
            
            # Final safety sort for CubicSpline
            sort_idx = np.argsort(x_clean)
            return x_clean[sort_idx], y_clean[sort_idx]

        # Apply the fixer
        xu, yu = make_strictly_monotonic(xu_raw, yu_raw)
        xl, yl = make_strictly_monotonic(xl_raw, yl_raw)

        
        # 3. Create Cosine Spacing
        theta = np.linspace(0, np.pi, n_points)
        x_new = 0.5 * (1 - np.cos(theta))
        
        # 4. Interpolate to Smooth Curves
        yu_new = CubicSpline(xu, yu)(x_new)
        yl_new = CubicSpline(xl, yl)(x_new)
        
        return x_new, yu_new, yl_new

    def get_bernstein_matrix(self, x, n):
        """Generates the Bernstein basis matrix for order n."""
        matrix = np.zeros((len(x), n + 1))
        for i in range(n + 1):
            matrix[:, i] = comb(n, i) * (x**i) * (1 - x)**(n - i)
        return matrix

    def fit_cst_to_coords(self, y_coords, n_order=8, n1=0.5):
        """
        Fits CST weights to a list of airfoil coordinates.
        Assumes points are normalized (0 to 1) and split into upper/lower.
        """
        # 1. Normalize and define Class Function
        psi = self.psi
        class_func = (psi**n1) * (1 - psi)**1.0
        
        # 2. Extract Shape Function values from target
        # S(psi) = [y(psi) - psi*y_te] / class_func(psi)
        y_te = y_coords[-1] # Trailing edge height

        shape_target = (y_coords - psi * y_te) / np.where(class_func == 0, 1e-12, class_func)        
        # 3. Build Linear System: B * A = Shape_Target
        B = self.get_bernstein_matrix(psi, n_order)
        
        # 4. Solve for weights A using Least Squares
        res = lsq_linear(B, shape_target)
        return res.x # These are your CST weights
    
    def align_and_scale(self, x, y):
        # 1. Translate LE to (0,0)
        le_idx = np.argmin(x)
        x -= x[le_idx]
        y -= y[le_idx]
        
        # 2. Rotate to put TE on x-axis
        te_idx = np.argmax(x)
        angle = np.arctan2(y[te_idx], x[te_idx])
        
        cos_a, sin_a = np.cos(-angle), np.sin(-angle)
        x_rot = x * cos_a - y * sin_a
        y_rot = x * sin_a + y * cos_a
        
        # 3. Final scale to chord = 1.0
        scale = x_rot[te_idx]
        return x_rot / scale, y_rot / scale
    
    def piecewise_reference_split(self, x_al, y_al):
        "" "Splits airfoil coordinates into upper and lower based on a piecewise linear reference line."""

        # 1. Calculate the mid-point reference
        mid_mask = (x_al > 0.4) & (x_al < 0.6)
        x_mid = 0.5
        y_mid = np.mean(y_al[mid_mask]) if np.any(mid_mask) else np.mean(y_al)
        
        # 2. Define the Reference Line Y(x)
        # y = m*x + c
        ref_y = np.zeros_like(x_al)
        
        # Segment 1: LE (0,0) to Mid (x_mid, y_mid)
        m1 = y_mid / x_mid
        mask1 = x_al <= x_mid
        ref_y[mask1] = m1 * x_al[mask1]
        
        # Segment 2: Mid (x_mid, y_mid) to TE (1,0)
        m2 = (0 - y_mid) / (1 - x_mid)
        c2 = y_mid - m2 * x_mid
        mask2 = x_al > x_mid
        ref_y[mask2] = m2 * x_al[mask2] + c2
        
        # 3. Split based on position relative to this 'Spine'
        upper_mask = y_al >= ref_y
        lower_mask = y_al < ref_y
        
        return x_al[upper_mask], y_al[upper_mask], x_al[lower_mask], y_al[lower_mask]
    
    def calculate_fit_error(self, y_raw, n1, n_order, lambda_reg=1e-8):
        """
        Calculates the error of a CST fit with a penalty for higher orders.
        lambda_reg: Controls how much we hate high complexity.
        """
        # 1. Perform the fit with the candidate N1 and order
        # Ensure this handles the N1 parameter correctly
        weights = self.fit_cst_to_coords(y_raw, n_order=n_order, n1=n1)
        
        # 2. Reconstruct the curve at target points
        # Normalize target x for the Class Function
        psi = self.psi 
        class_func = (psi**n1) * (1 - psi)**1.0
        B = self.get_bernstein_matrix(psi, n_order)
        y_fit = class_func * (B @ weights)
        
        # 3. Calculate Mean Squared Error (Geometric Accuracy)
        mse = np.mean((y_fit - y_raw)**2)
        
        # 4. Add the Complexity Penalty (L2-style on order)
        # We use n_order^2 to make the penalty accelerate at the high end (12-14)
        penalty = lambda_reg * (n_order**1.01)
        
        total_error = mse + penalty
        #print(f"N1: {n1:.4f}, Order: {n_order}, MSE: {mse:.2e}, Penalty: {penalty:.2e}, Total Error: {total_error:.2e}")
        return total_error

    def optimize_airfoil_parameterization(self,  y_raw):  
        best_err = float('inf')
        best_params = {'n1': 0.5, 'order': 8, 'weights': None}

        # Iterate through potential orders
        for order in range(2, 15): # 6 to 14
            # Optimize N1 for this specific order
            res = minimize_scalar(
                lambda n1: self.calculate_fit_error(y_raw, n1, order),
                bounds=(0.3, 0.6),
                method='bounded'
            )
            
            if res.fun < best_err:
                best_err = res.fun
                best_params['n1'] = res.x
                best_params['order'] = order
                # Store final weights for this best combo
                best_params['weights'] = self.fit_cst_to_coords(y_raw, n_order=order, n1=res.x)
                best_params['error'] = best_err
                
        return best_params
    
    def plot_fit(self):
        """Plots the original coordinates and the CST fit for comparison."""
        x_fit, y_fit_upper, y_fit_lower = self.calc_coords_from_params()

        
        plt.figure(figsize=(20, 16))
        plt.plot(x_fit, self.y_upper, 'ro', label='Original Coords Upper Error {error:.2e}'.format(error=self.params_upper['error']))
        plt.plot(x_fit, self.y_lower, 'mo', label='Original Coords Lower Error {error:.2e}'.format(error=self.params_lower['error']))

        plt.plot(x_fit, y_fit_upper, 'b-', label='CST Fit Upper')
        plt.plot(x_fit, y_fit_lower, 'g-', label='CST Fit Lower')
        plt.title(f"CST Airfoil Fit (Upper Order {self.params_upper['order']}  N1 {self.params_upper['n1']:.3f}\n Lower Order {self.params_lower['order']} N1 {self.params_lower['n1']:.3f})")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.legend()
        plt.grid()
        plt.show()  

    def calc_coords_from_params(self):
        """Calculates the airfoil coordinates from the fitted CST parameters."""
        psi = self.psi
        class_func_upper = (psi**self.params_upper['n1']) * (1 - psi)**1.0
        B_upper = self.get_bernstein_matrix(psi, self.params_upper['order'])
        shape_fit_upper = B_upper @ self.params_upper['weights']
        y_fit_upper = class_func_upper * shape_fit_upper + psi * self.y_upper[-1] # Add back TE contribution

        class_func_lower = (psi**self.params_lower['n1']) * (1 - psi)**1.0
        B_lower = self.get_bernstein_matrix(psi, self.params_lower['order'])
        shape_fit_lower = B_lower @ self.params_lower['weights']
        y_fit_lower = class_func_lower * shape_fit_lower + psi * self.y_lower[-1] # Add back TE contribution
        return psi, y_fit_upper, y_fit_lower


    def get_coords_sellig_format(self):
        """Returns the airfoil coordinates in Sellig format (TE to LE to TE)."""

        psi, y_fit_upper, y_fit_lower = self.calc_coords_from_params()
        # Combine upper and lower, ensuring TE is at the end
        x_combined = np.concatenate([psi[::-1], psi[1:]])  # TE to LE to TE
        y_combined = np.concatenate([y_fit_upper[::-1], y_fit_lower[1:]])
        coords = np.column_stack([x_combined, y_combined])
        return coords
    
    def get_resampled_spline_coords(self,x_query, le_idx):

        x_min = x_query[le_idx]
        x_max = x_query[np.argmax(x_query)]  

        x_upper = x_query[:le_idx]
        x_lower = x_query[le_idx:]
        print (f" Check LE position: {x_query[le_idx]}, Check TE position: {x_query[np.argmax(x_query)]}")
      
        y_fit_upper = self.get_upper_coords_at(x_upper, x_min, x_max)
        y_fit_lower = self.get_lower_coords_at(x_lower, x_min, x_max)

        return x_upper, y_fit_upper, x_lower, y_fit_lower
    


    def create_splinerep(self, x_te =2.0, x_le =-2.0):
       
        coords = self.get_coords_sellig_format()

        target_span = x_te - x_le

        coord_te = np.max(coords[:, 0]) 
        coord_le = np.min(coords[:, 0])
        coord_span = coord_te- coord_le

        scale = target_span / coord_span
        offset = x_te - coord_te
        print(f" Coords shape: {coords.shape}, Sample Coords: {coords.dtype}")
        print(f" Coords min x: {coord_le}, max x: {coord_te}")
        print(f" Scale: {scale}, Offset: {offset}"  )

        x = (coords[:, 0] ) * scale - x_te
        y = coords[:, 1]*scale

        print(f" After scaling, Coords min x: {np.min(x)}, max x: {np.max(x)}")
        self.b_spline = splprep([x,y], k=3, s=0, per=True)

        return self.b_spline
    
    def eval_spline_at_t(self, t):
        """Evaluate the spline representation at given parameter t."""
        if not hasattr(self, 'b_spline'):
            raise ValueError("Spline representation not created. Call create_splinerep() first.")
        
        x_eval, y_eval = splev(t, self.b_spline[0])
        return x_eval, y_eval
    
    def eval_spline_normal_at_t(self, t):
        """Evaluate the normal vectors of the spline at given parameter t."""
        if not hasattr(self, 'b_spline'):
            raise ValueError("Spline representation not created. Call create_splinerep() first.")
        
        dx, dy = splev(t, self.b_spline[0], der=1)
        tangents = np.column_stack([dx, dy])
        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        normals = np.column_stack([-dy / norms[:, 0], dx / norms[:, 0]])  # Rotate tangent by 90 degrees
        return normals
    
    def find_t_closest_xy(self, x, y, t_min, t_max):
        """
        Finds the B-spline parameter t that matches the 
        K-T theoretical nose (z_nose).
        """
        # Objective: Minimize the Euclidean distance between 
        # the B-spline point P(t) and the K-T nose (x_le_kt, y_le_kt)
        if not hasattr(self, 'b_spline'):
            raise ValueError("Spline representation not created. Call create_splinerep() first.")
               
        def objective(t):
            px, py = splev(t, self.b_spline[0])
            return np.sqrt((px - x)**2 + (py - y)**2)

        # Search in the middle of the loop (t ~ 0.5)
        res = minimize_scalar(objective, bounds=(t_min, t_max), method='bounded')
        print(f"Closest point on spline to ({x:.4f}, {y:.4f}) is at t={res.x:.4f} with distance {res.fun:.2e}")
        return res.x
    
    def snap_kt_to_bspline(self, kt_points,  t_init=0.0, t_delta=0.02, t_min=0, t_max=1.0):
        """
        Snaps K-T guide points to the B-spline along their normals.
        """
        snapped_points = []
        t_values = []

        t_guess = t_init  # Start with the first point's initial guess

        for i in range(len(kt_points)):
            # Objective: Find t that minimizes distance between B-spline P(t) 
            t_low = max(t_min, t_guess - t_delta)
            t_high = min(t_max, t_guess + t_delta)
            
            t_sol = self.find_t_closest_xy(kt_points[i].real, kt_points[i].imag, t_low, t_high)
            
            # Evaluate the spline at the found t
            px, py = splev(t_sol, self.b_spline[0])
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
        
        for i in range(len(kt_points)):
            p_kt = kt_points[i]
            n_kt = kt_normals[i] # Expecting [nx, ny]

            # Define the objective: Minimize the perpendicular distance 
            # from the spline point P(t) to the ray (p_kt + alpha * n_kt)
            def objective(t):
                px, py = splev(t, self.b_spline[0])
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
            px, py = splev(t_sol, self.b_spline[0])
            
            snapped_points.append(complex(px, py))
            t_values.append(t_sol)
            
            
        return np.array(snapped_points), t_values

    def get_upper_coords_at(self, x_query, x_min, x_max):
        """Get upper surface y-coordinates at specified x-coordinates."""

        print(f" Xmax: {x_max}, Xmin: {x_min}")

        scale = 1.0/(x_max - x_min)
        # normalize about [0-1]
        psi = (x_query - x_min) * scale

        class_func_upper = (psi**self.params_upper['n1']) * (1 - psi)**1.0
        B_upper = self.get_bernstein_matrix(psi, self.params_upper['order'])
        shape_fit_upper = B_upper @ self.params_upper['weights']
        y_fit_upper = class_func_upper * shape_fit_upper + psi * self.y_upper[-1] 

        # de-normalize back to original scale
        y_fit_upper = y_fit_upper/scale 
        print(f" Ymax: {np.max(y_fit_upper)}, Ymin: {np.min(y_fit_upper)}")

        return y_fit_upper
    
    def get_lower_coords_at(self, x_query,x_min,x_max):
        """Get lower surface y-coordinates at specified x-coordinates."""
        print(f" Xmax: {x_max}, Xmin: {x_min}")
        scale = 1.0/(x_max - x_min)

        # normalize about [0-1]
        psi = (x_query - x_min) * scale

        class_func_lower = (psi**self.params_lower['n1']) * (1 - psi)**1.0
        B_lower = self.get_bernstein_matrix(psi, self.params_lower['order'])
        shape_fit_lower = B_lower @ self.params_lower['weights']
        y_fit_lower = class_func_lower * shape_fit_lower + psi * self.y_lower[-1] 
        # de-normalize back to original scale
        y_fit_lower = y_fit_lower/scale 
        print(f" Ymax: {np.max(y_fit_lower)}, Ymin: {np.min(y_fit_lower)}")

        return y_fit_lower
    

    
    
def debug_mesh_connectivity(grid_coords, surface_points):
    # 1. Check Surface (Airfoil) level
    dist_surf = np.linalg.norm(surface_points[0] - surface_points[-1])
    print(f"--- Surface Connectivity (TE) ---")
    print(f"Point 0: {surface_points[0]}")
    print(f"Point N: {surface_points[-1]}")
    print(f"Distance: {dist_surf:.2e}")
    
    # 2. Check full Grid level (all layers)
    # Compare xi=0 and xi=N for all eta layers
    dist_layers = np.linalg.norm(grid_coords[0, :, :] - grid_coords[-1, :, :], axis=1)
    max_gap = np.max(dist_layers)
    mean_gap = np.mean(dist_layers)
    
    print(f"\n--- Grid Extrusion Connectivity ---")
    print(f"Max gap along wake: {max_gap:.2e}")
    print(f"Mean gap along wake: {mean_gap:.2e}")

    if max_gap > 1e-7:
        print("\n[!] WARNING: Mesh is DISCONNECTED at the Trailing Edge.")
        print("This will cause the Jacobian to explode or vanish at the seam.")
    else:
        print("\n[✓] SUCCESS: Mesh is physically connected.")


def generate_aligned_o_grid(surface_points, n_layers=40, expansion_ratio=1.02, R_far=8.0):
    n_pts = len(surface_points)
    # 1. Identify the TE angle to align the far-field circle
    te_angle = np.arctan2(surface_points[0, 1], surface_points[0, 0])
    
    grid_coords = np.zeros((n_pts, n_layers, 3))
    
    # 2. Use a stretch factor for radial layers (better for CFD/Potential Flow)
    # Using a geometric progression ensures more points near the airfoil
    indices = np.arange(n_layers)
    weights = (expansion_ratio**indices - 1) / (expansion_ratio**(n_layers - 1) - 1)
    
    for j in range(n_layers):
        w = weights[j] # Radial weight (0 at surface, 1 at far-field)
        
        # Create circle points starting EXACTLY at the TE angle
        # This prevents the 'twist' seen in your previous debug output
        theta = np.linspace(te_angle, te_angle + 2*np.pi, n_pts)
        circle_points = np.column_stack([np.cos(theta), np.sin(theta)]) * R_far
        
        # Blend the airfoil and the circle
        grid_coords[:, j, :2] = (1 - w) * surface_points + w * circle_points
        
        # 3. CRITICAL: Force the seam to be identical
        # This replaces the 7.81e-01 gap with 0.0
        grid_coords[-1, j, :2] = grid_coords[0, j, :2]
        
    return grid_coords


def sort_airfoil_to_sellig(coords):
    """
    Sort airfoil coordinates counter-clockwise starting from trailing edge.
    Preserves sharp trailing edge while using spline smoothing elsewhere.
    """
    if hasattr(coords, 'cpu'):  # Handle PyTorch tensors
        coords = coords.cpu().numpy()
    
    coords = np.array(coords)
    
    # Find trailing edge (point with maximum x) and leading edge (minimum x)
    te_idx = np.argmax(coords[:, 0])
    le_idx = np.argmin(coords[:, 0])
    te_point = coords[te_idx]
    le_point = coords[le_idx]
    
    # Separate into upper and lower surfaces based on y-coordinate relative to TE
    upper_surface = []
    lower_surface = []
    
    for i, point in enumerate(coords):
        if i == te_idx:
            continue  # Skip trailing edge for now
        
        if point[1] > te_point[1]:  # Above trailing edge
            upper_surface.append(i)
        else:  # Below or at trailing edge level
            lower_surface.append(i)
    
    # Identify very small LE region for special angular sorting
    chord_length = coords[:, 0].max() - coords[:, 0].min()
    le_region_size = 0.02 * chord_length  # Only 2% of chord
    le_threshold = le_point[0] + le_region_size
    
    # Separate points into LE region and far-field
    upper_le = [i for i in upper_surface if coords[i, 0] < le_threshold]
    upper_far = [i for i in upper_surface if coords[i, 0] >= le_threshold]
    lower_le = [i for i in lower_surface if coords[i, 0] < le_threshold]
    lower_far = [i for i in lower_surface if coords[i, 0] >= le_threshold]
    
    # Sort far-field regions by x-coordinate (proven method)
    upper_far = sorted(upper_far, key=lambda i: -coords[i, 0])  # TE to LE
    lower_far = sorted(lower_far, key=lambda i: coords[i, 0])   # LE to TE
    
    # For LE region, use arc length progression for smooth ordering
    def sort_by_arc_length(indices, start_point):
        """Sort points by cumulative arc length from start_point"""
        if not indices:
            return []
        
        points = coords[indices]
        # Calculate distances from start point
        distances = [np.linalg.norm(p - start_point) for p in points]
        
        # Sort by distance (closest first)
        sorted_pairs = sorted(zip(indices, distances), key=lambda x: x[1])
        
        # Build path by always choosing closest unvisited point
        path = [sorted_pairs[0][0]]  # Start with closest point
        remaining = [idx for idx, _ in sorted_pairs[1:]]
        
        while remaining:
            current_point = coords[path[-1]]
            # Find closest remaining point
            distances_to_current = [(idx, np.linalg.norm(coords[idx] - current_point)) 
                                  for idx in remaining]
            next_idx, _ = min(distances_to_current, key=lambda x: x[1])
            path.append(next_idx)
            remaining.remove(next_idx)
            
        return path
    
    # Sort LE regions by arc length progression
    if upper_le:
        # Start from the point closest to where upper far-field ends
        if upper_far:
            start_point = coords[upper_far[-1]]  # Last point of upper far-field
        else:
            start_point = te_point
        upper_le = sort_by_arc_length(upper_le, start_point)
        
    if lower_le:
        # Start from the leading edge point itself
        lower_le = sort_by_arc_length(lower_le, le_point)
    
    # Combine: TE -> upper far -> upper LE -> lower LE -> lower far
    ordered_indices = [te_idx] + upper_far + upper_le + lower_le + lower_far
    rough_coords = coords[ordered_indices]
    
    # Downsample if we have too many points (>500 is excessive for Arbifoil)
    if len(rough_coords) > 500:
        # Find the true leading edge point - handle multiple points at min x-coordinate
        min_x = np.min(rough_coords[:, 0])
        x_tolerance = 1e-6  # Small tolerance for floating point comparison
        le_candidates = np.where(np.abs(rough_coords[:, 0] - min_x) <= x_tolerance)[0]
        
        # Debug: Print all LE candidate points
        print(f"\nLE Debug: Found {len(le_candidates)} candidate points at min_x = {min_x:.8f}")
        for i, idx in enumerate(le_candidates):
            x, y = rough_coords[idx, 0], rough_coords[idx, 1]
            print(f"  Candidate {i}: idx={idx}, coord=({x:.8f}, {y:.8f})")
        
        if len(le_candidates) > 1:
            # Multiple points at min x - choose the one closest to y=0 (geometric center)
            le_y_distances = np.abs(rough_coords[le_candidates, 1])
            best_candidate_rel_idx = np.argmin(le_y_distances)
            le_rough_idx = le_candidates[best_candidate_rel_idx]
            print(f"  Selected candidate {best_candidate_rel_idx} (idx={le_rough_idx}) as closest to y=0")
        else:
            le_rough_idx = le_candidates[0]
            print(f"  Single candidate selected: idx={le_rough_idx}")
        
        # Show points around the LE region for context
        le_vicinity = 10  # Show ±10 points around LE
        start_idx = max(0, le_rough_idx - le_vicinity)
        end_idx = min(len(rough_coords), le_rough_idx + le_vicinity + 1)
        print(f"\nPoints around LE (idx {start_idx} to {end_idx-1}):")
        for i in range(start_idx, end_idx):
            x, y = rough_coords[i, 0], rough_coords[i, 1]
            marker = " <-- LE" if i == le_rough_idx else ""
            print(f"  [{i:3d}]: ({x:.6f}, {y:.6f}){marker}")
        
        # Keep every other point to reduce density
        step = len(rough_coords) // 400  # Target ~400 points to leave room for critical points
        step = max(2, step)  # At least every other point
        downsampled_indices = list(range(0, len(rough_coords), step))
        
        # Always preserve critical points: TE (first), LE, and last point
        critical_indices = {0, le_rough_idx, len(rough_coords) - 1}
        
        # Add critical indices that might be missing
        for idx in critical_indices:
            if idx not in downsampled_indices:
                downsampled_indices.append(idx)
        
        # Sort indices to maintain order
        downsampled_indices = sorted(set(downsampled_indices))
        
        rough_coords = rough_coords[downsampled_indices]
        print(f"Downsampled from {len(ordered_indices)} to {len(rough_coords)} points")
        print(f"Preserved LE at idx {le_rough_idx}, coord: ({rough_coords[downsampled_indices.index(le_rough_idx), 0]:.6f}, {rough_coords[downsampled_indices.index(le_rough_idx), 1]:.6f})")
        if len(le_candidates) > 1:
            print(f"Found {len(le_candidates)} candidate LE points, selected most centered")
    
    # Apply spline smoothing only to the middle sections, preserving TE sharpness
    try:
        # Find points that are NOT near the trailing edge
        x_range = coords[:, 0].max() - coords[:, 0].min()
        te_threshold = coords[:, 0].max() - 0.05 * x_range  # Within 5% of TE
        
        # Identify TE region points to preserve
        te_region_mask = rough_coords[:, 0] >= te_threshold
        
        if np.sum(~te_region_mask) > 10:  # Enough points for spline fitting
            # Split into sections: TE region, middle section, back to TE region
            te_start_indices = np.where(te_region_mask)[0]
            middle_indices = np.where(~te_region_mask)[0]
            
            if len(middle_indices) > 0:
                # Get the middle section for spline fitting
                middle_start = middle_indices[0]
                middle_end = middle_indices[-1]
                
                middle_coords = rough_coords[middle_start:middle_end+1]
                
                # Fit spline only to the middle section (not closed)
                tck, u = splprep([middle_coords[:, 0], middle_coords[:, 1]], 
                                s=0, per=False)  # Not periodic
                
                # Resample middle section
                n_middle = len(middle_coords)
                u_new = np.linspace(0, 1, n_middle)
                x_smooth, y_smooth = splev(u_new, tck)
                smooth_middle = np.column_stack([x_smooth, y_smooth])
                
                # Reconstruct: preserve TE regions, smooth middle
                final_coords = np.copy(rough_coords)
                final_coords[middle_start:middle_end+1] = smooth_middle
                
                return final_coords
        
        # Fallback: just return the rough coordinates with proper ordering
        return rough_coords
        
    except Exception as e:
        print(f"Partial spline fitting failed: {e}, returning rough coordinates")
        return rough_coords


def get_smoothed_airfoil_spline( coords, n_target=512):
        """
        Creates a periodic cubic spline of the airfoil and resamples it.
        z_points: Complex numpy array or tensor of coordinates.
        n_target: Target number of points for the Fornberg iteration.
        """


        # 1. Calculate cumulative arc length for the parameter 's'
        dx = np.diff(coords[:,0])
        dy = np.diff(coords[:,1])
        ds = np.sqrt(dx**2 + dy**2)
        s = np.concatenate(([0], np.cumsum(ds)))
        
        # 2. Fit periodic cubic splines for x and y separately
        # bc_type='periodic' ensures smooth closure at the trailing edge
        cs_x = CubicSpline(s, coords[:,0], bc_type='periodic')
        cs_y = CubicSpline(s, coords[:,1], bc_type='periodic')
        
        # 3. Resample onto a uniform arc-length grid
        s_new = np.linspace(0, s[-1], n_target, endpoint=True)
        x_new = cs_x(s_new)
        y_new = cs_y(s_new)

        return np.column_stack([x_new, y_new])