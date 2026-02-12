

from math import *
from pathlib import Path
import numpy as np
from sympy import zeta
import torch
import airfrans as af
import matplotlib.pyplot as plt
import sys
from scipy.interpolate import splprep, splev
from scipy.interpolate import CubicSpline, interp1d

import torch

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


class AirfoilTransformer:
    def __init__(self,coords):
        # coords should start and end at trailing edge rotating counter-clockwise
        # ensure closed loop 
        if not np.allclose(coords[-1,:], coords[0,:], atol=1e-6):
            print("Closing airfoil loop by adding trailing edge point at the end.")
            coords = np.vstack([coords, coords[0,:]])  
        
        
        # scale such that the airfoil spans [-2,2] in x-direction for better numerical stability
        x_span = coords[:,0].max() - coords[:,0].min()
        scale_factor = 4.0 / x_span
        print(f"Airfoil x-span before scaling: {x_span:.6f}")
        print(f" Scaling by {scale_factor:.6f} to achieve span of 4.0 units.")

        coords = coords * scale_factor
        print( f"Airfoil coordinates after scaling: x range [{coords[:,0].min():.6f}, {coords[:,0].max():.6f}], y range [{coords[:,1].min():.6f}, {coords[:,1].max():.6f}]")

        # move xmin to -2.0
        coords[:,0] = coords[:,0] - coords[:,0].max()/2
        print( f"Airfoil coordinates after translation: x range [{coords[:,0].min():.6f}, {coords[:,0].max():.6f}], y range [{coords[:,1].min():.6f}, {coords[:,1].max():.6f}]")

        self.coords = coords
       
        # convert to complex numbers
        self.z1 = torch.tensor(self.coords[:,0], dtype=torch.float32) + 1j * torch.tensor(self.coords[:,1], dtype=torch.float32)

        print(f"Airfoil complex coordinates range: Re({self.z1.real.min():.6f} to {self.z1.real.max():.6f}), Im({self.z1.imag.min():.6f} to {self.z1.imag.max():.6f})"  )


        #self.z2 = self.inverse_karman_trefftz(torch.tensor(self.z1, dtype=torch.cfloat), te_angle_deg=5.0)

        self.z2 = self.inverse_joukowsky(torch.tensor(self.z1, dtype=torch.cfloat))

        # shift z2 to be centered around the origin for better numerical stability
        self.centroid_z2 = torch.mean(self.z2)


        self.z2 = self.z2 - self.centroid_z2
        print(f"Transformed airfoil centroid after shifting: ({torch.mean(self.z2.real):.6f}, {torch.mean(self.z2.imag):.6f})")

        
        z2_new , e_new = self.get_smoothed_airfoil_spline( self.z2, n_target=4096)

        self.plot_spline_comparison(self.z2.cpu().numpy(), z2_new)

        # Force  to optimal origin to ensure better convergence in Fornberg iteration

        self.z2_origin_optimal = self.find_optimal_origin(z2_new)
        print(f"Optimal origin found at: ({self.z2_origin_optimal.real:.6f}, {self.z2_origin_optimal.imag:.6f})")   
        z2_new = z2_new - self.z2_origin_optimal

        self.z2 = z2_new.to(self.z2.device)

        self.plot_tangent_periodicity(e_new)

        #self.z2 = self.get_smoothed_airfoil_spline(self.z2.cpu().
        # do the fornberg iteration to get the Taylor coefficients
        self.coeffs, zeta_final = self.fornberg_outer_iteration(self.z2, n_iter=100, tol=1e-12)

        self.validate_mapping(zeta_final, self.coeffs)
        # truncate to 256 coefficients for use in the transformer model
        trunc_coeffs = 1024
        self.plot_truncated_reconstruction(trunc_coeffs)


    def fornberg_outer_iteration(self, zeta, n_iter=15, tol=1e-10):
        """
        zeta: Complex torch tensor [N] representing the closed z2 near-circle.
        n_iter: Max number of quadratically convergent outer steps.
        """
        N = zeta.shape[0]
        device = zeta.device
        
        # Precompute roots of unity w for Eq. 12b
        k_idx = torch.arange(N, device=device)
        w = torch.exp(-2j * torch.pi * k_idx / N) 

        for i in range(n_iter):
            # 1. Calculate current Taylor/Laurent coefficients (Eq. 10)
            # One FFT over N points
            d = torch.fft.fft(zeta) / N
            
            # Check residual: we want d_nu for nu <= 0 to be zero
            # In FFT output, nu <= 0 corresponds to indices [0] and [N/2+1:]
            res_vector = torch.cat([d[0:1], d[N//2+1:]])
            max_res = torch.max(torch.abs(res_vector))
            
            if max_res < tol:
                print(f"Converged at iteration {i}, residual: {max_res:.2e}")
                break

            # 2. Tangential directions e_k (Eq. 33)
            # Central difference for tangents on the closed curve
            diff = torch.roll(zeta, -1) - torch.roll(zeta, 1)
            e = diff / torch.abs(diff)

            # 3. Solve for tangential distances l_k (Eq. 12a/b)
            # For near-circles, we can use the Conjugate Gradient (CG) method
            # This solves G * l = b in O(N log N)
            l = self.solve_inner_cg(zeta, e, d, N)

            # add damping
            omega = 0.2

            l_damp = omega * l
            maxstep = 0.001
            l_clamp = torch.clamp(l_damp, -maxstep, maxstep)  # Limit max tangential step to prevent instability

            # 4. Move points and project back to curve J (Eq. 11, 34)
            # This provides the quadratic convergence
            zeta_new = zeta +  l_clamp* e
            # Inside fornberg_outer_iteration, after zeta_new = zeta + l * e:

            # Calculate distance from origin before snapping
            dist_before = torch.abs(zeta_new)

            # Apply the snap
            zeta_snapped = self.project_to_near_circle(zeta_new, self.z2)

            # Calculate distance from origin after snapping
            dist_after = torch.abs(zeta_snapped)

            # Validation Metrics
            # 1. The 'drift' caused by the tangential step
            drift = torch.mean(torch.abs(zeta_new - zeta))
            # 2. The correction magnitude
            correction = torch.mean(torch.abs(zeta_snapped - zeta_new))


            # Critical Check: Is the snapped point actually on the target boundary?
            # We compare the snapped radius to the reference boundary radius at that angle
            ref_angles = torch.angle(self.z2)
            ref_radii = torch.abs(self.z2)
            target_radius_at_angle = self.interpolate_radii(torch.angle(zeta_snapped), ref_angles, ref_radii)
            residual_radial_error = torch.max(torch.abs(dist_after - target_radius_at_angle))

            #print(f"   Radial residual after snap: {residual_radial_error:.2e}")


            zeta = self.project_to_near_circle(zeta_new, zeta) # Newton-like projection

        print(f" Final  Outer Iter {i}: Avg Drift {drift:.2e}, Avg Snap Correction {correction:.2e}")

            
        # Final Taylor coefficients
        final_coeffs = torch.fft.fft(zeta) / N
        return final_coeffs[:N//2], zeta

    def solve_inner_cg(self, zeta, e, d, N, cg_tol=1e-6):
        """
        Solves G*l = b using the conjugate gradient method.
        """
        # 1. Identify negative/zero frequency indices for analyticity
        neg_idx = torch.cat([torch.tensor([0], device=zeta.device), 
                            torch.arange(N//2 + 1, N, device=zeta.device)])
        
        # 2. Construct b: The 'target' tangential movement to cancel residual d
        b_buffer = torch.zeros(N, dtype=torch.complex64, device=zeta.device)
        b_buffer[neg_idx] = -d[neg_idx] 
        # Project back to real tangential space
        b = (torch.fft.ifft(b_buffer) * N * e.conj()).real
        
        l = torch.zeros(N, device=zeta.device)
        
        # 3. CG Iteration Loop
        r = b - self.apply_G_operator(l, e, N) # zeta is removed here
        p = r.clone()
        
        # Near-circular shapes converge in about 6 steps
        for i in range(12): 
            Gp = self.apply_G_operator(p, e, N)
            
            # Use torch.sum for the dot product of real vectors
            r_sq = torch.sum(r * r)
            alpha = r_sq / torch.sum(p * Gp)
            
            l = l + alpha * p
            r_new = r - alpha * Gp
            
            if torch.sqrt(torch.sum(r_new * r_new)) < cg_tol:
                break
                
            beta = torch.sum(r_new * r_new) / r_sq
            p = r_new + beta * p
            r = r_new
            #print(f"   CG Iter {i+1}, residual norm: {torch.sqrt(torch.sum(r*r)):.2e}")
                
        return l

    def apply_G_operator(self, l, e, N):
        """
        G = I - A, where A = R^T * R.
        The operator represents the projection of the tangential movement onto 
        the analyticity-violating subspace.
        """
        # Map tangential movement to coefficient space
        # We divide by N here because torch.fft.fft is unscaled
        le_fft = torch.fft.fft(l * e) / N
        
        # Constraint indices (nu <= 0)
        neg_idx = torch.cat([torch.tensor([0], device=l.device), 
                            torch.arange(N//2 + 1, N, device=l.device)])
        
        # R * l: Extract current analyticity violations
        Cl_neg = le_fft[neg_idx] 
        
        # R^T * (R * l): Project back to tangential space
        rt_buffer = torch.zeros(N, dtype=torch.complex64, device=l.device)
        rt_buffer[neg_idx] = Cl_neg 
        
        # Inverse FFT brings it back to the spatial domain
        RtRl_complex = torch.fft.ifft(rt_buffer) * N
        
        # Final projection to real tangential distance
        At_l = (RtRl_complex * e.conj()).real
        
        return l - At_l
            
    def inverse_karman_trefftz(self, z1, te_angle_deg=10.0):
        """
        Inverse Karman-Trefftz to map airfoil to near-circle.
        z1: Complex torch tensor of airfoil coordinates.
        """
        # 1. Define parameters
        tau = torch.tensor(te_angle_deg * torch.pi / 180.0, device=z1.device)
        k = 2.0 - (tau / torch.pi)
        
        # l is half-chord, typically 1.0 if airfoil is scaled to [-2, 2]
        l = 1.0 
        
        # 2. Solve for the intermediate ratio
        # R = (z1 - kl) / (z1 + kl)
        R = (z1 - k * l) / (z1 + k * l)
        
        # 3. Apply the root: (z2 - l) / (z2 + l) = R^(1/k)
        # This 'unfolds' the trailing edge angle
        zeta = R ** (1.0 / k)
        
        # 4. Solve for z2: z2 = l * (1 + zeta) / (1 - zeta)
        z2 = l * (1.0 + zeta) / (1.0 - zeta)
        
        return z2

    def inverse_joukowsky(self,z1):
        """
        Maps airfoil coordinates (z1) to a near-circle (z2).
        z1: Complex torch tensor [N_points]
        """
        # Standard quadratic solution
        # We use z1 / 2 to simplify the sqrt( (z1/2)^2 - 1 )
        term = z1 / 2.0
        sqrt_term = torch.sqrt(term**2 - 1.0)
        
        # Calculate both potential branches
        z2_plus = term + sqrt_term
        z2_minus = term - sqrt_term
        
        # Select the branch that lies on/outside the circle (|z| >= 1)
        # This prevents the airfoil from mapping inside the circle
        
        z2 = torch.where(torch.abs(z2_plus) >= 1.0, z2_plus, z2_minus)
        
        return z2
        

    def project_to_near_circle(self, zeta_moved, zeta_target_boundary):
        """
        Snaps points back to the boundary J using a Newton step.
        
        zeta_moved: The points after the tangential step (zeta + l*e).
        zeta_target_boundary: The original near-circle points used to define J.
        """
        # 1. We treat the original 'zeta_target_boundary' as a reference spline or polar curve
        # For airfoils, a polar distance check is often sufficient.
        target_angles = torch.angle(zeta_target_boundary)
        target_radii = torch.abs(zeta_target_boundary)
        
        # 2. Get the current polar coordinates of the moved points
        moved_angles = torch.angle(zeta_moved)
        
        # 3. Interpolate the target radius at the new angles
        # This acts as the f(x,y)=0 check from Fornberg Section 5.
        # We use linear interpolation here for efficiency on the GPU.
        new_radii = self.interpolate_radii(moved_angles, target_angles, target_radii)
        
        # 4. Snap the point back: zeta_projected = new_radius * exp(i * moved_angle)
        # This keeps the 'theta' correspondence established in the outer iteration.
        zeta_projected = torch.polar(new_radii, moved_angles)
        
        return zeta_projected

    def find_optimal_origin(self,zeta):
        """
        Calculates the area-weighted centroid of the polygon defined by zeta.
        This provides a more 'economical' origin for the Taylor series.
        """
        x = zeta.real
        y = zeta.imag
        
        # Shoelace formula for area
        area = 0.5 * torch.sum(x * torch.roll(y, -1) - torch.roll(x, -1) * y)
        
        # Centroid coordinates
        cx = torch.sum((x + torch.roll(x, -1)) * (x * torch.roll(y, -1) - torch.roll(x, -1) * y)) / (6.0 * area)
        cy = torch.sum((y + torch.roll(y, -1)) * (x * torch.roll(y, -1) - torch.roll(x, -1) * y)) / (6.0 * area)
        
        return cx + 1j * cy



    def interpolate_radii(self,query_angles, ref_angles, ref_radii):
        """
        Linear interpolation helper to find the 'true' boundary radius at any angle.
        Ensures points stay on the J curve.
        """
        # Normalize angles to [0, 2pi]
        query_angles = query_angles % (2 * torch.pi)
        ref_angles = ref_angles % (2 * torch.pi)
        
        # Sort reference angles to allow searchsorted
        idx = torch.argsort(ref_angles)
        ref_angles, ref_radii = ref_angles[idx], ref_radii[idx]
        
        # Find neighbors for interpolation
        upper_idx = torch.searchsorted(ref_angles, query_angles)
        lower_idx = (upper_idx - 1) % len(ref_angles)
        upper_idx = upper_idx % len(ref_angles)
        
        # Linear interpolation factor
        d_theta = (ref_angles[upper_idx] - ref_angles[lower_idx]) % (2 * torch.pi)
        weight = ((query_angles - ref_angles[lower_idx]) % (2 * torch.pi)) / d_theta
        
        return (1 - weight) * ref_radii[lower_idx] + weight * ref_radii[upper_idx]

    def plot_airfoil(self):

        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(8,4))
        ax1.plot(self.coords[:,0], self.coords[:,1], 'k-', label='Airfoil')
        ax1.axis('equal')
        ax1.set_title('Airfoil Geometry')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.grid(True)
        ax1.legend()
        ax2.scatter(self.z2.real.numpy(), self.z2.imag.numpy(), color='red', label='Centroid')
        ax2.axis('equal')
        ax2.set_xlabel('eta')
        ax2.set_ylabel('psi')

        plt.show()


    def plot_joukowsky_mapping(self):
        plt.figure(figsize=(8, 8))

        # 1. Plot the resulting near-circle (z2 plane)
        plt.plot(self.z2.real.numpy(), self.z2.imag.numpy(), 'r-', label='Near-Circle ($z_2$)')
        plt.fill(self.z2.real.numpy(), self.z2.imag.numpy(), 'r', alpha=0.1)

        # 2. Plot a reference unit circle for comparison
        theta = np.linspace(0, 2*np.pi, 200)
        plt.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5, label='Unit Circle ($|z|=1$)')

        plt.title("Inverse Joukowsky: Airfoil $\\rightarrow$ Near-Circle")
        plt.xlabel("Re($z_2$)")
        plt.ylabel("Im($z_2$)")
        plt.legend()
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_truncated_reconstruction(self, n_coeffs=256):
        """
        Reconstructs the airfoil using only the first M coefficients.
        Useful for validating the 'economy' of the mapping for GINO training.
        """
        N = 4096  # Using your successful high-res count
        device = self.coeffs.device
        
        # 1. Truncate and Pad
        # We take the first n_coeffs and zero out the rest of the spectrum
        truncated_coeffs = torch.zeros(N, dtype=torch.complex64, device=device)
        # coeffs[0] is the DC/constant term (the shifted origin)
        truncated_coeffs[:n_coeffs] = self.coeffs[:n_coeffs]
        
        # 2. Inverse FFT to get reconstructed points in the near-circle z2 plane
        # Scaling by N because torch.fft.ifft is 1/N by default
        z2_recon = torch.fft.ifft(truncated_coeffs) * N
        
        # 3. Shift back to the original Joukowsky frame if you shifted the origin
        # (Assuming you stored the shift value in self.origin_shift)
        z2_recon = z2_recon + self.z2_origin_optimal

        # Shift back to original centroid 
        z2_recon = z2_recon + self.centroid_z2  
        
        # 4. Forward Joukowsky Transform to get back to physical airfoil coordinates
        # z1 = z2 + 1/z2
        z1_recon = z2_recon + (1.0 / z2_recon)
        
        # Create a parameter t for the original 508 points
        t_orig = np.linspace(0, 1, len(self.z1))
        t_new = np.linspace(0, 1, N)
        
        z1_raw_np = self.z1.detach().cpu().numpy()

        # Interpolate real and imag parts separately to 4096 points
        f_real = interp1d(t_orig, z1_raw_np.real, kind='cubic')
        f_imag = interp1d(t_orig, z1_raw_np.imag, kind='cubic')
        z1_target_np = f_real(t_new) + 1j * f_imag(t_new)


        z1_recon_np = z1_recon.detach().cpu().numpy()
        
        plt.figure(figsize=(12, 5))
        plt.plot(z1_target_np.real, z1_target_np.imag, 'k-', alpha=0.4, label='Original Airfoil (High Res)')
        plt.plot(z1_recon_np.real, z1_recon_np.imag, 'r--', label=f'Reconstructed ({n_coeffs} coeffs)')
        
        # Zoom in on the Leading Edge to check for 'smoothing'
        plt.title(f"Reconstruction Fidelity: M={n_coeffs} vs N={N}")
        plt.axis('equal')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Calculate error in physical space
        # We interpolate to align points for a direct comparison
        phys_error = np.max(np.abs(z1_target_np - z1_recon_np))
        print(f"Max Physical Reconstruction Error (M={n_coeffs}): {phys_error:.2e}")

    def validate_mapping(self, zeta_final, c_nu):
        N = len(zeta_final)
        d = torch.fft.fft(zeta_final) / N
        
        # Check 1: Analyticity Residual
        residual = torch.max(torch.abs(torch.cat([d[0:1], d[N//2+1:]])))
        print(f"Analyticity Residual: {residual:.2e}")

        # Check 3: Reconstruction Fidelity
        # Ensure c_nu includes the DC component if the origin was shifted
        padded_coeffs = torch.zeros(N, dtype=torch.complex64, device=zeta_final.device)
        padded_coeffs[:len(c_nu)] = c_nu
        zeta_recon = torch.fft.ifft(padded_coeffs) * N
        
        error = torch.max(torch.abs(zeta_final - zeta_recon))
        print(f"Reconstruction Max Error: {error:.2e}")

        plt.semilogy(torch.abs(c_nu).cpu())
        plt.title("Taylor Coefficient Decay")
        plt.ylabel("|c_nu|")
        plt.show()

    def get_smoothed_airfoil_spline(self, z_points, n_target=512):
        """
        Creates a periodic cubic spline of the airfoil and resamples it.
        z_points: Complex numpy array or tensor of coordinates.
        n_target: Target number of points for the Fornberg iteration.
        """
        if torch.is_tensor(z_points):
            z_points = z_points.detach().cpu().numpy()

        # 1. Calculate cumulative arc length for the parameter 's'
        dx = np.diff(z_points.real)
        dy = np.diff(z_points.imag)
        ds = np.sqrt(dx**2 + dy**2)
        s = np.concatenate(([0], np.cumsum(ds)))
        
        # 2. Fit periodic cubic splines for x and y separately
        # bc_type='periodic' ensures smooth closure at the trailing edge
        cs_x = CubicSpline(s, z_points.real, bc_type='periodic')
        cs_y = CubicSpline(s, z_points.imag, bc_type='periodic')
        
        # 3. Resample onto a uniform arc-length grid
        s_new = np.linspace(0, s[-1], n_target, endpoint=False)
        x_new = cs_x(s_new)
        y_new = cs_y(s_new)
        
        # 4. Calculate analytic tangents (e_k) using derivatives
        # Tangent vector e = (dx/ds + i*dy/ds) / magnitude
        dx_ds = cs_x(s_new, 1)
        dy_ds = cs_y(s_new, 1)
        mag = np.sqrt(dx_ds**2 + dy_ds**2)
        
        e_new = (dx_ds + 1j * dy_ds) / mag
        z_new = x_new + 1j * y_new

        return torch.tensor(z_new, dtype=torch.cfloat), torch.tensor(e_new, dtype=torch.cfloat)

    def plot_spline_comparison(self, z_orig, z_spline):
        # Ensure both are numpy for matplotlib
        if torch.is_tensor(z_orig):
            z_orig = z_orig.detach().cpu().numpy()
        if torch.is_tensor(z_spline):
            z_spline = z_spline.detach().cpu().numpy()

        plt.figure(figsize=(10, 10))
        
        # Plot original faint dots
        plt.scatter(z_orig.real, z_orig.imag, color='gray', alpha=0.3, s=10, label='Original')
        
        # Plot the smooth spline curve
        plt.plot(z_spline.real, z_spline.imag, 'r-', linewidth=1, alpha=0.7, label='Cubic Spline')
        plt.scatter(z_spline.real, z_spline.imag, color='red', s=15)
        
        plt.axis('equal')
        plt.legend()
        plt.show()

        plt.show()

    def plot_tangent_periodicity(self, tangents):
        """
        Plots the components of the complex tangent vector e.
        tangents: Complex torch tensor [N]
        """
        e_np = tangents.detach().cpu().numpy()
        N = len(e_np)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Real part of tangent (dx/ds)
        ax1.plot(e_np.real, 'b-', label='Re(e) - Tangent X-component')
        # Plot 'wrap-around' indicators
        ax1.scatter([0, N-1], [e_np[0].real, e_np[-1].real], color='red', zorder=5)
        ax1.set_ylabel("dx/ds")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Imaginary part of tangent (dy/ds)
        ax2.plot(e_np.imag, 'g-', label='Im(e) - Tangent Y-component')
        ax2.scatter([0, N-1], [e_np[0].imag, e_np[-1].imag], color='red', zorder=5)
        ax2.set_ylabel("dy/ds")
        ax2.set_xlabel("Point Index (0 to N-1)")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.suptitle("Tangent Continuity Check (Periodic Boundary)")
        plt.show()

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


        foil_mapped = AirfoilTransformer(sorted_foil_points)
        foil_mapped.plot_airfoil()
        foil_mapped.plot_joukowsky_mapping()


        #except Exception as e:
        #    print(f"Error processing {sim_dir}: {e}")
        #    continue





if __name__ == "__main__":

    PATH_TO_DATASET = "/home/timm/Projects/PIML/Dataset"
    PATH_TO_OUTPUT = "/home/timm/Projects/PIML/Dataset_Cyl_Pot_FNO_X5Y4"
    GRID_SIZE = (64, 64)
    REGENERATE = True  # Set to True to regenerate all pt files
    xlen = float(6.0)   # domain length to be sampled  (-xlen/2, xlen/2)
    ylen = float(3.0)   # domain height to be sampled  (-ylen/2, ylen/2)
    xoffset = float(1.0)  # x-offset to be sampled

    convert(PATH_TO_DATASET, PATH_TO_OUTPUT, xlen, ylen, xoffset, GRID_SIZE)