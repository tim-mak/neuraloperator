import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.insert(0, '/home/timm/Projects/PIML/arbifoil')
import arbifoil

def load_naca0012_reference():
    """Load a known good NACA0012 airfoil for comparison"""
    naca_path = '/home/timm/Projects/PIML/arbifoil/airfoils/naca0012.txt'
    return np.loadtxt(naca_path)

def sort_airfoil_for_theodorsen(coords):
    """
    Sort airfoil coordinates for proper Theodorsen mapping
    
    The Theodorsen method expects:
    1. Starting at trailing edge (x=1, y=0)
    2. Counter-clockwise progression 
    3. Parameter θ goes from 0 to 2π
    4. No duplicate points
    """
    coords = np.asarray(coords)
    
    # Find trailing edge (maximum x coordinate)
    te_idx = np.argmax(coords[:, 0])
    te_point = coords[te_idx]
    
    # Find leading edge (minimum x coordinate) 
    le_idx = np.argmin(coords[:, 0])
    le_point = coords[le_idx]
    
    print(f"TE point: ({te_point[0]:.6f}, {te_point[1]:.6f}) at index {te_idx}")
    print(f"LE point: ({le_point[0]:.6f}, {le_point[1]:.6f}) at index {le_idx}")
    
    # Calculate angles from trailing edge to all other points
    angles = np.arctan2(coords[:, 1] - te_point[1], coords[:, 0] - te_point[0])
    
    # Adjust angles to be in [0, 2π] range, starting from TE
    angles = np.mod(angles, 2*np.pi)
    
    # Sort by angle (counter-clockwise from TE)
    sorted_indices = np.argsort(angles)
    sorted_coords = coords[sorted_indices]
    sorted_angles = angles[sorted_indices]
    
    # Remove duplicate trailing edge if present at the end
    if np.allclose(sorted_coords[0], sorted_coords[-1], atol=1e-6):
        sorted_coords = sorted_coords[:-1]
        sorted_angles = sorted_angles[:-1]
    
    print(f"Sorted {len(sorted_coords)} points")
    print(f"First point: ({sorted_coords[0][0]:.6f}, {sorted_coords[0][1]:.6f})")
    print(f"Last point: ({sorted_coords[-1][0]:.6f}, {sorted_coords[-1][1]:.6f})")
    
    # Verify the sorting is reasonable
    verify_sorting(sorted_coords, sorted_angles)
    
    return sorted_coords

def verify_sorting(coords, angles):
    """Verify the coordinate sorting is reasonable"""
    
    # Check if angles are monotonically increasing
    angle_diffs = np.diff(angles)
    if np.any(angle_diffs < -np.pi):  # Handle wrap-around
        angle_diffs[angle_diffs < -np.pi] += 2*np.pi
    
    if np.all(angle_diffs >= 0):
        print("✓ Angles are monotonically increasing (counter-clockwise)")
    else:
        print("⚠ Warning: Angles are not monotonic!")
        
    # Check distance between consecutive points
    distances = np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1))
    max_dist = np.max(distances)
    mean_dist = np.mean(distances)
    
    print(f"Point spacing - Mean: {mean_dist:.6f}, Max: {max_dist:.6f}")
    
    if max_dist > 5 * mean_dist:
        print("⚠ Warning: Large gaps between consecutive points!")
        large_gaps = np.where(distances > 3 * mean_dist)[0]
        for gap_idx in large_gaps[:5]:  # Show first 5
            print(f"  Large gap at index {gap_idx}: {distances[gap_idx]:.6f}")
    
def test_theodorsen_with_naca():
    """Test Theodorsen mapping with known good NACA0012"""
    print("Testing Theodorsen mapping with NACA0012...")
    
    # Load and sort NACA0012
    naca_coords = load_naca0012_reference()
    sorted_coords = sort_airfoil_for_theodorsen(naca_coords)
    
    # Create arbifoil object 
    try:
        foil = arbifoil.foil(datFileName=None, coords=sorted_coords)
        print("✓ Theodorsen mapping successful!")
        
        # Get coefficients
        theta, psi, eta = foil.getCoeffs()
        
        print(f"Coefficient ranges:")
        print(f"  theta: [{theta.min():.4f}, {theta.max():.4f}]")
        print(f"  psi: [{psi.min():.4f}, {psi.max():.4f}]") 
        print(f"  eta: [{eta.min():.4f}, {eta.max():.4f}]")
        
        # Check for discontinuities
        te_jump_psi = abs(psi[0] - psi[-1])
        te_jump_eta = abs(eta[0] - eta[-1])
        print(f"Trailing edge discontinuities - psi: {te_jump_psi:.6f}, eta: {te_jump_eta:.6f}")
        
        return foil, theta, psi, eta
        
    except Exception as e:
        print(f"✗ Theodorsen mapping failed: {e}")
        return None, None, None, None

def plot_coefficient_comparison(theta, psi, eta):
    """Plot the Theodorsen coefficients"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(theta, psi, 'b-', label='ψ(θ)')
    ax1.set_ylabel('ψ')
    ax1.set_title('Theodorsen Coefficients')
    ax1.grid(True)
    ax1.legend()
    
    ax2.plot(theta, eta, 'r-', label='η(θ)')
    ax2.set_xlabel('θ (rad)')
    ax2.set_ylabel('η')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('theodorsen_coeffs_naca0012.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    # Test with reference NACA0012
    foil, theta, psi, eta = test_theodorsen_with_naca()
    
    if foil is not None:
        print("\nPlotting coefficients...")
        plot_coefficient_comparison(theta, psi, eta)
        
        # Test pressure coefficient calculation
        try:
            aoa = 5.0  # degrees
            print(f"\nTesting C_p calculation at AoA = {aoa}°...")
            foil.C_p(aoa)
            print("✓ C_p calculation successful!")
        except Exception as e:
            print(f"✗ C_p calculation failed: {e}")