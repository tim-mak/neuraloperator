import torch
import numpy as np
import matplotlib.pyplot as plt
from neuralop.layers.fourier_continuation import FCLegendre

def test_fc_legendre_analytical():
    print("--- Running FCLegendre Analytical Toy Test ---")
    
    # 1. Define the Grid
    Nx, Ny = 100, 100
    x_1d = np.linspace(0, 4 * np.pi, Nx)
    y_1d = np.linspace(0, 4 * np.pi, Ny)
    X, Y = np.meshgrid(x_1d, y_1d, indexing='ij')
    
    # 2. Define the Function: sin(x) + sin(y) + cx + dy
    c, d = 0.5, 0.2
    Z_numpy = np.sin(X) + np.sin(Y) + c * X + d * Y
    
    # Convert to PyTorch tensor with batch and channel dims: (B, C, X, Y)
    z_tensor = torch.tensor(Z_numpy, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # 3. Initialize Padder
    # Using d=5 (poly degree) and 50 additional points (standard library default)
    padder = FCLegendre(d=5, n_additional_pts=50)
    
    # 4. Apply Padding strictly to the X-axis (dim=-2)
    try:
        z_padded = padder.extend(z_tensor, dim=(-2,))
        print(f"Original shape: {z_tensor.shape}")
        print(f"Padded shape:   {z_padded.shape}")
    except Exception as e:
        print(f"CRASH during padding: {e}")
        return

    # Convert back to numpy for plotting
    Z_orig = z_tensor[0, 0].cpu().numpy()
    Z_pad = z_padded[0, 0].cpu().numpy()
    
    # --- Plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("FCLegendre Unit Test: $f(x,y) = \sin(x) + \sin(y) + cx + dy$", fontsize=16)
    
    # Plot A: Original 2D
    im0 = axes[0].imshow(Z_orig.T, origin='lower', cmap='viridis', aspect='auto')
    axes[0].set_title("Original Data")
    fig.colorbar(im0, ax=axes[0])
    
    # Plot B: Padded 2D
    im1 = axes[1].imshow(Z_pad.T, origin='lower', cmap='viridis', aspect='auto')
    axes[1].set_title("Padded Data (X-axis extended)")
    fig.colorbar(im1, ax=axes[1])
    
    # Plot C: 1D Cross-Section (The critical view)
    # Take a slice through the middle of the Y-axis
    y_mid = Ny // 2
    slice_orig = Z_orig[:, y_mid]
    slice_pad = Z_pad[:, y_mid]
    
    # Create an X-axis for the padded array
    c_pts = padder.n_additional_pts // 2
    x_axis_pad = np.arange(-c_pts, Nx + c_pts)
    x_axis_orig = np.arange(0, Nx)
    
    axes[2].plot(x_axis_pad, slice_pad, color='red', label='Padded Signal (Polynomials)', linewidth=2)
    axes[2].plot(x_axis_orig, slice_orig, color='blue', label='Original Signal', linestyle='--')
    
    # Highlight the padding regions
    axes[2].axvspan(-c_pts, 0, color='red', alpha=0.1, label='Padding Region')
    axes[2].axvspan(Nx, Nx + c_pts, color='red', alpha=0.1)
    
    axes[2].set_title("1D Cross-Section (Middle of Y-axis)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("fc_legendre_test.png", dpi=150)
    plt.show()

# Run the test
test_fc_legendre_analytical()