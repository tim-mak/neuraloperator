import torch
import numpy as np
import matplotlib.pyplot as plt
from neuralop.layers.fourier_continuation import FCLegendre

from pathlib import Path

def test_fc_legendre_real_data(data_file, sample_idx=0):
    print(f"--- Running FCLegendre Real CFD Test on {data_file} ---")
    
    # 1. Load the dataset
    if not Path(data_file).exists():
        print(f"Error: Could not find file {data_file}")
        return
        
    data = torch.load(data_file)
    
    # 2. Extract the Specific Channel: Cp_delta
    # y shape is expected to be [N, C, X, Y]. 
    # We want sample_idx, and channel 0 (Cp_delta).
    # We slice [sample_idx:sample_idx+1, 0:1, ...] to keep the (B, C, X, Y) dummy dims for the padder
    try:
        y_tensor = data['y'][sample_idx:sample_idx+1, 0:1, :, :]
    except KeyError:
        print("Error: Dataset does not contain the 'y' key.")
        return
        
    _, _, Nx, Ny = y_tensor.shape
    print(f"Extracted shape: {y_tensor.shape} (Physical Grid: {Nx}x{Ny})")
    
    # 3. Initialize Padder
    # Using the same parameters that caused the explosion earlier
    padder = FCLegendre(d=5, n_additional_pts=50).to(y_tensor.device)
    
    # 4. Apply Padding strictly to the X-axis (dim=-2)
    try:
        y_padded = padder.extend(y_tensor, dim=(-2,))
        print(f"Padded shape:    {y_padded.shape}")
    except Exception as e:
        print(f"CRASH during padding: {e}")
        return

    # Convert back to numpy for plotting (squeeze out B and C dims)
    Z_orig = y_tensor[0, 0].detach().cpu().numpy()
    Z_pad = y_padded[0, 0].detach().cpu().numpy()
    
    # --- Plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"FCLegendre Real CFD Test: Sample {sample_idx}, Channel: Cp_delta", fontsize=16)
    
    # Plot A: Original 2D
    im0 = axes[0].imshow(Z_orig.T, origin='lower', cmap='viridis', aspect='auto')
    axes[0].set_title(f"Original Physical Data\nMin: {Z_orig.min():.3f} | Max: {Z_orig.max():.3f}")
    fig.colorbar(im0, ax=axes[0])
    
    # Plot B: Padded 2D
    im1 = axes[1].imshow(Z_pad.T, origin='lower', cmap='viridis', aspect='auto')
    axes[1].set_title(f"Padded Data (Polynomial Extension)\nMin: {Z_pad.min():.3f} | Max: {Z_pad.max():.3f}")
    fig.colorbar(im1, ax=axes[1])
    
    # Plot C: 1D Cross-Section
    # Take a slice through the middle of the Y-axis (right through the wake!)
    y_mid = Ny // 2
    slice_orig = Z_orig[:, y_mid]
    slice_pad = Z_pad[:, y_mid]
    
    c_pts = padder.n_additional_pts // 2
    x_axis_pad = np.arange(-c_pts, Nx + c_pts)
    x_axis_orig = np.arange(0, Nx)
    
    axes[2].plot(x_axis_pad, slice_pad, color='red', label='Padded Signal (Polynomials)', linewidth=2)
    axes[2].plot(x_axis_orig, slice_orig, color='blue', label='Original Physical Signal', linestyle='--')
    
    # Highlight the padding regions
    axes[2].axvspan(-c_pts, 0, color='red', alpha=0.1, label='Left Pad')
    axes[2].axvspan(Nx, Nx + c_pts, color='red', alpha=0.1, label='Right Pad')
    
    axes[2].set_title("1D Cross-Section (y_mid)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("fc_legendre_real_data_test.png", dpi=150)
    plt.show()

# Run the test (just pass the path to your training or test file)


test_fc_legendre_real_data("/home/timm/storage/AF_NO_DATASET/train/airfoil_aoa_train_CMesh_256x32.pt", sample_idx=20)