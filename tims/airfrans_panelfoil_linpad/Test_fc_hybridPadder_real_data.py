import torch
import numpy as np
import matplotlib.pyplot as plt
from neuralop.layers.fourier_continuation import FCLegendre
from tims.airfrans_panelfoil_linpad.HybridSpectralPadder import HybridSpectralPadder

from pathlib import Path

def test_hybrid_padder_real_data(data_file, sample_idx=0):
    print(f"--- Running Hybrid Padder Test on {data_file} ---")
    
    if not Path(data_file).exists():
        print(f"Error: Could not find {data_file}")
        return
        
    data = torch.load(data_file)
    
    # Extract Cp_delta for a single sample
    try:
        y_tensor = data['y'][sample_idx:sample_idx+1, 2:3, :, :]
    except KeyError:
        print("Error: Dataset does not contain the 'y' key.")
        return
        
    _, _, Nx, Ny = y_tensor.shape
    
    # Initialize Hybrid Padder
    padder = HybridSpectralPadder(d=2, n_additional_pts=50).to(y_tensor.device)
    
    try:
        y_padded = padder.pad(y_tensor)

        print(f"Original shape: {y_tensor.shape}")
        print(f"Padded shape:   {y_padded.shape}")

        c_pts = padder.x_continuation.n_additional_pts // 2

    except Exception as e:
        print(f"CRASH during padding: {e}")
        return
    
    # Inside your test function:



    # Convert back to numpy (squeeze out B and C dims)
    Z_orig = y_tensor[0, 0].detach().cpu().numpy()
    Z_pad = y_padded[0, 0].detach().cpu().numpy()

    Nx = Z_orig.shape[0]
    Z_pad_crop = Z_pad[c_pts : c_pts + Nx, :]
    physical_part = y_padded[:, :, c_pts : c_pts + Nx, :Ny]
    error = torch.abs(physical_part - y_tensor).max()
    print(f"Direct Comparison Error: {error.item()}")
    
    # --- Plotting ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Hybrid Padder Test: Sample {sample_idx}, Channel: Cp_delta", fontsize=16, fontweight='bold')
    
    # A. Original 2D
    im0 = axes[0, 0].imshow(Z_orig.T, origin='lower', cmap='viridis', aspect='auto')
    axes[0, 0].set_title(f"Original Physical Data ({Nx}x{Ny})\nMin: {Z_orig.min():.3f} | Max: {Z_orig.max():.3f}")
    fig.colorbar(im0, ax=axes[0, 0])
    
    # B. Padded 2D
    im1 = axes[0, 1].imshow(Z_pad.T, origin='lower', cmap='viridis', aspect='auto')
    axes[0, 1].set_title(f"Hybrid Padded Data ({Z_pad.shape[0]}x{Z_pad.shape[1]})\nMin: {Z_pad.min():.3f} | Max: {Z_pad.max():.3f}")
    fig.colorbar(im1, ax=axes[0, 1])
    
    # C. 1D Cross-Section (X-Axis: Testing FCLegendre)
    y_mid = Ny // 2
    slice_orig_x = Z_orig[:, y_mid]
    slice_pad_x = Z_pad[:, y_mid]
    
    c_pts = padder.x_continuation.n_additional_pts // 2
    x_axis_pad = np.arange(-c_pts, Nx + c_pts)
    x_axis_orig = np.arange(0, Nx)
    
    axes[1, 0].plot(x_axis_pad, slice_pad_x, color='red', label='Legendre Polynomials', linewidth=2)
    axes[1, 0].plot(x_axis_orig, slice_orig_x, color='blue', label='Physical Signal', linestyle='--')
    axes[1, 0].axvspan(-c_pts, 0, color='red', alpha=0.1)
    axes[1, 0].axvspan(Nx, Nx + c_pts, color='red', alpha=0.1)
    axes[1, 0].set_title("X-Axis Cross-Section (Polynomial Bridge)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # D. 1D Cross-Section (Y-Axis: Testing Mirror)
    c = padder.x_continuation.n_additional_pts // 2
    x_mid_orig = Nx // 2
    x_mid_padded = Nx // 2 +c

    slice_orig_y = Z_orig[x_mid_orig, :]
    slice_pad_y = Z_pad[x_mid_padded, :]
    
    y_axis_pad = np.arange(0, Ny * 2)
    y_axis_orig = np.arange(0, Ny)
    
    axes[1, 1].plot(y_axis_pad, slice_pad_y, color='purple', label='Mirrored Signal', linewidth=2)
    axes[1, 1].plot(y_axis_orig, slice_orig_y, color='blue', label='Physical Signal', linestyle='--')
    axes[1, 1].axvspan(Ny, Ny * 2, color='purple', alpha=0.1, label='Mirror Region')
    axes[1, 1].set_title("Y-Axis Cross-Section (Perfect Mirror)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("hybrid_padder_test_lognutratio.png", dpi=150)
    plt.show()

test_hybrid_padder_real_data("/home/timm/storage/AF_NO_DATASET/train/airfoil_aoa_train_CMesh_256x32.pt", sample_idx=200)