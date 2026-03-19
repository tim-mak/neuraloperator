import torch
import numpy as np
import matplotlib.pyplot as plt

def analyze_high_cp_indices(data_file):
    print(f"--- Analyzing High Delta Cp Indices in {data_file} ---")
    
    # 1. Load the dataset
    data = torch.load(data_file)
    y = data['y']  # Expected shape: [N, C, Nx, Ny]
    
    # 2. Extract Delta Cp (Channel 0)
    cp_delta = y[:, 0, :, :]
    
    # 3. Find indices where Delta Cp > 6.0
    # Returns [N_points, 3] tensor of (sample_idx, x_idx, y_idx)
    mask = cp_delta > 5.0
    indices = torch.nonzero(mask)
    
    num_points = indices.shape[0]
    if num_points == 0:
        print("No points found with Delta Cp > 6.0. Try a lower threshold (e.g., 2.0).")
        return

    print(f"Found {num_points} points exceeding threshold.")
    
    # 4. Extract indices for plotting
    sample_indices = indices[:, 0].cpu().numpy()
    x_indices = indices[:, 1].cpu().numpy()
    y_indices = indices[:, 2].cpu().numpy()
    
    # 5. Global Max Check
    max_val, max_flat_idx = torch.max(cp_delta.reshape(-1), 0)
    s_m, x_m, y_m = np.unravel_index(max_flat_idx.item(), cp_delta.shape)
    print(f"Absolute Global Max Delta Cp: {max_val:.4f}")
    print(f"Location: Sample {s_m}, Mesh Index (i={x_m}, j={y_m})")

    # 6. Plotting
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # We color by Sample Index to see if it's localized to one airfoil or systemic
    scatter = ax.scatter(x_indices, y_indices, c=sample_indices, cmap='viridis', 
                        alpha=0.6, s=15, edgecolors='none')
    
    ax.set_title(f"Mesh Locations (i, j) where Delta Cp > 6.0\nTotal Points: {num_points}", 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Mesh X-Index (i) - Chordwise/Wake", fontsize=12)
    ax.set_ylabel("Mesh Y-Index (j) - Normal to Surface", fontsize=12)
    
    # In a 256x32 C-mesh, j=0 is usually the airfoil surface
    ax.set_ylim(-1, 32)
    ax.set_xlim(-5, 260)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.colorbar(scatter, label='Sample Index')
    plt.tight_layout()
    plt.savefig("high_cp_mesh_indices.png", dpi=150)
    print("Plot saved as high_cp_mesh_indices.png")

# Run it on your training file
analyze_high_cp_indices("/home/timm/storage/AF_NO_DATASET/train/airfoil_aoa_train_CMesh_256x32.pt")