import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # Ensure headless rendering is active

def plot_fno_predictions(model, test_loader, output_normalizer, epoch, save_dir, device='cuda'):
    """
    Pulls a single batch from the test loader, runs the FNO, and plots 
    the physical Truth vs Prediction vs Error for all 4 channels.
    """
    model.eval() # Turn off dropout/batchnorm for testing
    
    with torch.no_grad():
        # 1. Grab exactly one batch from the test dataset
        sample = next(iter(test_loader))
        x = sample['x'].to(device)
        y_truth = sample['y'].to(device)
        
        # 2. Run the FNO prediction
        y_pred = model(x)
        
        # 3. CRITICAL: Un-normalize the data back to physical fluid units
        y_truth_phys = output_normalizer.inverse_transform(y_truth)
        y_pred_phys = output_normalizer.inverse_transform(y_pred)
        
        # 4. Extract the very first airfoil in the batch and move to CPU for plotting
        # Shape becomes [4 channels, 256 radial, 32 azimuthal]
        truth = y_truth_phys[0].cpu().numpy()
        pred = y_pred_phys[0].cpu().numpy()
        
    # 5. Set up the plotting grid (4 rows for channels, 3 columns for Truth/Pred/Error)
    variables = [r'$\Delta C_p$', r'$\Delta U_x$', r'$\Delta U_y$', r'$\log_{10}(\nu_t/\nu)$']
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(16, 14))
    
    for i in range(4):
        # Ground Truth Column
        ax = axes[i, 0]
        # Using 'aspect=auto' stretches the 256x32 grid into a readable square
        im1 = ax.imshow(truth[i], aspect='auto', cmap='jet', origin='lower')
        ax.set_title(f"Truth: {variables[i]}")
        fig.colorbar(im1, ax=ax)
        
        # FNO Prediction Column
        ax = axes[i, 1]
        im2 = ax.imshow(pred[i], aspect='auto', cmap='jet', origin='lower')
        ax.set_title(f"FNO Prediction: {variables[i]}")
        fig.colorbar(im2, ax=ax)
        
        # Absolute Error Column
        ax = axes[i, 2]
        error = np.abs(truth[i] - pred[i])
        # Using a different colormap (magma) specifically for errors highlights hotspots
        im3 = ax.imshow(error, aspect='auto', cmap='magma', origin='lower')
        ax.set_title(f"Absolute Error: {variables[i]}")
        fig.colorbar(im3, ax=ax)
        
    plt.suptitle(f"FNO Validation - Epoch {epoch}", fontsize=16)
    plt.tight_layout()
    
    # Save the image cleanly
    save_path = save_dir / f"fno_diagnostic_epoch_{epoch}.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Diagnostic plot saved to {save_path}")