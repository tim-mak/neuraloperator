import torch
import json
from pathlib import Path
from tqdm import tqdm
import shutil
import numpy as np
import pyvista as pv



def consolidate_airfrans_dataset(
    data_root: Path,
    output_dir: Path,
    splits_config: dict,
    xlim: float = 6.0,
    ylim: float = 3.0,
    xoffset: float = 1.0,
    grid_sizes: list = [  (256, 32),(512, 64),( 1024, 128)]

):
    """
    Consolidate individual simulation .pt files into PTDataset format.
    
    Args:
        data_root: Root directory containing simulation folders
        output_dir: Directory to save consolidated files
        splits_config: Dict mapping split names to their file lists from manifest
        xlim, ylim: Domain parameters used in filenames
        xoffset: X-offset parameter 
        grid_sizes: List of grid resolutions to process
    """
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for grid_size in grid_sizes:
        print(f"\nProcessing grid size {grid_size[0]}x{grid_size[1]}")
        
        for split_name, sim_runs in splits_config.items():
            if not sim_runs:  # Skip empty splits
                continue
                
            print(f"  Processing {split_name} split with {len(sim_runs)} simulations")
            
            # Collect all data for this split
            all_x = []
            all_y = []
            all_props = []
            
            for sim_run in tqdm(sim_runs, desc=f"Loading {split_name}"):
                sim_path = data_root 
                data_file = sim_path / f"{sim_run}_C_mesh_{grid_size[0]}x{grid_size[1]}.pt"
                
                if data_file.exists():
                    try:
                        data = torch.load(data_file)
                        data.pop('y_out', None)  # Remove y_out if it exists
                        data['y'] = data.pop('y_delta') # Rename to y for consistency

                        all_x.append(data['x'].unsqueeze(0))  # Add batch dimension
                        all_y.append(data['y'].unsqueeze(0))  # Add batch dimension
                        
                        p = data.get('props', {})
                        if isinstance(p, list) and len(p) > 0:
                            all_props.append(p[0])
                        else:
                            all_props.append(p)
                    except Exception as e:
                        print(f"Error loading {data_file}: {e}")
                        continue
                else:
                    print(f"Warning: {data_file} not found")
            
            if all_x:
                # Stack all samples along batch dimension
                consolidated_x = torch.cat(all_x, dim=0)  # Shape: [N, C, H, W]
                consolidated_y = torch.cat(all_y, dim=0)  # Shape: [N, C, H, W]
                
                consolidated_data = {
                    'x': consolidated_x,
                    'y': consolidated_y,
                    'props': all_props
                }
                
                # Determine if this is train or test split
                split_type = 'train' if 'train' in split_name else 'test'
                output_cons_dir = output_dir / split_type
                output_cons_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_cons_dir / f"airfoil_{split_name}_CMesh_{grid_size[0]}x{grid_size[1]}.pt"
                
                torch.save(consolidated_data, output_file)
                print(f"  Saved {output_file} with {consolidated_x.shape[0]} samples")
                print(f"    Input shape: {consolidated_x.shape}")
                print(f"    Output shape: {consolidated_y.shape}")
                print(f"    Props count: {len(all_props)}")
                print(f"    Props keys: {list(all_props[0].keys()) if all_props else 'N/A'}")


def main():
    # Configuration
    data_root = Path("/home/timm/storage/AF_NO_DATASET/Archive")
    output_dir = Path("/home/timm/storage/AF_NO_DATASET")

    manifest_file = Path("/home/timm/Projects/PIML/Dataset/manifest.json")
    
    # Load manifest
    if manifest_file.exists():
        with open(manifest_file, 'r') as f:
            manifest_data = json.load(f)
    else:
        print(f"Error: Manifest file not found at {manifest_file}")
        return
    
    # Define which splits to use for train vs test
    # You can adjust this mapping based on your needs
    splits_config = {
        # Training splits - will be combined into one train file
        'full_train': manifest_data.get('full_train', []),
        'scarce_train': manifest_data.get('scarce_train', []),
        'aoa_train': manifest_data.get('aoa_train', []),
        'reynolds_train': manifest_data.get('reynolds_train', []),
        
        # Test splits - each will create a separate test file
        'full_test': manifest_data.get('full_test', []),
        'aoa_test': manifest_data.get('aoa_test', []),
        'reynolds_test': manifest_data.get('reynolds_test', [])
           
        }
    
    #grid_sizes=[(64,64),(128,128), (256,256), (512,512), (1024, 1024)]
    # Run consolidation
    consolidate_airfrans_dataset(
        data_root=data_root,
        output_dir=output_dir,
        splits_config=splits_config,
        xlim=6,
        ylim=3,
        grid_sizes=[ (256, 32),(512, 64),( 1024, 128)]
    )
    shutil.copy(manifest_file, Path(data_root) / "manifest.json")
    shutil.copy(manifest_file, Path(output_dir) / "manifest.json")
    print(f"\nConsolidation complete! Files saved to {output_dir}")
    print("You can now use these with PTDataset:")
    output_train_dir = output_dir / "train"
    output_test_dir = output_dir / "test"
    for f in output_train_dir.glob("*.pt"):
        print(f"  {f.name}")
    for f in output_test_dir.glob("*.pt"):
        print(f"  {f.name}")

if __name__ == "__main__":
    main()