import os
import glob
import numpy as np
import pyvista as pv
from collections import defaultdict

def analyze_block_statistics(dataset_path="OF_dataset"):
    """
    Scans through all OpenFOAM simulations in the dataset,
    reads the blockMesh.vtm file, and aggregates Ni and Nj stats for each block.
    """
    # Find all blockMesh.vtm files in the nested structure
    search_pattern = os.path.join(dataset_path, "*", "constant", "blockMeshVTK", "blockMesh.vtm")
    vtm_files = glob.glob(search_pattern)
    
    if not vtm_files:
        print(f"No .vtm files found using pattern: {search_pattern}")
        return

    print(f"Found {len(vtm_files)} simulations. Extracting block statistics...\n")

    # Data structure to hold our stats: { 'block_0': {'ni': [], 'nj': []}, ... }
    stats = defaultdict(lambda: {'ni': [], 'nj': []})
    
    # Track the total Ni to cross-reference with your 1031-1376 finding
    total_ni_list = []

    for filepath in vtm_files:
        try:
            # Read the MultiBlock dataset
            multi_block = pv.read(filepath)
            
            sim_total_ni = 0
            
            # Iterate through the blocks in the .vtm file
            for i in range(multi_block.n_blocks):
                block = multi_block[i]
                
                # PyVista MultiBlock keys are usually the names of the source files (e.g., 'block_0')
                block_name = multi_block.keys()[i] 
                
                ni, nj, nk = block.dimensions
                
                stats[block_name]['ni'].append(ni)
                stats[block_name]['nj'].append(nj)
                
                sim_total_ni += (ni - 1) # Subtract 1 for overlap, assuming typical C-mesh chain
                
            total_ni_list.append(sim_total_ni)
            
        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    # --- Print the Results in a Clean Table ---
    print(f"{'Block Name':<15} | {'Ni (Min - Max)':<20} | {'Ni (Mean)':<10} | {'Nj (Min - Max)':<20}")
    print("-" * 75)
    
    # Sort keys so block_0 to block_5 print in order
    for block_name in sorted(stats.keys()):
        ni_arr = np.array(stats[block_name]['ni'])
        nj_arr = np.array(stats[block_name]['nj'])
        
        ni_str = f"{ni_arr.min()} - {ni_arr.max()}"
        nj_str = f"{nj_arr.min()} - {nj_arr.max()}"
        ni_mean = f"{ni_arr.mean():.1f}"
        
        print(f"{block_name:<15} | {ni_str:<20} | {ni_mean:<10} | {nj_str:<20}")
        
    print("-" * 75)
    total_arr = np.array(total_ni_list)
    print(f"\nTotal Stitched Ni (Min - Max): {total_arr.min()} - {total_arr.max()}")

if __name__ == "__main__":

    OF_DATASET_DIR = "/home/timm/Projects/PIML/OF_dataset"
    analyze_block_statistics(OF_DATASET_DIR)