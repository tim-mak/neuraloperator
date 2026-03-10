import os
import sys


DATASET_PATH = "/home/timm/Projects/PIML/OF_dataset"

SOURCE_OF = "source /usr/lib/openfoam/openfoam2512/etc/bashrc"

os.system(SOURCE_OF)
 
dirs = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
dirs = ["airFoil2D_SST_67.783_-2.041_4.431_3.865_18.25"]

print("Available simulations:")
for d in dirs:
    #check if structuredblockMesh exists in constant/blockMeshVTK
    if os.path.exists(os.path.join(DATASET_PATH, d, "constant", "blockMeshVTK", "blockMesh.vtm")):
        print(f"Exists  {d}")
        os.system(f"cd {os.path.join(DATASET_PATH, d)} && BlockMeshStructured")

    else:

        # run the BlockMeshStructured  user application to create the structured mesh and save it as constant/blockMeshVTK.vtm

        print(f"Missing {d} - running BlockMeshStructured to create it...")
        os.system(f"cd {os.path.join(DATASET_PATH, d)} && BlockMeshStructured")


