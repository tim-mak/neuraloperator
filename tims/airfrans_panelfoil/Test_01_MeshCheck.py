

import torch
import torch.nn.functional as F
import os
import pyvista as pv
import airfrans as af

from BlockMeshInterpolator import extract_arrays, resample_block_pchip,  chain_stitch_orientation, compute_jacobian_metrics, concatenate_blocks_ml_tensor, read_openfoam_results, read_structured_grids_from_vtm


AF_ROOT = "/home/timm/storage/AF_small/Dataset"

SIM_PATH = "/home/timm/storage/AF_NO_DATASET/OF_dataset/airFoil2D_SST_31.68_0.424_0.273_4.301_1.0_11.616"
sim_name = os.path.basename(SIM_PATH)
VTM_PATH = f"{SIM_PATH}/constant/blockMeshVTK/blockMesh.vtm"
FOAM_PATH = f"{SIM_PATH}/touch.foam"

OUT_DIR = "structured_block"

output_type = 'vts'

if not os.path.exists(FOAM_PATH):
    os.system(f"touch {FOAM_PATH}")  # create empty file to satisfy PyVista reader

simulation = af.Simulation(root=AF_ROOT, name=sim_name)
nu = simulation.NU
rho = simulation.RHO
U_inf = simulation.inlet_velocity
reynolds = rho * U_inf / nu

foam_results = read_openfoam_results(FOAM_PATH)

# Merge all OpenFOAM blocks into one UnstructuredGrid so we can call .interpolate()
# use_all_points=True keeps ghost/boundary points; progress_bar for large meshes
print("\nMerging OpenFOAM unstructured blocks into a single mesh for interpolation...")
foam_combined = foam_results.combine(merge_points=True)
print(f"\nCombined OpenFOAM mesh: type={type(foam_combined).__name__}  "
        f"pts={foam_combined.n_points}  cells={foam_combined.n_cells}")
print(f"Available arrays: {foam_combined.array_names}")
# move data to point data for sampling
foam_combined = foam_combined.cell_data_to_point_data(pass_cell_data=True)  # Ensure all cell data is also available as point data for sampling
foam_combined['x'] = foam_combined.points[:, 0]
foam_combined['y'] = foam_combined.points[:, 1]
foam_combined['U_x'] = foam_combined['U'][:, 0]
foam_combined['U_y'] = foam_combined['U'][:, 1]
foam_combined["Cp"] = foam_combined["p"] / (0.5 * U_inf**2)  # Note no rho because incompressible Openfoam results 

xmin, xmax = 0.95, 1.05
ymin, ymax = -0.05, 0.05

zmin, zmax = foam_combined.bounds[4], foam_combined.bounds[5]
bounds = (xmin, xmax, ymin, ymax, zmin, zmax)


# Read the structured grids from the VTM file
grids = read_structured_grids_from_vtm(VTM_PATH)

grid_by_name = {name: grid for name, grid in grids}
# Looking only at block 3 which has problems with the interpolation. We want to visualize the raw sampled fields on this block to diagnose the issue.
bname = 'block_3'
grid = grid_by_name[bname]
ni, nj, nk = grid.dimensions

sub = grid.extract_subset([0, ni - 1, 0,  nj - 1, 0, 1])
sub['x'] = sub.points[:, 0]
sub['y'] = sub.points[:, 1]

blk_sampled = sub.sample(foam_combined, snap_to_closest_point=True)


print(blk_sampled)
blk_subset = blk_sampled.clip_box(bounds, invert=False)
foam_subset = foam_combined.clip_box(bounds, invert=False)


pltter = pv.Plotter()
pltter.add_mesh(
    blk_subset,
    color="tomato",
    style="wireframe",
    render_points_as_spheres=True,
    point_size=4,
)

pltter.add_mesh(
    foam_subset,
    color="deepskyblue",
    style="points",
    render_points_as_spheres=True,
    point_size=3,
    opacity=0.35,
)
pltter.add_legend([
    ["sampled structured points", "tomato"],
    ["openfoam points", "deepskyblue"],
])

# Force camera framing to the same spatial subset bounds.
pltter.camera_position = "xy"
pltter.enable_parallel_projection()
pltter.reset_camera(bounds=bounds)
pltter.show()



pltter = pv.Plotter()
pltter.add_mesh(
    blk_sampled,
    scalars="U_y",
    cmap="coolwarm",
    clim=[0.0, 0.05],
    scalar_bar_args={"title": "U_y"},
)

# Force camera framing to the same spatial subset bounds.
pltter.camera_position = "xy"
pltter.enable_parallel_projection()
pltter.reset_camera(bounds=bounds)
pltter.show()



pltter = pv.Plotter()
pltter.add_mesh(
    blk_sampled,
    scalars="Cp",
    cmap="coolwarm",
    clim=[0.0, 0.05],
    scalar_bar_args={"title": "Cp"},
)

# Force camera framing to the same spatial subset bounds.
pltter.camera_position = "xy"
pltter.enable_parallel_projection()
pltter.reset_camera(bounds=bounds)
pltter.show()