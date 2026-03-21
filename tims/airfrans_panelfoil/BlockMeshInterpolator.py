import pyvista as pv
import numpy as np
import os
from scipy.interpolate import PchipInterpolator

def read_structured_grids_from_vtm(vtm_path):
    """Reads a VTM file and returns a list of all StructuredGrids found."""
    mb = pv.read(vtm_path)
    print(f"\nLoaded: {vtm_path}")
    print(f"Top-level type : {type(mb).__name__}")
    print(f"Top-level blocks: {mb.n_blocks}\n")

    def collect_structured_grids(mb, results=None):
        if results is None:
            results = []
        for i in range(mb.n_blocks):
            child = mb[i]
            if isinstance(child, pv.StructuredGrid):
                name = mb.get_block_name(i) or str(i)
                print(f"  Found StructuredGrid '{name}'  dims={child.dimensions}  "
                      f"pts={child.n_points}  cells={child.n_cells}")
                results.append((name, child))
            elif isinstance(child, pv.MultiBlock):
                collect_structured_grids(child, results)
        return results

    grids = collect_structured_grids(mb)
    if not grids:
        raise ValueError("No StructuredGrids found in the VTM file.")
    print(f"\nTotal structured grids found: {len(grids)}")
    return grids

def read_openfoam_results(foam_path):
    """Reads an OpenFOAM case and returns the MultiBlock dataset at the last time step."""
    reader = pv.POpenFOAMReader(foam_path)
    reader.enable_all_patch_arrays()
    # Skip time 0 — initial conditions use $variable substitution that PyVista
    # cannot parse. Read only the final (solved) time step instead.
    times = reader.time_values
    print(f"Available time steps: {times}")
    reader.set_active_time_value(times[-1])
    print(f"Reading time step: {times[-1]}")
    mesh = reader.read()
    print(f"\nLoaded: {foam_path}")
    print(f"Dataset type  : {type(mesh).__name__}")
    print(f"Number of blocks: {mesh.n_blocks}")

    def print_block(mb, indent=0):
        prefix = "  " * indent
        for i in range(mb.n_blocks):
            name = mb.get_block_name(i) or str(i)
            child = mb[i]
            if isinstance(child, pv.MultiBlock):
                print(f"{prefix}[{i}] '{name}'  ({child.n_blocks} children)")
                print_block(child, indent + 1)
            elif child is not None:
                arrays = list(child.cell_data.keys()) + list(child.point_data.keys())
                print(f"{prefix}[{i}] '{name}'  type={type(child).__name__}  "
                      f"pts={child.n_points}  cells={child.n_cells}  arrays={arrays}")

    print_block(mesh)
    return mesh


def _flip_i_axis(sub: pv.StructuredGrid) -> pv.StructuredGrid:
    """
    Return a new StructuredGrid with the i-axis reversed.
    All point and cell data arrays are reordered to match.
    """
    ni, nj, nk = sub.dimensions
    pts = np.array(sub.points).reshape((ni, nj, nk, 3), order='F')
    new_pts = pts[::-1, :, :, :].reshape((-1, 3), order='F')

    new_grid = pv.StructuredGrid()
    new_grid.dimensions = sub.dimensions
    new_grid.points = new_pts

    nc_i, nc_j, nc_k = ni - 1, nj - 1, max(nk - 1, 1)

    def flip_cell(arr):
        shape = (nc_i, nc_j, nc_k) + arr.shape[1:]
        return arr.reshape(shape, order='F')[::-1].reshape(arr.shape, order='F')

    def flip_point(arr):
        shape = (ni, nj, nk) + arr.shape[1:]
        return arr.reshape(shape, order='F')[::-1].reshape(arr.shape, order='F')

    for name in sub.cell_data.keys():
        new_grid.cell_data[name] = flip_cell(np.array(sub.cell_data[name]))
    for name in sub.point_data.keys():
        new_grid.point_data[name] = flip_point(np.array(sub.point_data[name]))
    return new_grid


def chain_stitch_orientation(subgrids: list) -> list:
    """
    Given an ordered list of (name, StructuredGrid) that traces j=0
    continuously around a C-mesh in the physical order supplied by the caller,
    ensure the i-axis of every block runs in the same chainwise direction.

    Algorithm:
      - Block 0 is the anchor.  Its natural i=0 end is taken as the chain start
        (no flip applied to block 0).
      - For each subsequent block k, the j=0 wall edge has two candidate
        endpoints: start=(i=0,j=0) and end=(i=ni-1,j=0).  Whichever is
        spatially closer to the previous block's end-point (i=ni-1,j=0)
        becomes the new i=0 (flip if needed).

    This is purely topological — physical coordinates are unchanged.
    """
    result = []
    prev_end_xy = None

    for seq_idx, (name, sub) in enumerate(subgrids):
        ni, nj, nk = sub.dimensions
        pts3d = np.array(sub.points).reshape((ni, nj, nk, 3), order='F')
        start_xy = pts3d[0,  0, 0, :2]   # i=0,    j=0
        end_xy   = pts3d[-1, 0, 0, :2]   # i=ni-1, j=0

        if seq_idx == 0:
            # Anchor: accept natural direction
            print(f"  [chain] '{name}' anchor  j=0: i=0 {start_xy} → i=ni-1 {end_xy}")
            prev_end_xy = end_xy
            result.append((name, sub))
            continue

        dist_start = float(np.linalg.norm(start_xy - prev_end_xy))
        dist_end   = float(np.linalg.norm(end_xy   - prev_end_xy))

        if dist_end < dist_start:
            sub = _flip_i_axis(sub)
            pts3d = np.array(sub.points).reshape((ni, nj, nk, 3), order='F')
            start_xy = pts3d[0,  0, 0, :2]
            end_xy   = pts3d[-1, 0, 0, :2]
            print(f"  [chain] '{name}' FLIPPED  j=0: i=0 {start_xy} → i=ni-1 {end_xy}  "
                  f"(dist_start={dist_start:.4f} dist_end={dist_end:.4f})")
        else:
            print(f"  [chain] '{name}' ok       j=0: i=0 {start_xy} → i=ni-1 {end_xy}  "
                  f"(dist_start={dist_start:.4f} dist_end={dist_end:.4f})")

        prev_end_xy = end_xy
        result.append((name, sub))

    return result



def compute_jacobian_metrics(sub: pv.StructuredGrid) -> dict:
    """
    Compute 2D Jacobian metric tensor components at cell centres of a structured
    grid by treating the (i, j) grid lines as the computational coordinates.

    Uses the exact bilinear-cell formula (average of the two opposing edge
    differences) so that the result is the Jacobian evaluated at the cell
    centre (ξ=½, η=½) of each quad.

    Returns a dict of flat numpy arrays (VTK Fortran/column-major cell order):
        x_xi, y_xi   – ∂x/∂i, ∂y/∂i  (streamwise metric)
        x_eta, y_eta – ∂x/∂j, ∂y/∂j  (wall-normal metric)
        det_J        – determinant = x_xi·y_eta − x_eta·y_xi
        xi_x, xi_y  – ∂i/∂x, ∂i/∂y  (inverse, first row)
        eta_x, eta_y – ∂j/∂x, ∂j/∂y (inverse, second row)
        det_J_vtk    – minimum Jacobian from compute_cell_quality (cross-check)
    """
    ni, nj, nk = sub.dimensions             # node counts
    nc_i, nc_j = ni - 1, nj - 1             # cell counts

    # ── Extract k=0 node plane (x, y) ───────────────────────────────────────
    pts = np.array(sub.points)              # (N_pts, 3)
    # VTK StructuredGrid stores points in Fortran order: fastest→i, then j, then k
    pts3d = pts.reshape((ni, nj, nk, 3), order='F')  # (ni, nj, nk, 3)
    X = pts3d[:, :, 0, 0]   # (ni, nj)  x-coordinates at k=0
    Y = pts3d[:, :, 0, 1]   # (ni, nj)  y-coordinates at k=0

    # --- 2. Nodal Metrics (For FNO Input) ---
    # np.gradient uses second-order central differences internally 
    # and first-order at the boundaries.
    x_xi_p, x_eta_p = np.gradient(X, edge_order=2)
    y_xi_p, y_eta_p = np.gradient(Y, edge_order=2)

    det_p = x_xi_p * y_eta_p - x_eta_p * y_xi_p

    # ── Bilinear Jacobian at cell centres ────────────────────────────────────
    # For cell (I, J) with corners at grid nodes (I,J),(I+1,J),(I,J+1),(I+1,J+1):
    #   ∂x/∂i ≈ ½ [(X[I+1,J]−X[I,J]) + (X[I+1,J+1]−X[I,J+1])]
    #   ∂x/∂j ≈ ½ [(X[I,J+1]−X[I,J]) + (X[I+1,J+1]−X[I+1,J])]
    # (and same for y)
    dX_di  = 0.5 * ((X[1:, :-1] - X[:-1, :-1]) + (X[1:, 1:] - X[:-1, 1:]))  # (nc_i, nc_j)
    dX_dj  = 0.5 * ((X[:-1, 1:] - X[:-1, :-1]) + (X[1:, 1:] - X[1:, :-1]))
    dY_di  = 0.5 * ((Y[1:, :-1] - Y[:-1, :-1]) + (Y[1:, 1:] - Y[:-1, 1:]))
    dY_dj  = 0.5 * ((Y[:-1, 1:] - Y[:-1, :-1]) + (Y[1:, 1:] - Y[1:, :-1]))

    det    = dX_di * dY_dj - dX_dj * dY_di   # (nc_i, nc_j)

    # Inverse metrics (∂computational/∂physical)
    inv    = np.where(np.abs(det) > 1e-30, 1.0 / det, np.nan)
    di_dx  =  dY_dj * inv
    di_dy  = -dX_dj * inv
    dj_dx  = -dY_di * inv
    dj_dy  =  dX_di * inv

    # ── Flatten in VTK cell order (Fortran / column-major over i,j) ──────────
    def F(arr2d):
        # arr2d is (nc_i, nc_j); VTK cell index = J*nc_i + I  → ravel F-order
        return arr2d.ravel(order='F')

    # ── Cross-check: VTK minimum Jacobian ────────────────────────────────────
    # compute_cell_quality works on the hex cells; for a unit-thick z slab the
    # VTK Jacobian = 2D det_J * dz (dz = z-thickness of the slab).
    dz = float(np.abs(pts3d[0, 0, 1, 2] - pts3d[0, 0, 0, 2]))
    if dz == 0.0:
        dz = 1.0  # degenerate slab, avoid division by zero
    quality = sub.compute_cell_quality(quality_measure='jacobian')
    vtk_j   = quality.cell_data['CellQuality']                    # (nc_i * nc_j,)

    # The VTK Jacobian is the minimum over the 8 corners of each hex; for a nearly
    # uniform slab it is ≈ det_J * dz.  Normalise and compare.
    vtk_j_normalised = vtk_j / dz

    result_points = dict(
        x_xi_p   = F(x_xi_p),
        y_xi_p   = F(y_xi_p),
        x_eta_p  = F(x_eta_p),
        y_eta_p  = F(y_eta_p),
        det_p    = F(det_p),
    )

    result_cell_center = dict(
        x_xi   = F(dX_di),
        y_xi   = F(dY_di),
        x_eta  = F(dX_dj),
        y_eta  = F(dY_dj),
        det_J  = F(det),
        xi_x   = F(di_dx),
        xi_y   = F(di_dy),
        eta_x  = F(dj_dx),
        eta_y  = F(dj_dy),
        #det_J_vtk = vtk_j_normalised,
    )

    print(f" Check size of PointData arrays: {[(k, len(v)) for k, v in result_points.items()]}")
    print(f" Check size of CellData arrays: {[(k, len(v)) for k, v in result_cell_center.items()]}")
    print(f" Check if PointData and CellData lengths match expected counts: "
          f"pts={sub.n_points}  cells={sub.n_cells}")
    print(f" result_points {x_xi_p.shape}  vs result_cell_center {dX_di.shape}")
    # Quick sanity report
    max_rel_err = np.nanmax(np.abs(np.abs(F(det)) - np.abs(vtk_j_normalised)) /
                            (np.abs(F(det)) + 1e-30))
    sign = "negative" if F(det).mean() < 0 else "positive"
    print(f"  Jacobian check ({sign} orientation)  "
          f"|det_J| range [{np.abs(F(det)).min():.3e}, {np.abs(F(det)).max():.3e}]  "
          f"vtk_normalised range [{vtk_j_normalised.min():.3e}, {vtk_j_normalised.max():.3e}]  "
          f"max_rel_err={max_rel_err:.3e}")
    return result_points, result_cell_center

import numpy as np

def concatenate_blocks_ml_tensor(resampled_blocks: dict, BLOCK_ORDER):
    """
    Concatenates the resampled block dictionaries into a single, perfectly 
    aligned ML tensor, dropping the overlapping boundary nodes.
    
    Args:
        resampled_blocks: Dictionary containing 'data', 'x', and 'y' arrays per block.
        J_MAX: The final normal-direction crop (drops extreme far-field).
        
    Returns:
        final_data: np.ndarray [Channels, 1024, J_MAX]
        final_x: np.ndarray [1024, J_MAX]
        final_y: np.ndarray [1024, J_MAX]
    """
    # The physical wrapper sequence around the C-mesh
    
    data_parts = []
    x_parts = []
    y_parts = []
    w_parts =[] 
    
    for i, name in enumerate(BLOCK_ORDER):
        b = resampled_blocks[name]
        d = b['data']  # [Channels, Ni, Nj]
        x = b['x']     # [Ni, Nj]
        y = b['y']     # [Ni, Nj]
        w = b['wss']   # [Ni]
        
        # Deduplication: Drop the last i-node for every block EXCEPT the final wake block
        if i < len(BLOCK_ORDER) - 1:
            data_parts.append(d[:, :-1, :])
            x_parts.append(x[:-1, :])
            y_parts.append(y[:-1, :])
            w_parts.append(w[:-1])
        else:
            data_parts.append(d)
            x_parts.append(x)
            y_parts.append(y)
            w_parts.append(w)
            
    # Concatenate along the i-axis (dim=1 for data, dim=0 for coordinates)
    final_data = np.concatenate(data_parts, axis=1)
    final_x = np.concatenate(x_parts, axis=0)
    final_y = np.concatenate(y_parts, axis=0)
    final_w = np.concatenate(w_parts,axis=1)
        
    print(f"\nFinal ML Tensor Assembled: {final_data.shape}")
    print(f"Total i-nodes (Data): {final_data.shape[1]}  (Target: 1024)")
    print(f"Total i-nodes (Y-coord): {final_y.shape[0]}  (Target: 1024)") 
    print(f"Total i-nodes (WSS): {final_w.shape[1]}  (Target: 1024)")

    return final_data, final_x, final_y,final_w



def concatenate_blocks(sampled: list, z_ref: float = 0.0) -> pv.StructuredGrid:
    """
    Concatenate a chain-ordered list of (name, StructuredGrid) along the i-axis
    into a single 2-D StructuredGrid (dimensions NI × nj × 1).

    z_ref controls which z-plane is extracted from each block:
      z_ref=0.0  → the k-slice whose mean z is closest to 0   (default)
      z_ref=0.5  → average of both k-planes (midplane)

    Because some blocks have their z-axis flipped, always taking k=0 selects
    z≈1 for those blocks.  Auto-selecting by z_ref fixes this.

    Shared i-planes at block junctions are deduplicated.
    """
    if not sampled:
        raise ValueError("Empty list")

    njs = [g.dimensions[1] for _, g in sampled]
    assert len(set(njs)) == 1, f"nj mismatch across blocks: {njs}"
    nj = njs[0]

    def extract_plane(g: pv.StructuredGrid) -> np.ndarray:
        """Return (ni, nj, 1, 3) point array at z≈z_ref (or midplane)."""
        ni, _nj, nk = g.dimensions
        p3d = np.array(g.points).reshape((ni, nj, nk, 3), order='F')
        if z_ref == 0.5 or nk == 1:
            # Midplane: average the two z-slabs
            return (0.5 * (p3d[:, :, 0, :] + p3d[:, :, -1, :])).reshape(
                (ni, nj, 1, 3))
        else:
            # Pick the k-index whose mean |z - z_ref| is smallest
            z0 = float(np.abs(p3d[:, :, 0,  2].mean() - z_ref))
            z1 = float(np.abs(p3d[:, :, -1, 2].mean() - z_ref))
            k_sel = 0 if z0 <= z1 else nk - 1
            return p3d[:, :, k_sel:k_sel+1, :]   # (ni, nj, 1, 3)

    # ── Concatenate points ────────────────────────────────────────────────────
    pts_parts = []
    for seq, (name, g) in enumerate(sampled):
        p2d = extract_plane(g)                        # (ni, nj, 1, 3)
        pts_parts.append(p2d if seq == 0 else p2d[1:, :, :, :])

    pts_cat = np.concatenate(pts_parts, axis=0)       # (NI, nj, 1, 3)
    NI = pts_cat.shape[0]

    merged = pv.StructuredGrid()
    merged.dimensions = (NI, nj, 1)
    merged.points = pts_cat.reshape((-1, 3), order='F')
    print(f"  Concatenated 2D grid  z_ref={z_ref}  dimensions={merged.dimensions}  "
          f"pts={merged.n_points}  cells={merged.n_cells}")

    # ── Concatenate cell data (nc_k=1 in both 3-D source and 2-D merged) ─────
    nc_j = nj - 1
    nc_k = 1
    for key in list(sampled[0][1].cell_data.keys()):
        parts = []
        for name, g in sampled:
            nc_i = g.dimensions[0] - 1
            arr  = np.array(g.cell_data[key])
            extra = arr.shape[1:] if arr.ndim > 1 else ()
            parts.append(arr.reshape((nc_i, nc_j, nc_k) + extra, order='F'))
        cat = np.concatenate(parts, axis=0)
        merged.cell_data[key] = cat.ravel(order='F') if cat.ndim == 3 \
            else cat.reshape((-1,) + cat.shape[3:], order='F')

    # ── Concatenate point data, using same z-plane selection ─────────────────
    for key in list(sampled[0][1].point_data.keys()):
        parts = []
        for seq, (name, g) in enumerate(sampled):
            ni, _nj, nk = g.dimensions
            arr  = np.array(g.point_data[key])
            extra = arr.shape[1:] if arr.ndim > 1 else ()
            p3d  = arr.reshape((ni, nj, nk) + extra, order='F')
            if z_ref == 0.5 or nk == 1:
                p2d = 0.5 * (p3d[:, :, 0] + p3d[:, :, -1])   # (ni, nj, ...)
            else:
                p3d_xyz = np.array(g.points).reshape((ni, nj, nk, 3), order='F')
                z0 = float(np.abs(p3d_xyz[:, :, 0,  2].mean() - z_ref))
                z1 = float(np.abs(p3d_xyz[:, :, -1, 2].mean() - z_ref))
                k_sel = 0 if z0 <= z1 else nk - 1
                p2d = p3d[:, :, k_sel]                         # (ni, nj, ...)
            parts.append(p2d if seq == 0 else p2d[1:, :])
        cat = np.concatenate(parts, axis=0)                    # (NI, nj, ...)
        merged.point_data[key] = cat.ravel(order='F') if cat.ndim == 2 \
            else cat.reshape((-1,) + cat.shape[2:], order='F')

    return merged



import numpy as np

def extract_arrays(subgrid, feature_list):
    """
    Extracts coordinates and flow variables from a PyVista StructuredGrid block
    and reshapes them into proper 2D spatial numpy arrays.
    
    Args:
        subgrid: pyvista.StructuredGrid (e.g., from chain_stitch_orientation)
        feature_list: list of strings (keys in subgrid.point_data)
        
    Returns:
        data: np.ndarray shape [Channels, Ni, Nj]
        X: np.ndarray shape [Ni, Nj]
        Y: np.ndarray shape [Ni, Nj]
    """
    ni, nj, nk = subgrid.dimensions
    
    # 1. Extract and reshape physical coordinates
    # order='F' maps the flat VTK array back to (i, j, k) correctly
    # [:, :, 0] safely drops the empty Z (k) dimension
    X = subgrid.points[:, 0].reshape((ni, nj, nk), order='F')[:, :, 0]
    Y = subgrid.points[:, 1].reshape((ni, nj, nk), order='F')[:, :, 0]
    
    # 2. Initialize the multi-channel data tensor
    C = len(feature_list)
    data = np.zeros((C, ni, nj), dtype=np.float32)
    
    # 3. Extract and reshape each requested flow feature
    for c, feat_name in enumerate(feature_list):
        if feat_name not in subgrid.point_data:
            raise KeyError(f"Feature '{feat_name}' not found in block point_data. Available: {list(subgrid.point_data.keys())}")
            
        flat_array = subgrid.point_data[feat_name]
        data[c, :, :] = flat_array.reshape((ni, nj, nk), order='F')[:, :, 0]
        
    return data, X, Y

def resample_block_pchip(block_data, block_x, block_y, target_ni=256):
    """
    block_data: [Channels, Ni, Nj]
    block_x, block_y: [Ni, Nj]
    Returns perfectly interpolated data and coordinates to target_ni length.
    """
    C, Ni, Nj = block_data.shape
    
    # We use 'computational space' (index fraction from 0 to 1)
    old_s = np.linspace(0, 1, Ni)
    new_s = np.linspace(0, 1, target_ni)
    
    # Initialize output arrays
    new_data = np.zeros((C, target_ni, Nj), dtype=np.float32)
    new_x = np.zeros((target_ni, Nj), dtype=np.float32)
    new_y = np.zeros((target_ni, Nj), dtype=np.float32)
    
    # Interpolate J-layer by J-layer (Zero vertical bleeding)
    for j in range(Nj):
        # 1. Resample coordinates (Preserves LE clustering)
        interp_x = PchipInterpolator(old_s, block_x[:, j])
        interp_y = PchipInterpolator(old_s, block_y[:, j])
        new_x[:, j] = interp_x(new_s)
        new_y[:, j] = interp_y(new_s)
        
        # 2. Resample flow channels
        for c in range(C):
            interp_c = PchipInterpolator(old_s, block_data[c, :, j])
            new_data[c, :, j] = interp_c(new_s)
            
    return new_data, new_x, new_y


if __name__ == "__main__":

    SIM_PATH = "/home/timm/Projects/PIML/OF_dataset/airFoil2D_SST_31.68_0.424_0.273_4.301_1.0_11.616"

    VTM_PATH = f"{SIM_PATH}/constant/blockMeshVTK/blockMesh.vtm"
    FOAM_PATH = f"{SIM_PATH}/touch.foam"

    OUT_DIR = "structured_block"

    output_type = 'vts'

    if not os.path.exists(FOAM_PATH):
        os.system(f"touch {FOAM_PATH}")  # create empty file to satisfy PyVista reader


    foam_results = read_openfoam_results(FOAM_PATH)

    # Merge all OpenFOAM blocks into one UnstructuredGrid so we can call .interpolate()
    # use_all_points=True keeps ghost/boundary points; progress_bar for large meshes
    foam_combined = foam_results.combine(merge_points=True)
    print(f"\nCombined OpenFOAM mesh: type={type(foam_combined).__name__}  "
          f"pts={foam_combined.n_points}  cells={foam_combined.n_cells}")
    print(f"Available arrays: {foam_combined.array_names}")

    grids = read_structured_grids_from_vtm(VTM_PATH)

    # Physical traversal order around the C-mesh (clockwise, i increases along chain):
    #   block_0 → wake (forward, toward TE)
    #   block_3 → lower surface TE region
    #   block_5 → lower surface LE / LE wrap
    #   block_4 → upper surface LE region
    #   block_2 → upper surface TE region
    #   block_1 → wake (back, away from TE)
    BLOCK_ORDER = ['block_0', 'block_3', 'block_5', 'block_4', 'block_2', 'block_1']
    grid_by_name = {name: grid for name, grid in grids}

    # need only first 158 points in j direction to cover cartesian domain used in Airfrans data
    # extract_subset takes a single flat extent: [i_min, i_max, j_min, j_max, k_min, k_max]
    J_MAX = 158
    I0_min = 58
    I1_max = 225 - I0_min   # symmetric trim on the other wake half
    subgrids_raw = []
    for bname in BLOCK_ORDER:
        grid = grid_by_name[bname]
        ni, nj, nk = grid.dimensions
        if bname == 'block_0':
            sub = grid.extract_subset([I0_min, ni - 1, 0, min(J_MAX, nj - 1), 0, 1])
        elif bname == 'block_1':
            sub = grid.extract_subset([I0_min, ni - 1, 0, min(J_MAX, nj - 1), 0, 1])
        else:
            sub = grid.extract_subset([0, ni - 1, 0, min(J_MAX, nj - 1), 0, 1])
        subgrids_raw.append((bname, sub))
        print(f"'{bname}': {grid.dimensions} → {sub.dimensions}  "
              f"pts={sub.n_points}  cells={sub.n_cells}")

    print("\nChain-stitching i-axis orientations:")
    subgrids = chain_stitch_orientation(subgrids_raw)

    # Keep only the fields we need in the source mesh
    CELL_FIELDS  = ["U", "p", "nut"]           # cell-centred → cell_data
    POINT_FIELDS = ["p", "wallShearStress"]     # node-interpolated → point_data
    KEEP = set(CELL_FIELDS + POINT_FIELDS)
    for arr in list(foam_combined.array_names):
        if arr not in KEEP:
            if arr in foam_combined.cell_data:  foam_combined.cell_data.remove(arr)
            if arr in foam_combined.point_data: foam_combined.point_data.remove(arr)
    print(f"Retained arrays in source: {foam_combined.array_names}")

    # ── Sample each block, compute Jacobians, collect ─────────────────────────
    import os
    sampled_list = []
    for name, sub in subgrids:
        print(f"Sampling '{name}'...")

        # Cell-centred fields: U, p, nut
        cc = sub.cell_centers()
        cc_sampled = cc.sample(foam_combined)
        out = sub.copy(deep=False)
        out.clear_data()
        for field in CELL_FIELDS:
            if field in cc_sampled.point_data:
                out.cell_data[field] = cc_sampled.point_data[field]
            else:
                print(f"  WARNING: cell field '{field}' not found for '{name}'")

        # Node fields: p, wallShearStress
        pts_sampled = sub.sample(foam_combined)
        for field in POINT_FIELDS:
            if field in pts_sampled.point_data:
                out.point_data[field] = pts_sampled.point_data[field]
            else:
                print(f"  WARNING: point field '{field}' not found for '{name}'")

        # Jacobian metric tensor
        jac = compute_jacobian_metrics(sub)
        for key, arr in jac.items():
            out.cell_data[key] = arr

        sampled_list.append((name, out))

    # ── Concatenate into a single StructuredGrid and save ─────────────────────
    print("\nConcatenating blocks along i-axis...")
    merged = concatenate_blocks(sampled_list, z_ref=0.0)
    if output_type =='vts':
        os.makedirs(OUT_DIR, exist_ok=True)
        out_path = os.path.join(OUT_DIR, "airfoil_cmesh.vts")
        merged.save(out_path)
        print(f"Saved '{out_path}'  "
            f"dims={merged.dimensions}  "
            f"cell_data={list(merged.cell_data.keys())}  "
            f"point_data={list(merged.point_data.keys())}")
    elif output_type == 'pt':
        pass