from csv import reader
import airfrans as af
import pyvista as pv
import numpy as np
import os
import xml.etree.ElementTree as ET
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt
from scipy.spatial import KDTree


def _sanitize_vtk_block_name(name):
    return "".join(ch if (ch.isalnum() or ch in ("_", "-")) else "_" for ch in str(name))


def _vtk_extent_from_dimensions(dimensions):
    ni, nj, nk = [int(v) for v in dimensions]
    return f"0 {ni - 1} 0 {nj - 1} 0 {nk - 1}"


def _vtk_type_from_dtype(dtype):
    d = np.dtype(dtype)
    if np.issubdtype(d, np.floating):
        return "Float32" if d.itemsize <= 4 else "Float64"
    if np.issubdtype(d, np.signedinteger):
        return "Int32" if d.itemsize <= 4 else "Int64"
    if np.issubdtype(d, np.unsignedinteger):
        return "UInt32" if d.itemsize <= 4 else "UInt64"
    raise TypeError(f"Unsupported dtype for VTK ASCII export: {d}")


def _normalize_dataarray_for_vtk(arr, expected_tuples, array_name):
    arr = np.asarray(arr)
    if arr.ndim == 0:
        raise ValueError(f"Data array '{array_name}' is scalar; expected tuple array.")

    if arr.ndim == 1:
        tuple_count = arr.shape[0]
        num_components = 1
        out = arr.reshape((-1, 1))
    else:
        tuple_count = arr.shape[0]
        num_components = int(np.prod(arr.shape[1:]))
        out = arr.reshape((tuple_count, num_components))

    if tuple_count != expected_tuples:
        raise ValueError(
            f"Data array '{array_name}' tuple count mismatch: expected {expected_tuples}, got {tuple_count}."
        )
    return out, num_components


def _format_ascii_numeric(value, is_float):
    if is_float:
        return f"{float(value):.16g}"
    return str(int(value))


def _dataarray_to_ascii_block(data_2d):
    is_float = np.issubdtype(data_2d.dtype, np.floating)
    lines = []
    for row in data_2d:
        lines.append(" ".join(_format_ascii_numeric(v, is_float) for v in row))
    return "\n".join(lines)


def _write_structured_grid_vts_ascii(grid, vts_path):
    ni, nj, nk = [int(v) for v in grid.dimensions]
    n_points = ni * nj * nk
    n_cells = (ni - 1) * (nj - 1) * (1 if nk == 1 else (nk - 1))
    extent = _vtk_extent_from_dimensions((ni, nj, nk))

    point_data_xml = []
    for key in list(grid.point_data.keys()):
        data_2d, n_comp = _normalize_dataarray_for_vtk(grid.point_data[key], n_points, key)
        vtk_type = _vtk_type_from_dtype(data_2d.dtype)
        payload = _dataarray_to_ascii_block(data_2d)
        comp_attr = f' NumberOfComponents="{n_comp}"' if n_comp > 1 else ""
        point_data_xml.append(
            f'        <DataArray type="{vtk_type}" Name="{key}"{comp_attr} format="ascii">\n'
            f'{payload}\n'
            f'        </DataArray>'
        )

    cell_data_xml = []
    for key in list(grid.cell_data.keys()):
        data_2d, n_comp = _normalize_dataarray_for_vtk(grid.cell_data[key], n_cells, key)
        vtk_type = _vtk_type_from_dtype(data_2d.dtype)
        payload = _dataarray_to_ascii_block(data_2d)
        comp_attr = f' NumberOfComponents="{n_comp}"' if n_comp > 1 else ""
        cell_data_xml.append(
            f'        <DataArray type="{vtk_type}" Name="{key}"{comp_attr} format="ascii">\n'
            f'{payload}\n'
            f'        </DataArray>'
        )

    pts = np.asarray(grid.points, dtype=np.float64)
    if pts.shape != (n_points, 3):
        raise ValueError(
            f"StructuredGrid point shape mismatch: expected {(n_points, 3)}, got {pts.shape}."
        )
    points_payload = _dataarray_to_ascii_block(pts)

    if point_data_xml:
        point_data_block = "\n".join(["      <PointData>"] + point_data_xml + ["      </PointData>"])
    else:
        point_data_block = "      <PointData/>"

    if cell_data_xml:
        cell_data_block = "\n".join(["      <CellData>"] + cell_data_xml + ["      </CellData>"])
    else:
        cell_data_block = "      <CellData/>"

    xml_text = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="StructuredGrid" version="0.1" byte_order="LittleEndian">\n'
        f'  <StructuredGrid WholeExtent="{extent}">\n'
        f'    <Piece Extent="{extent}">\n'
        f'{point_data_block}\n'
        f'{cell_data_block}\n'
        '      <Points>\n'
        '        <DataArray type="Float64" NumberOfComponents="3" format="ascii">\n'
        f'{points_payload}\n'
        '        </DataArray>\n'
        '      </Points>\n'
        '    </Piece>\n'
        '  </StructuredGrid>\n'
        '</VTKFile>\n'
    )

    with open(vts_path, "w", encoding="utf-8") as f:
        f.write(xml_text)


def _write_multiblock_vtm_ascii(named_grids, out_dir, root_name):
    os.makedirs(out_dir, exist_ok=True)
    dataset_entries = []

    for idx, (block_name, grid) in enumerate(named_grids):
        safe_name = _sanitize_vtk_block_name(block_name)
        vts_file = f"{root_name}_{safe_name}.vts"
        vts_path = os.path.join(out_dir, vts_file)
        _write_structured_grid_vts_ascii(grid, vts_path)
        dataset_entries.append((idx, block_name, vts_file))

    vtm_lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="vtkMultiBlockDataSet" version="1.0" byte_order="LittleEndian">',
        '  <vtkMultiBlockDataSet>',
    ]
    for idx, block_name, vts_file in dataset_entries:
        vtm_lines.append(
            f'    <DataSet index="{idx}" name="{block_name}" file="{vts_file}"/>'
        )
    vtm_lines.extend([
        '  </vtkMultiBlockDataSet>',
        '</VTKFile>',
        '',
    ])

    vtm_path = os.path.join(out_dir, f"{root_name}.vtm")
    with open(vtm_path, "w", encoding="utf-8") as f:
        f.write("\n".join(vtm_lines))
    return vtm_path




def write_single_structured_cmesh_vtm(plot_dict, SIM_PATH, name="cmesh"):
    """
    Write a single-block structured C-mesh as a .vtm dataset.

    The function expects nodal coordinate channels "X" and "Y" in plot_dict,
    and writes all remaining same-shape channels as StructuredGrid point_data.

    Output path:
        {SIM_PATH}/constant/cmeshVTK/{name}.vtm
    """
    if "X" not in plot_dict or "Y" not in plot_dict:
        raise KeyError("plot_dict must contain 'X' and 'Y' coordinate arrays.")

    X = np.asarray(plot_dict["X"])
    Y = np.asarray(plot_dict["Y"])

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError(f"X and Y must be 2D arrays, got X.ndim={X.ndim}, Y.ndim={Y.ndim}.")
    if X.shape != Y.shape:
        raise ValueError(f"X and Y shape mismatch: X{X.shape} vs Y{Y.shape}.")

    ni, nj = X.shape
    nk = 1

    grid = pv.StructuredGrid()
    grid.dimensions = (ni, nj, nk)

    points = np.zeros((ni, nj, nk, 3), dtype=np.float64)
    points[:, :, 0, 0] = X
    points[:, :, 0, 1] = Y
    grid.points = points.reshape((-1, 3), order='F')

    for key, value in plot_dict.items():
        if key in ("X", "Y"):
            continue

        arr = np.asarray(value)
        if arr.shape[:2] != (ni, nj):
            continue

        if arr.ndim == 2:
            grid.point_data[key] = np.asarray(arr, dtype=np.float32).ravel(order='F')
        elif arr.ndim == 3:
            n_comp = arr.shape[2]
            if n_comp in (2, 3):
                vtk_arr = np.asarray(arr, dtype=np.float32).reshape((-1, n_comp), order='F')
                grid.point_data[key] = vtk_arr

    out_dir = os.path.join(SIM_PATH, "constant", "cmeshVTK")
    os.makedirs(out_dir, exist_ok=True)

    out_path = _write_multiblock_vtm_ascii([("cmesh", grid)], out_dir, name)
    print(f"Wrote structured C-mesh VTM: {out_path}")
    return out_path


def write_single_structured_orig_cmesh_vtm(master_tensor, master_x, master_y, ml_features, SIM_PATH, name="orig_cmesh"):
    """
    Write the original stitched C-mesh (from master tensor channels) as a
    single-block .vtm dataset under:
        {SIM_PATH}/constant/origCmeshVTK/{name}.vtm

    Args:
        master_tensor: np.ndarray with shape (C, ni, nj)
        master_x: np.ndarray with shape (ni, nj)
        master_y: np.ndarray with shape (ni, nj)
        ml_features: list of channel names with length C
    """
    master_tensor = np.asarray(master_tensor)
    master_x = np.asarray(master_x)
    master_y = np.asarray(master_y)

    if master_tensor.ndim != 3:
        raise ValueError(f"master_tensor must be 3D [C, ni, nj], got shape={master_tensor.shape}.")
    if master_x.ndim != 2 or master_y.ndim != 2:
        raise ValueError(f"master_x/master_y must be 2D, got {master_x.shape} and {master_y.shape}.")
    if master_x.shape != master_y.shape:
        raise ValueError(f"master_x/master_y shape mismatch: {master_x.shape} vs {master_y.shape}.")

    c, ni, nj = master_tensor.shape
    if master_x.shape != (ni, nj):
        raise ValueError(
            f"Coordinate shape {master_x.shape} incompatible with master_tensor spatial shape {(ni, nj)}."
        )
    if len(ml_features) != c:
        raise ValueError(f"ml_features length {len(ml_features)} must match channel count {c}.")

    grid = pv.StructuredGrid()
    grid.dimensions = (ni, nj, 1)

    points = np.zeros((ni, nj, 1, 3), dtype=np.float64)
    points[:, :, 0, 0] = master_x
    points[:, :, 0, 1] = master_y
    grid.points = points.reshape((-1, 3), order='F')

    for idx, channel_name in enumerate(ml_features):
        grid.point_data[channel_name] = np.asarray(master_tensor[idx], dtype=np.float32).ravel(order='F')

    out_dir = os.path.join(SIM_PATH, "constant", "origCmeshVTK")
    os.makedirs(out_dir, exist_ok=True)

    out_path = _write_multiblock_vtm_ascii([("orig_cmesh", grid)], out_dir, name)
    print(f"Wrote original stitched C-mesh VTM: {out_path}")
    return out_path


def write_stitched_raw_subgrids_vtm(
    subgrids_or_merged,
    SIM_PATH,
    name="stitched_cmesh_raw",
    fields_to_include=None,
    output_subdir="origCmeshVTK",
    z_ref=0.0,
):
    """
    Write stitched raw C-mesh to VTM.

    Accepts either:
      1) chain-ordered list of (name, StructuredGrid) blocks, or
      2) a pre-merged StructuredGrid.
    """
    if isinstance(subgrids_or_merged, pv.StructuredGrid):
        merged = subgrids_or_merged
    elif isinstance(subgrids_or_merged, (list, tuple)):
        merged = concatenate_blocks(list(subgrids_or_merged), z_ref=z_ref)
    else:
        raise TypeError(
            "subgrids_or_merged must be a StructuredGrid or list of (name, StructuredGrid) tuples"
        )

    if fields_to_include is not None:
        keep = set(fields_to_include)
        for key in list(merged.point_data.keys()):
            if key not in keep:
                merged.point_data.remove(key)
        for key in list(merged.cell_data.keys()):
            if key not in keep:
                merged.cell_data.remove(key)

    out_dir = os.path.join(SIM_PATH, "constant", output_subdir)
    os.makedirs(out_dir, exist_ok=True)

    out_path = _write_multiblock_vtm_ascii([("stitched_raw_cmesh", merged)], out_dir, name)
    print(f"Wrote stitched raw C-mesh VTM: {out_path}")
    return out_path


def write_sampled_blocks_vtm(
    sampled_subgrids,
    SIM_PATH,
    name="sampled_blockmesh_fields",
    fields_to_include=None,
    output_subdir="origCmeshVTK",
):
    """
    Export per-block sampled StructuredGrids to a single MultiBlock VTM.

    Each block keeps the original blockMesh topology and may carry sampled field
    data, so sampling quality can be inspected block-by-block in ParaView.
    """
    keep = set(fields_to_include) if fields_to_include is not None else None
    named_grids = []

    for block_name, grid in sampled_subgrids:
        g = grid.copy()
        if keep is not None:
            for key in list(g.point_data.keys()):
                if key not in keep:
                    g.point_data.remove(key)
            for key in list(g.cell_data.keys()):
                if key not in keep:
                    g.cell_data.remove(key)
        named_grids.append((block_name, g))

    out_dir = os.path.join(SIM_PATH, "constant", output_subdir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = _write_multiblock_vtm_ascii(named_grids, out_dir, name)
    print(f"Wrote possibly sampled block-level VTM to ASCII files: {out_path}")
    return out_path


def _parse_vtk_extent(extent_text):
    values = [int(token) for token in extent_text.split()]
    if len(values) != 6:
        raise ValueError(f"Expected 6 integers in VTK extent, got: {extent_text!r}")
    i0, i1, j0, j1, k0, k1 = values
    return i0, i1, j0, j1, k0, k1


def _structured_grid_dimensions_from_extent(extent_text):
    i0, i1, j0, j1, k0, k1 = _parse_vtk_extent(extent_text)
    return (i1 - i0 + 1, j1 - j0 + 1, k1 - k0 + 1)


def read_ascii_structured_grid_from_vts(vts_path):
    """
    Read an ASCII VTS StructuredGrid directly from XML without using pv.read.

    This loader is intentionally narrow in scope: it reads the point coordinates
    from the <Points><DataArray ... format="ascii"> section and builds a
    pyvista.StructuredGrid from those coordinates.
    """
    root = ET.parse(vts_path).getroot()
    if root.tag != "VTKFile":
        raise ValueError(f"Unexpected root tag {root.tag!r} in {vts_path}")
    if root.attrib.get("type") != "StructuredGrid":
        raise ValueError(f"{vts_path} is not a StructuredGrid VTS file.")

    structured = root.find("StructuredGrid")
    if structured is None:
        raise ValueError(f"Missing <StructuredGrid> in {vts_path}")

    piece = structured.find("Piece")
    if piece is None:
        raise ValueError(f"Missing <Piece> in {vts_path}")
    if "Extent" not in piece.attrib:
        raise ValueError(f"Missing Piece Extent in {vts_path}")

    dimensions = _structured_grid_dimensions_from_extent(piece.attrib["Extent"])
    points_node = piece.find("Points")
    if points_node is None:
        raise ValueError(f"Missing <Points> section in {vts_path}")

    data_array = points_node.find("DataArray")
    if data_array is None:
        raise ValueError(f"Missing point DataArray in {vts_path}")
    if data_array.attrib.get("format") != "ascii":
        raise ValueError(
            f"Only ASCII point DataArray is supported, got format={data_array.attrib.get('format')!r} in {vts_path}"
        )

    n_comp = int(data_array.attrib.get("NumberOfComponents", "1"))
    if n_comp != 3:
        raise ValueError(f"Expected 3 point components in {vts_path}, got {n_comp}")

    raw_text = data_array.text or ""
    points = np.fromstring(raw_text, sep=" ", dtype=np.float64)
    expected_values = dimensions[0] * dimensions[1] * dimensions[2] * 3
    if points.size != expected_values:
        raise ValueError(
            f"Point count mismatch in {vts_path}: expected {expected_values} values, got {points.size}"
        )

    grid = pv.StructuredGrid()
    grid.dimensions = dimensions
    grid.points = points.reshape((-1, 3))
    return grid


def read_structured_grids_from_vtm_ascii_vts(vtm_path):
    """
    Read a blockMesh-style VTM index and load each referenced ASCII VTS block.

    The VTM file is treated only as an index of block names to .vts files.
    Each block is then parsed directly from XML using
    read_ascii_structured_grid_from_vts().
    """
    root = ET.parse(vtm_path).getroot()
    if root.tag != "VTKFile":
        raise ValueError(f"Unexpected root tag {root.tag!r} in {vtm_path}")
    if root.attrib.get("type") != "vtkMultiBlockDataSet":
        raise ValueError(f"{vtm_path} is not a vtkMultiBlockDataSet VTM file.")

    multiblock = root.find("vtkMultiBlockDataSet")
    if multiblock is None:
        raise ValueError(f"Missing <vtkMultiBlockDataSet> in {vtm_path}")

    base_dir = os.path.dirname(vtm_path)
    grids = []
    print(f"\nLoaded VTM index: {vtm_path}")

    for dataset in multiblock.findall("DataSet"):
        file_name = dataset.attrib.get("file")
        if not file_name:
            continue
        name = dataset.attrib.get("name") or os.path.splitext(os.path.basename(file_name))[0]
        vts_path = os.path.join(base_dir, file_name)
        grid = read_ascii_structured_grid_from_vts(vts_path)
        print(
            f"  Loaded StructuredGrid '{name}' from ASCII VTS  dims={grid.dimensions}  "
            f"pts={grid.n_points}  cells={grid.n_cells}"
        )
        grids.append((name, grid))

    if not grids:
        raise ValueError(f"No VTS blocks referenced in {vtm_path}")
    return grids

def read_structured_grids_from_vtm(vtm_path):
    """Reads a blockMesh-style VTM file and returns all StructuredGrids found."""
    grids = read_structured_grids_from_vtm_ascii_vts(vtm_path)
    if not grids:
        raise ValueError("No StructuredGrids found in the VTM file.")
    print(f"\nTotal structured grids found: {len(grids)}")

    # Debug print successive j-line deltas at i=0 and i=max for block_5.
    print("Debug: Block 'block_5' successive j-line deltas at i=0 and i=max:")
    for name, grid in grids:
        if name == "block_5":
            ni, nj, nk = grid.dimensions
            pts3d = np.asarray(grid.points).reshape((ni, nj, nk, 3), order='F')
            max_j = min(6, nj - 1)
            for j in range(max_j):
                p_i0_a = pts3d[0, j, 0, :]
                p_i0_b = pts3d[0, j + 1, 0, :]
                p_imax_a = pts3d[ni - 1, j, 0, :]
                p_imax_b = pts3d[ni - 1, j + 1, 0, :]
                delta_i0 = np.linalg.norm(p_i0_b - p_i0_a)
                delta_imax = np.linalg.norm(p_imax_b - p_imax_a)
                print(
                    f"Block '{name}' j={j}->{j + 1}: "
                    f"|pti0_b | = {p_i0_b }, "
                    f"pti0_a | = {p_i0_a }, "
                    f"|delta at i=0| = {delta_i0:.12e}, "
                    f"pimax_b | = {p_imax_b }, "
                    f"pimax_a | = {p_imax_a }, "
                    f"|delta at i=max| = {delta_imax:.12e}"
                )


    return grids

def read_openfoam_results(foam_path, fields_to_keep=['U','p','nut','wallShearStress']):
    """Reads an OpenFOAM case and returns the MultiBlock dataset at the last time step."""
    reader = pv.POpenFOAMReader(foam_path)


    # 2. Disable ALL arrays to prevent reading unwanted .gz files
    reader.disable_all_cell_arrays()
    reader.disable_all_point_arrays()
    # Need to enable patch arrays otherwise 0 blocks are read, 
    reader.enable_all_patch_arrays()
     
    # 3. Selectively enable only the fields you care about
    for field in fields_to_keep:
        # OpenFOAM data usually lives in cell arrays, but we enable both to be safe
        reader.enable_cell_array(field)
        reader.enable_point_array(field)
        if hasattr(reader, 'enable_patch_array'):
            reader.enable_patch_array(field)
        print(f"Enabled field '{field}' for reading.")

    # Skip time 0 — initial conditions use $variable substitution that PyVista
    # cannot parse. Read only the final (solved) time step instead.
    times = reader.time_values
    print(f"Available time steps: {times}")
    reader.set_active_time_value(times[-1])
    print(f"Reading time step: {times[-1]}")
    mesh = reader.read()  # Use the reader to get Multiblock unstructured mesh
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

    def find_named_block(mb, target_name):
        for i in range(mb.n_blocks):
            name = mb.get_block_name(i) or str(i)
            child = mb[i]
            if child is None:
                continue
            if name == target_name:
                return child
            if isinstance(child, pv.MultiBlock):
                result = find_named_block(child, target_name)
                if result is not None:
                    return result
        return None

    def count_block_sizes(mb):
        if not isinstance(mb, pv.MultiBlock):
            return int(getattr(mb, "n_points", 0)), int(getattr(mb, "n_cells", 0))
        total_points = 0
        total_cells = 0
        for i in range(mb.n_blocks):
            child = mb[i]
            if child is None:
                continue
            if isinstance(child, pv.MultiBlock):
                p, c = count_block_sizes(child)
                total_points += p
                total_cells += c
            else:
                total_points += int(getattr(child, "n_points", 0))
                total_cells += int(getattr(child, "n_cells", 0))
        return total_points, total_cells

    print_block(mesh)
    internal_mesh = find_named_block(mesh, "internalMesh")
    if internal_mesh is None:
        raise ValueError("Could not find 'internalMesh' in OpenFOAM reader output.")

    print(" Using only 'internalMesh' and ignoring boundary patches before combine.")
    print(" Now combining internalMesh into a single mesh for interpolation...this is slow for large meshes.")
    pre_combine_points, pre_combine_cells = count_block_sizes(internal_mesh)
    print(f" Number of points before combining: {pre_combine_points}  cells before combining: {pre_combine_cells}")
    if isinstance(internal_mesh, pv.MultiBlock):
        mesh = internal_mesh.combine(merge_points=True, tolerance=1e-11)
    else:
        mesh = internal_mesh.copy()
    post_combine_points = mesh.n_points
    post_combine_cells = mesh.n_cells

    print(f" Number of points after combining: {post_combine_points}  cells after combining: {post_combine_cells}")
    print(f" Change in points: {post_combine_points - pre_combine_points}  change in cells: {post_combine_cells - pre_combine_cells}    ")
    print(" Converting cell data to point data for smooth interpolation...")
    mesh = mesh.cell_data_to_point_data()

    



    print(f" Finished reading and merging OpenFOAM results.  {foam_path}")
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
        w = b['w'][:2, :]   # [Ni]  for wall shear stress in X,Y directions (take only first 2 channels if w has more)
        
        # Deduplication: Drop the last i-node for every block EXCEPT the final wake block
        if i < len(BLOCK_ORDER) - 1:
            data_parts.append(d[:, :-1, :])
            x_parts.append(x[:-1, :])
            y_parts.append(y[:-1, :])
            w_parts.append(w[:, :-1]) # <-- Slice axis 1, not axis 0
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

    return final_data, final_x, final_y, final_w



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

def resample_block_pchip(block_data, block_x, block_y, target_ni=256, block_name="Unknown"):
    """
    block_data: [Channels, Ni, Nj]
    block_x, block_y: [Ni, Nj]
    Returns perfectly interpolated data and coordinates to target_ni length.
    """
    C, Ni, Nj = block_data.shape
    
    # --- 1. ROBUST CLEANING WITH DEBUGGING ---
    for j in range(Nj):
        # A. COORDINATE CHECK
        for name, coord in [("X", block_x), ("Y", block_y)]:
            mask = ~np.isfinite(coord[:, j])
            if mask.any():
                num_bad = mask.sum()
                if mask.all():
                    print(f"❌ CRITICAL: {block_name} - {name} coord at j={j} is ENTIRELY non-finite!")
                else:
                    print(f"⚠️ DEBUG: {block_name} - {name} coord has {num_bad}/{Ni} NaNs at j={j}. Patching...")
                    idx = np.arange(Ni)
                    coord[mask, j] = np.interp(idx[mask], idx[~mask], coord[~mask, j])

        # B. FLOW CHANNEL CHECK
        for c in range(C):
            mask = ~np.isfinite(block_data[c, :, j])
            if mask.any():
                num_bad = mask.sum()
                if mask.all():
                    # If the entire row is NaN, interpolation is impossible
                    print(f"🔥 FATAL: {block_name} - Channel {c} at j={j} is 100% NaN. PCHIP will fail.")
                else:
                    print(f"⚠️ DEBUG: {block_name} - Channel {c} has {num_bad}/{Ni} NaNs at j={j}. Patching...")
                    idx = np.arange(Ni)
                    block_data[c, mask, j] = np.interp(idx[mask], idx[~mask], block_data[c, ~mask, j])

    # --- 2. PCHIP INTERPOLATION ---
    # (Rest of the function remains the same)
    old_s = np.linspace(0, 1, Ni)
    new_s = np.linspace(0, 1, target_ni)
    new_data = np.zeros((C, target_ni, Nj), dtype=np.float32)
    new_x, new_y = np.zeros((target_ni, Nj)), np.zeros((target_ni, Nj))

    try:
        for j in range(Nj):
            new_x[:, j] = PchipInterpolator(old_s, block_x[:, j])(new_s)
            new_y[:, j] = PchipInterpolator(old_s, block_y[:, j])(new_s)
            for c in range(C):
                new_data[c, :, j] = PchipInterpolator(old_s, block_data[c, :, j])(new_s)
    except ValueError as e:
        print(f"💥 PCHIP CRASHED in {block_name}: {e}")
        # Hint: This usually happens if old_s or the data contains NaNs after cleaning
        raise

    return new_data, new_x, new_y

def extract_and_resample_wss(block_subgrid, target_ni):
    """
    Extracts wallShearStress from the j=0 wall and resamples it 
    to match the new 1024 topology.
    """
    ni, nj, nk = block_subgrid.dimensions
    
    # 1. Grab the raw WSS vector field (usually X, Y, Z components)
    if 'wallShearStress' not in block_subgrid.point_data:
        # If this is a wake block with no wall, just return zeros
        return np.zeros((3, target_ni), dtype=np.float32)
        
    wss_flat = block_subgrid.point_data['wallShearStress']
    
    # 2. Reshape to (Ni, Nj, 3 components) and slice strictly at j=0
    wss_2d = wss_flat.reshape((ni, nj, nk, 3), order='F')[:, :, 0, :]
    wss_wall_true_length = wss_2d[:, 0, :] # Shape: [Ni, 3]
    
    # 3. 1D PCHIP to the target length
    old_s = np.linspace(0, 1, ni)
    new_s = np.linspace(0, 1, target_ni)
    
    wss_resampled = np.zeros((3, target_ni), dtype=np.float32)
    for comp in range(3): # For X, Y, Z shear components
        interp = PchipInterpolator(old_s, wss_wall_true_length[:, comp])
        wss_resampled[comp, :] = interp(new_s)
        
    return wss_resampled



def calculate_and_append_sdf(master_tensor, master_x, master_y, k=0.5):
    """
    Calculates the SDF using the NumPy coordinate grids and appends 
    exp_sdf as a new channel to the master_tensor.
    """
    print("\nCalculating SDF on the 1024x216 master grid...")
    wake_length_start = 128
    airfoil_start = wake_length_start
    airfoil_end = 1024 - 127 # Exclude the trailing wake block
    # 1. Extract the Wall Coordinates
    # In your stitched topology, the wall is strictly at index j=0
    wall_x = master_x[airfoil_start:airfoil_end, 0]  # Shape: (1024,)
    wall_y = master_y[airfoil_start:airfoil_end, 0]  # Shape: (1024,)
    wall_xy = np.column_stack((wall_x, wall_y)) # Shape: (1024, 2)
    
    # 2. Flatten all coordinates for the KDTree query
    all_x = master_x.flatten()
    all_y = master_y.flatten()
    all_xy = np.column_stack((all_x, all_y))    # Shape: (221184, 2)
    
    # 3. Query the KDTree
    tree = KDTree(wall_xy)
    sdf_flat, _ = tree.query(all_xy, workers=-1)
    
    # 4. Reshape back to the 2D spatial grid
    sdf_2d = sdf_flat.reshape(master_x.shape)   # Shape: (1024, 216)
    sdf_fixed = np.nan_to_num(sdf_2d, nan=0.0).astype(np.float32)
    
    # 5. Calculate the Exponential SDF
    exp_sdf_2d = np.exp(-k * sdf_fixed).astype(np.float32)
    
    print(f"SDF Range: min={sdf_fixed.min():.6f}, max={sdf_fixed.max():.6f}, mean={sdf_fixed.mean():.6f}")
    print(f"Exp_SDF Range: min={exp_sdf_2d.min():.6f}, max={exp_sdf_2d.max():.6f}")
    
    # 6. Append exp_sdf to the master_tensor
    # We add a dummy channel dimension [1, 1024, 216] so it concatenates properly
    exp_sdf_expanded = np.expand_dims(exp_sdf_2d, axis=0)
    master_tensor_updated = np.concatenate([master_tensor, exp_sdf_expanded], axis=0)
    
    print(f"Updated Master Tensor Shape: {master_tensor_updated.shape}")
    
    return master_tensor_updated, sdf_fixed, exp_sdf_2d


def debug_plot_raw_blocks(blocks_dict):
    """Plots the raw, un-interpolated data from specific blocks to find artifacts."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Let's look at the beginning of the wake (Block 0) 
    # and the end of the wake (Block 1)
    for i, b_name in enumerate(['block_0', 'block_1']):
        block = blocks_dict[b_name]
        # Extract raw nut or U_x from the vtk/pyvista object
        # Using .point_data if you've already moved it there, or .cell_data
        raw_val = block.point_data['U_of_x'].reshape(block.dimensions[0], block.dimensions[1])
        
        im = axes[i].imshow(raw_val.T, origin='lower', aspect='auto', cmap='jet')
        axes[i].set_title(f"RAW {b_name} - U_x")
        plt.colorbar(im, ax=axes[i])
        
    plt.tight_layout()
    plt.show()

def debug_te_junction(sampled_blocks):
    """Inspects the TE transition between airfoil blocks and wake blocks."""
    # Airfoil ends at the end of block_5 (lower) and starts at the beginning of block_4 (upper)
    # The wake starts at block_0 and ends at block_1
    
    print("\n--- Trailing Edge Junction Debug ---")
    
    # Check Lower TE (Block 5 end vs Block 0 start)
    b5_end_u = sampled_blocks['block_5']['data'][ML_FEATURES.index("U_of_x"), -1, 0]
    b0_start_u = sampled_blocks['block_0']['data'][ML_FEATURES.index("U_of_x"), 0, 0]
    
    # Check Upper TE (Block 4 start vs Block 1 end)
    b4_start_u = sampled_blocks['block_4']['data'][ML_FEATURES.index("U_of_x"), 0, 0]
    b1_end_u = sampled_blocks['block_1']['data'][ML_FEATURES.index("U_of_x"), -1, 0]

    print(f"Lower TE Junction: Block_5_end={b5_end_u:.4f} | Block_0_start={b0_start_u:.4f}")
    print(f"Upper TE Junction: Block_4_start={b4_start_u:.4f} | Block_1_end={b1_end_u:.4f}")
    
    if abs(b5_end_u - b0_start_u) > 1e-3:
        print("!!! ALERT: Discontinuity detected at Lower Trailing Edge!")

from scipy.spatial import cKDTree
import numpy as np
import matplotlib.pyplot as plt

def plot_geometric_drift(foam_mesh, vts_blocks_dict, block_names):
    print("\nBuilding KDTree for OpenFOAM points...")
    # Create a fast search tree of the true 64-bit OpenFOAM coordinates
    tree = cKDTree(foam_mesh.points)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Geometric Drift (VTS vs OpenFOAM) - Absolute Distance", fontsize=16)
    axes = axes.flatten()
    
    for i, name in enumerate(block_names):
        ax = axes[i]
        block = vts_blocks_dict[name]
        ni, nj, nk = block.dimensions
        
        # 1. Find the nearest OpenFOAM node for every VTS node
        distances, nearest_indices = tree.query(block.points, k=1)
        
        # 2. Extract the exact matching OpenFOAM coordinates
        foam_nearest_pts = foam_mesh.points[nearest_indices]
        
        # 3. Calculate exact dx and dy (if you want to inspect directional drift)
        dx = block.points[:, 0] - foam_nearest_pts[:, 0]
        dy = block.points[:, 1] - foam_nearest_pts[:, 1]
        
        # We will plot the absolute magnitude of the drift (distances)
        # CRITICAL: Use Fortran ordering to match your grid!
        error_3d = distances.reshape((ni, nj, nk), order='F')
        error_2d = error_3d[:, :, 0]
        
        # Plot the error heatmap
        im = ax.imshow(error_2d.T, origin='lower', aspect='auto', cmap='magma')
        
        # Print the max error in the title
        ax.set_title(f"{name} Max Drift: {error_2d.max():.2e} m")
        plt.colorbar(im, ax=ax)
        
    plt.tight_layout()
    plt.show()



def plot_block_scalar(raw_vtk_blocks, BLOCK_ORDER,scalar_name='U_of_x'):

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Raw VTK Block Data - {scalar_name}", fontsize=16)
    axes = axes.flatten()

    for i, name in enumerate(BLOCK_ORDER):
        ax = axes[i]
        block = raw_vtk_blocks[name]
        # 1. Grab the correct dimensions for THIS specific block first!
        ni, nj, nk = block.dimensions
        # Extract coordinates (slice the first Z-plane)
        block_x = block.points[:, 0].reshape(ni, nj, nk)[:, :, 0]
        block_y = block.points[:, 1].reshape(ni, nj, nk)[:, :, 0]
        
        # Reshape the 1D point data back to the grid dimensions
        ni, nj, nk = block.dimensions
        print(f"Plotting raw data for '{name}' with dimensions (ni={ni}, nj={nj}, nk={nk})")
        # Let's look at U_of_x or nut to find that artifact
        data_3d = block.point_data[scalar_name].reshape(ni, nj, nk, order='F')
        data_2d = data_3d[:, :, 0]

        # Use pcolormesh with the block's own X,Y to see it in physical space
        # (Or imshow(data_2d.T) to see it in latent space)
        im = ax.imshow(data_2d.T, origin='lower', aspect='auto', cmap='turbo')
        ax.set_title(f"{name} (ni={ni}, nj={nj})")
        fig.colorbar(im, ax=ax)

    fig.tight_layout()
    plt.show()
    plt.close(fig)

if __name__ == "__main__":

    AF_ROOT = "/home/timm/Projects/PIML/Dataset"

    SIM_PATH = "/home/timm/storage/AF_NO_DATASET/OF_dataset/airFoil2D_SST_87.491_11.397_2.605_5.288_1.0_15.864"
    sim_name = os.path.basename(SIM_PATH)
    VTM_PATH = f"{SIM_PATH}/constant/blockMeshVTK/blockMesh.vtm"
    FOAM_PATH = f"{SIM_PATH}/touch.foam"

    OUT_DIR = "structured_block"

    output_type = 'vts'

    if not os.path.exists(FOAM_PATH):
        os.system(f"touch {FOAM_PATH}")  # create empty file to satisfy PyVista reader
    
    simulation = af.Simulation(root=AF_ROOT, name=sim_name)
    U_inf = simulation.inlet_velocity
    nu = simulation.NU
    rho = simulation.RHO
    reynolds = rho * U_inf / nu

    foam_combined = read_openfoam_results(FOAM_PATH)

    # Merge all OpenFOAM blocks into one UnstructuredGrid so we can call .interpolate()
    # use_all_points=True keeps ghost/boundary points; progress_bar for large meshes
    print("\nMerging OpenFOAM blocks into a single mesh for interpolation...")
    print(f"\nCombined OpenFOAM mesh: type={type(foam_combined).__name__}  "
          f"pts={foam_combined.n_points}  cells={foam_combined.n_cells}")
    print(f"Available arrays: {foam_combined.array_names}")

    foam_combined["Cp"] = foam_combined["p"] / (0.5 * U_inf**2)  # Note no rho because incompressible Openfoam results 
    foam_combined["nut_ratio_cube_root"] = np.cbrt(foam_combined["nut"]+1e-12 / nu)
    foam_combined["U_of_x"] = foam_combined["U"][:, 0]  # X-component of velocity
    foam_combined["U_of_y"] = foam_combined["U"][:, 1]  # Y-component of velocity
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
    
    J_MAX = 216
    # extract subgrids with correct i extents, keeping all j points up to J_MAX
    print("\nExtracting subgrids with specified i extents and J_MAX limit...")
    subgrids_raw = []
    num_wake_cells = 0
    for bname in BLOCK_ORDER:
        grid = grid_by_name[bname]
        ni, nj, nk = grid.dimensions
        if bname == 'block_0':
            sub = grid.extract_subset([0, ni - 1, 0, min(J_MAX, nj - 1), 0, 1])
            num_wake_cells = sub.dimensions[0]  # length along i-axis
        elif bname == 'block_1':
            sub = grid.extract_subset([0, ni - 1, 0, min(J_MAX, nj - 1), 0, 1])
        else:
            sub = grid.extract_subset([0, ni - 1, 0, min(J_MAX, nj - 1), 0, 1])

        # resample to new std block size
        subgrids_raw.append((bname, sub))
        print(f"'{bname}': {grid.dimensions} → {sub.dimensions}  "
              f"pts={sub.n_points}  cells={sub.n_cells}")

    print("\nChain-stitching i-axis orientations:")
    subgrids = chain_stitch_orientation(subgrids_raw)

    # Keep only the fields we need in the source mesh
    CELL_FIELDS  = ["U", "p", "nut" , "Cp", "nut_ratio_cube_root"]           # cell-centred → cell_data
    POINT_FIELDS = ["p", "wallShearStress", "Cp", "nut_ratio_cube_root"]     # node-interpolated → point_data
    KEEP = set(CELL_FIELDS + POINT_FIELDS)
    for arr in list(foam_combined.array_names):
        if arr not in KEEP:
            if arr in foam_combined.cell_data:  foam_combined.cell_data.remove(arr)
            if arr in foam_combined.point_data: foam_combined.point_data.remove(arr)
    print(f"Retained arrays in source: {foam_combined.array_names}")

    # ── Sample each block, compute Jacobians, collect ─────────────────────────
   # Dictionary to hold our final resampled numpy arrays for each block
    sampled_blocks = {}
    # 1. Initialize empty list and strict exclusions
    ML_FEATURES = [] 
    EXCLUDE_FROM_VOLUMETRIC = ["wallShearStress", "U"] # Must exclude the raw U vector!

    raw_vtk_blocks = {}
    for name, sub in subgrids:
        print(f"Sampling '{name}'...")

        # Cell-centred fields: U, p, nut
        #cc = sub.cell_data Old way 
        #  # Convert to point data for sampling
        sub = sub.cell_data_to_point_data()
        sub_sampled = sub.sample(foam_combined,snap_to_closest_point=True)


        # Split the velocity vector into explicit scalar fields
        if 'U' in sub_sampled.point_data:
            sub_sampled.point_data['U_of_x'] = sub_sampled.point_data['U'][:, 0]
            sub_sampled.point_data['U_of_y'] = sub_sampled.point_data['U'][:, 1]
        # Define the EXACT order of your volumetric channels
        # (This order becomes the channel indices of your master_tensor)

        # Jacobian metric tensors at both points and cell centres for physics-informed interpolation
        jac_point,jac_center = compute_jacobian_metrics(sub)
        
     
        # Store the actual PyVista object 'out' before it gets converted to a numpy array
        raw_vtk_blocks[name] = sub_sampled.copy()
    # --- NEW PLOTTING BLOCK FOR RAW DIAGNOSIS ---

    plot_geometric_drift(foam_combined, raw_vtk_blocks, BLOCK_ORDER)

    plot_block_scalar(raw_vtk_blocks, BLOCK_ORDER,scalar_name='U_of_x')
    plot_block_scalar(raw_vtk_blocks, BLOCK_ORDER,scalar_name='U_of_y')
    plot_block_scalar(raw_vtk_blocks, BLOCK_ORDER,scalar_name='Cp')
    plot_block_scalar(raw_vtk_blocks, BLOCK_ORDER,scalar_name='nut_ratio_cube_root')
        
    # #     for key, arr in jac_point.items():
    # #         print(f"Adding point data '{key}' to '{name}' with shape {arr.shape} and range [{arr.min():.3e}, {arr.max():.3e}]")
    # #         # need both z-planes
    # #         nz = out.dimensions[2]
    # #         arr = np.tile(arr, nz)
    # #         out.point_data[key] = arr

    # #             # Lock the dynamic feature order on the first block
    # #     if not ML_FEATURES:
    # #         all_keys = list(out.point_data.keys())
    # #         # Sorting guarantees the same channel indices every single run
    # #         ML_FEATURES = sorted([k for k in all_keys if k not in EXCLUDE_FROM_VOLUMETRIC])
    # #         print(f"Locked in ML_FEATURES channel order: {ML_FEATURES}")   


    # #     block_data, block_x, block_y = extract_arrays(out, ML_FEATURES)

    # #             # Save to dictionary for concatenation
    # #     sampled_blocks[name] = {
    # #         'data': block_data, 'x': block_x, 'y': block_y
    # #     }

    # #     wss_blk = extract_and_resample_wss(sub, 1024)
    # #     sampled_blocks[name]['w'] = wss_blk


    # # master_tensor, master_x, master_y , master_w = concatenate_blocks_ml_tensor(sampled_blocks, BLOCK_ORDER)
    # # k_sdf = 5.0 # Adjust this parameter to control the decay rate of the exponential SDF
    # # master_tensor, sdf_grid, exp_sdf_grid = calculate_and_append_sdf( master_tensor, master_x, master_y, k=k_sdf )


    # # U_of_x = master_tensor[ML_FEATURES.index("U_of_x")]
    # # U_of_y = master_tensor[ML_FEATURES.index("U_of_y")]
    # # p_of   = master_tensor[ML_FEATURES.index("p")]
    # # nut_of = master_tensor[ML_FEATURES.index("nut")]



    # AOA = simulation.angle_of_attack * 180 / np.pi
    # U_inf = simulation.inlet_velocity
    # nu = simulation.NU
    # rho = simulation.RHO
    # reynolds = rho * U_inf / nu
    # log_re = np.log(reynolds)   

    # # 2. Normalize OpenFOAM CFD Data
    # U_x = U_of_x / U_inf
    # U_y = U_of_y / U_inf
    # Cp_of = p_of / (0.5 * U_inf**2)  # Note no rho because incompressible Openfoam results 
    
    # nut_eps = 1e-12
    # nut_ratio = np.clip(nut_of / nu, nut_eps, None)
    # nut_ratio_cuberoot = np.power(nut_ratio, 1/3).astype(np.float32)
    
    
    # # Create a dictionary out of your calculated features
    # plot_dict = {
    #     "X": master_x,
    #     "Y": master_y,
    #     "Cp": Cp_of, "U_x": U_x, "U_y": U_y,
    #     "nut": nut_of,
    #     "nut_ratio_cuberoot": nut_ratio_cuberoot,
    #     "sdf": sdf_grid,
    #     "exp_sdf": exp_sdf_grid,

    # }

    # debug_plot_raw_blocks(plot_dict)
