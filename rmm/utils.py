from tqdm import tqdm
import readdy
import numpy as np
import igraph as ig
import tifffile
from typing import List, Optional, Union
import open3d as o3d
from scipy.ndimage import distance_transform_edt as edt
from scipy.ndimage import binary_dilation, binary_erosion


def load_tif(path: str) -> np.ndarray:
    """
    Load a TIFF image and return as a numpy array. 
    :param path: Path to the TIFF file. Expected shape is (Z, -Y, X). 
    :return: Numpy array with shape (X, Y, Z).
    """
    img = np.array(tifffile.imread(path), dtype=np.float32)
    img = np.transpose(img, (2, 1, 0))
    img = np.flip(img, axis=1)
    return img

def permute_array(data):
    return [np.copy(data), np.rot90(data, axes=(0, 1)), np.rot90(data, axes=(0, 2))]

def combine_permuted_arrays(data):
    data[1] = np.rot90(data[1], axes=(1, 0))
    data[2] = np.rot90(data[2], axes=(2, 0))
    return np.logical_or.reduce(data)

def redirect_graph(g, root_vertex, towards_root=True):
    """Returns a directed graph with edges directed towards the centrosome."""
    g_undirected = g.as_undirected()
    g_directed = g.as_directed(mode="mutual")
    paths = g_undirected.get_shortest_paths(root_vertex, mode="in", output="vpath")

    # Assign the length of each vertex's minimum path to the corresponding vertex
    for i, path in enumerate(paths):
        g_directed.vs[i]['path_length'] = len(path)

    new_edges = set()
    for path in paths:
        vs_path = [g_undirected.vs[idx] for idx in path]
        for i in range(len(vs_path) - 1):
            if towards_root:
                new_edges.add((vs_path[i + 1].index, vs_path[i].index))
            else:
                new_edges.add((vs_path[i].index, vs_path[i + 1].index))

    # Remove all edges in g_directed
    g_directed.delete_edges(g_directed.get_edgelist())
    g_directed.add_edges(list(new_edges))

    return g_directed.simplify()

def dilate_mask_3d(data, dilation_array):
    """Dilate a 3D mask."""
    permuted_arrays = permute_array(data)
    for axis, permuted_array in enumerate(permuted_arrays):
        iterations = dilation_array[axis]
        structure = np.ones((2 * iterations + 1,) * 2)
        for slice_idx in range(permuted_array.shape[0]):
            permuted_array[slice_idx] = binary_dilation(permuted_array[slice_idx], structure=structure)
    dilated_data = combine_permuted_arrays(permuted_arrays)
    return dilated_data

def erode_mask_3d(data, radius, voxel_size=[0.111, 0.111, 0.111]):
    """Erode a 3D mask."""
    erosion_array = np.array([radius / vs for vs in voxel_size], dtype=int)
    permuted_arrays = permute_array(data)
    for axis, permuted_array in enumerate(permuted_arrays):
        iterations = erosion_array[axis]
        structure = np.ones((2 * iterations + 1,) * 2)
        for slice_idx in range(permuted_array.shape[0]):
            permuted_array[slice_idx] = binary_erosion(permuted_array[slice_idx], structure=structure)
    eroded_data = combine_permuted_arrays(permuted_arrays)
    return eroded_data

def polarize(g, root_vertex_id="C", mode="closest", min_vertices=3, centrosome_coordinate=None):
    """ Adds polarity to an undirected/depolarized microtubule graph network. """
    comps = g.components(mode="WEAK")
    # subgraphs = [g.subgraph(c) for c in comps if len(c) >= min_vertices]
    subgraphs = [g.subgraph(c) for c in comps]
    # subgraphs = g.decompose()
    gs = []

    if centrosome_coordinate is None:
        # Redirect the main centrosome graph
        g_centrosome = [sg for sg in subgraphs if any([v['type'] == root_vertex_id for v in sg.vs])][0]
        centrosome_coordinate = g_centrosome.vs.find(type=root_vertex_id)['coordinate']
        g_centrosome = redirect_graph(g_centrosome, g_centrosome.vs.find(type=root_vertex_id))
        gs.append(g_centrosome)
    
    sg_list = [sg for sg in subgraphs if all([v['type'] != root_vertex_id for v in sg.vs])]
    
    # Redirect disjoint subgraphs 
    for i in tqdm(range(len(sg_list)), desc="Polarizing Microtubules", leave=False):
        sg = sg_list[i]
        vs_deg1 = [v for v in sg.vs if v.degree() == 1]
        if len(vs_deg1) == 0:
            continue
        if len(sg.vs) < min_vertices:
            continue

        if mode == "closest":
            vs_coords = np.array([v['coordinate'] for v in vs_deg1])
            dists = np.linalg.norm(vs_coords - centrosome_coordinate, axis=1)
            min_idx = np.argmin(dists)
            v = vs_deg1[min_idx]
            sg_directed = redirect_graph(sg, v)
            gs.append(sg_directed)
        elif mode == "random":
            v = np.random.choice(vs_deg1)
            sg_directed = redirect_graph(sg, v)
            gs.append(sg_directed)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    return combine_and_simplify(gs)
    # return gs

#-- Newer version that polarizes only using the centrosome coordinate
# def polarize(g, min_vertices=3, centrosome_coordinate=None):
#     """ Adds polarity to an undirected/depolarized microtubule graph network. """
#     comps = g.components(mode="WEAK")
#     subgraphs = [g.subgraph(c) for c in comps if len(c) >= min_vertices]
#     # subgraphs = g.decompose()

#     # g_centrosome = [sg for sg in subgraphs if any([v['type'] == root_vertex_id for v in sg.vs])][0]
#     # sg_list = [sg for sg in subgraphs if all([v['type'] != root_vertex_id for v in sg.vs])]
#     # centrosome_coordinate = g_centrosome.vs.find(type=root_vertex_id)['coordinate']

#     # Redirect disjoint subgraphs 
#     gs = []
#     for i in tqdm(range(len(subgraphs)), desc="Polarizing Microtubules", leave=False):
#         sg = subgraphs[i]
#         vs_deg1 = [v for v in sg.vs if v.degree() == 1]
#         if len(vs_deg1) == 0:
#             continue


#         vs_coords = np.array([v['coordinate'] for v in vs_deg1])
#         dists = np.linalg.norm(vs_coords - centrosome_coordinate, axis=1)
#         min_idx = np.argmin(dists)
#         v = vs_deg1[min_idx]
#         sg_directed = redirect_graph(sg, v)
#         gs.append(sg_directed)
#         # elif mode == "random":
#         #     v = np.random.choice(vs_deg1)
#         #     sg_directed = redirect_graph(sg, v)
#         #     sg_list_disjoint.append(sg_directed)
#         # else:
#         #     raise ValueError(f"Unknown mode: {mode}")

#     # Redirect the main centrosome graph
#     # g_centrosome = redirect_graph(g_centrosome, g_centrosome.vs.find(type=root_vertex_id))
    
#     return combine_and_simplify(gs)

def combine_and_simplify(graphs: List[ig.Graph]) -> ig.Graph:
    """Combines multiple graphs into one and simplifies it."""
    if len(graphs) == 0:
        raise ValueError("No graphs to combine.")
    g_combined = graphs[0]
    for g in graphs[1:]:
        g_temp = g_combined.disjoint_union(g)
        g_combined = g_temp
    return g_combined.simplify()

def get_closest_vertex(g, coordinate, vertex_degree=None):
    """Returns the closest vertex to a given coordinate."""
    if vertex_degree is not None:
        if not isinstance(vertex_degree, list):
            vertex_degree = [vertex_degree]
        vs = [v for v in g.vs if v.degree() in vertex_degree]
    else:
        vs = g.vs

    if not vs:
        return None, None

    vs_coords = np.array([v['coordinate'] for v in vs])
    dists = np.linalg.norm(vs_coords - coordinate, axis=1)

    min_idx = np.argmin(dists)
    return vs[min_idx], dists[min_idx]

def extract_topology_graphs_from_frame(
    trajectory_file: str,
    frame_index: int = -1,
    particle_types: Optional[Union[str, List[str]]] = ["P1", "P2", "C"],
    *,
    exact_match: bool = False,
    exclude_types: Optional[List[str]] = None
) -> List[ig.Graph]:
    """
    Build igraph graphs for a single frame from a ReaDDy trajectory,
    using the same structure/attributes as in TopologyGraphs._setup.

    Vertex attributes on each graph:
      - 'uid'         : particle id (int)
      - 'type'        : particle type (str)
      - 'coordinate'  : [x, y, z] list

    Optional filtering:
      - particle_types: keep vertices whose type matches (exact or substring).
                        If None, no filtering is applied.
      - exclude_types : drop any vertices whose type is in this list.
      - exact_match   : if True, require exact type match; if False, substring match.

    Args:
        trajectory_file: Path to the .h5 trajectory.
        frame_index    : Frame to extract (supports negative indexing).
        particle_types : str or list of str, e.g. "mitochondria" or ["mito", "mt"].
        exact_match    : Use exact string equality if True, else substring matching.
        exclude_types  : List of types to always exclude.

    Returns:
        List[ig.Graph]: graphs for the requested frame (possibly filtered).
    """
    if exclude_types is None:
        exclude_types = []

    # Normalize input
    if isinstance(particle_types, str):
        particle_types = [particle_types]

    traj = readdy.Trajectory(trajectory_file)
    _times, all_topologies = traj.read_observable_topologies()
    all_particles = traj.read()

    n_frames = len(traj)
    if frame_index < 0:
        frame_index = n_frames + frame_index
    if not (0 <= frame_index < n_frames):
        raise IndexError(f"frame_index {frame_index} out of range [0, {n_frames-1}]")

    frame_particles = all_particles[frame_index]
    frame_topologies = all_topologies[frame_index]

    frame_graphs: List[ig.Graph] = []
    for top in frame_topologies:
        vs_top = top.particles
        es_top = top.edges

        ptypes = [frame_particles[v_id].type for v_id in vs_top]
        ppos = np.vstack([frame_particles[v_id].position for v_id in vs_top])
        puids = [frame_particles[v_id].id for v_id in vs_top]

        g = ig.Graph(n=len(vs_top), edges=es_top)
        g.vs["uid"] = puids
        g.vs["type"] = ptypes
        g.vs["coordinate"] = ppos.tolist()

        # Apply filtering only if particle_types is specified
        if particle_types is not None:
            to_remove = []
            for v in g.vs:
                vtype = v["type"]
                if vtype in exclude_types:
                    to_remove.append(v.index)
                    continue
                if exact_match:
                    if vtype not in particle_types:
                        to_remove.append(v.index)
                else:
                    if not any(pt in vtype for pt in particle_types):
                        to_remove.append(v.index)

            if len(to_remove) == len(g.vs):
                continue  # skip graphs with all vertices removed

            g.delete_vertices(to_remove)
            g.simplify()

        frame_graphs.append(g)

    print(f"Extracted {len(frame_graphs)} topology graphs")

    # Print the average number of vertices per graph
    if len(frame_graphs) > 0:
        avg_vertices = np.mean([len(g.vs) for g in frame_graphs])
        print(f"Average number of vertices per graph: {avg_vertices:.2f}")

    return combine_and_simplify(frame_graphs)

# def plot_graphs(graphs, 
#                 as_components=True, 
#                 with_arrows=False, 
#                 arrow_radius=0.01, 
#                 arrow_length_ratio=0.2,
#                 show_centrosome=False
#                 ):
#     """
#     Plot graphs in Open3D. Uses line sets for undirected graphs and arrows for directed graphs.

#     Args:
#         graphs: ig.Graph or list of ig.Graph
#         as_components: split into connected components if True
#         arrow_radius: radius of the arrow cylinder/cone
#         arrow_length_ratio: cone length relative to edge length
#     """
#     if isinstance(graphs, ig.Graph):
#         if as_components:
#             comps = graphs.connected_components(mode="WEAK")
#             graphs = [graphs.subgraph(c) for c in comps]
#             colors = np.random.rand(len(graphs), 3)
#         else:
#             colors = np.array([[1, 0, 1]])
#             graphs = [graphs]
#     else:
#         colors = np.random.rand(len(graphs), 3)

#     geoms = []
#     for i, g in enumerate(graphs):
#         coords = np.array(g.vs["coordinate"])
#         if g.is_directed() and with_arrows:
#             # Make arrows
#             for e in g.es:
#                 src, tgt = e.tuple
#                 p0, p1 = coords[src], coords[tgt]
#                 v = p1 - p0
#                 length = np.linalg.norm(v)
#                 if length == 0:
#                     continue
#                 # Create arrow mesh
#                 arrow = o3d.geometry.TriangleMesh.create_arrow(
#                     cylinder_radius=arrow_radius,
#                     cone_radius=arrow_radius * 1.5,
#                     cylinder_height=length * (1 - arrow_length_ratio),
#                     cone_height=length * arrow_length_ratio
#                 )
#                 arrow.paint_uniform_color(colors[i])

#                 # Align arrow with vector v
#                 arrow.translate([0, 0, 0])
#                 R = arrow.get_rotation_matrix_from_xyz([0, 0, 0])  # identity placeholder
#                 z = np.array([0, 0, 1.0])
#                 v_norm = v / length
#                 axis = np.cross(z, v_norm)
#                 if np.linalg.norm(axis) > 1e-6:
#                     axis /= np.linalg.norm(axis)
#                     angle = np.arccos(np.dot(z, v_norm))
#                     R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
#                 arrow.rotate(R, center=(0, 0, 0))
#                 arrow.translate(p0)
#                 geoms.append(arrow)
#         else:
#             # Use lines for undirected
#             line = o3d.geometry.LineSet()
#             line.points = o3d.utility.Vector3dVector(coords)
#             line.lines = o3d.utility.Vector2iVector(np.array(g.get_edgelist()))
#             line.paint_uniform_color(colors[i])
#             geoms.append(line)

#     if show_centrosome:
#         # Find the centrosome (type "C") and plot as sphere
#         for g in graphs:
#             for v in g.vs:
#                 if v['type'] == "C" or "centrosome" or "Centrosome":
#                     center = v['coordinate']
#                     sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
#                     sphere.translate(center)
#                     sphere.paint_uniform_color([1, 0, 0])
#                     geoms.append(sphere)

#     vis = o3d.visualization.Visualizer()
#     vis.create_window(width=1920, height=1080)

#     coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
#     vis.add_geometry(coord_frame)
#     for g in geoms:
#         vis.add_geometry(g)
#     vis.run()
#     vis.destroy_window()


def mask_to_sdf(mask: np.ndarray) -> np.ndarray:
    """
    Compute signed distance field (SDF; negative inside, positive outside).
    Uses EDT (Euclidean Distance Transform) inside/outside.
    """
    mask = mask.astype(bool)
    if mask.all():
        return -edt(mask).astype(np.float32)
    if (~mask).all():
        return edt(~mask).astype(np.float32)
    d_out = edt(~mask).astype(np.float32)
    d_in = edt(mask).astype(np.float32)
    return d_out - d_in

def combine_masks(masks, inversion_flags):
    """
    Combine multiple segmentation masks with optional inversion flags.
    """
    combined_mask = np.ones_like(masks[0], dtype=np.uint8)
    for mask, invert in zip(masks, inversion_flags):
        # Convert to boolean
        mask_bool = mask.astype(bool)
        if invert:
            mask_bool = ~mask_bool
        
        # Convert to zero-one
        mask = mask_bool.astype(np.uint8)
        combined_mask *= mask
    return combined_mask

def mask_graph(g: ig.Graph, mask: np.ndarray, voxel_size=np.array([0.111, 0.111, 0.111])) -> ig.Graph:
    """
    Remove vertices from graph g that are outside the given mask.
    """
    n_initial = len(g.vs)
    to_remove = []
    offset = np.array(mask.shape) * voxel_size / 2.0 + voxel_size / 2.0
    # coords = np.array(g.vs["coordinate"])

    inside_fn = lambda coord: mask[
        int((coord[0] + offset[0]) / voxel_size[0]),
        int((coord[1] + offset[1]) / voxel_size[1]),
        int((coord[2] + offset[2]) / voxel_size[2])
    ] > 0

    for i, v in enumerate(g.vs):
        coord = v["coordinate"]
        try:
            inside = inside_fn(coord)
        except IndexError:
            inside = False

        if not inside:
            to_remove.append(v.index)

    g.delete_vertices(to_remove)
    n_final = len(g.vs)
    print(f"Initial vertices: {n_initial}, Final vertices: {n_final}, Removed: {n_initial - n_final}")
    return g

if __name__ == "__main__":
    memfile = "/Users/earkfeld/Projects/readdy-made-models/data/membrane.tif"
    nucfile = "/Users/earkfeld/Projects/readdy-made-models/data/nucleus.tif"
    import tifffile as tiff
    mem_mask = tiff.imread(memfile)
    nuc_mask = tiff.imread(nucfile)
    combined_mask = combine_masks([mem_mask, nuc_mask], [False, True])
    combined_mask = combined_mask.astype(np.uint8) * 255
    
    # Save the mask
    tiff.imwrite("combined_mask.tif", combined_mask, imagej=True, metadata={'axes': 'ZYX'})
