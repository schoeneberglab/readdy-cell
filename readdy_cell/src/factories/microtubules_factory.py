import numpy as np
import igraph as ig
from typing import Optional
from tqdm import tqdm
from scipy.ndimage import uniform_filter
from skimage import io
from skimage.morphology import skeletonize
import os
import open3d as o3d

from src.core.model import Model
from src.core.model_utils import ModelUtils
from src.core.data_loader import DataLoader
from src.factories.base_factory import Factory

from rmm.utils import *
from rmm.model_builder import ModelBuilder

def plot_graphs(graphs, 
                as_components=True, 
                with_arrows=False, 
                arrow_radius=0.01, #0.01, 
                arrow_length_ratio=0.2 #0.2
                ):
    """
    Plot graphs in Open3D. Uses line sets for undirected graphs and arrows for directed graphs.

    Args:
        graphs: ig.Graph or list of ig.Graph
        as_components: split into connected components if True
        arrow_radius: radius of the arrow cylinder/cone
        arrow_length_ratio: cone length relative to edge length
    """
    if isinstance(graphs, ig.Graph):
        if as_components:
            comps = graphs.connected_components(mode="WEAK")
            graphs = [graphs.subgraph(c) for c in comps]
            colors = np.random.rand(len(graphs), 3)
        else:
            colors = np.array([[1, 0, 1]])
            graphs = [graphs]
    else:
        colors = np.random.rand(len(graphs), 3)

    geoms = []
    for i, g in enumerate(graphs):
        coords = np.array(g.vs["coordinate"])
        if g.is_directed() and with_arrows:
            # Make arrows
            for e in g.es:
                src, tgt = e.tuple
                p0, p1 = coords[src], coords[tgt]
                v = p1 - p0
                length = np.linalg.norm(v)
                if length == 0:
                    continue
                # Create arrow mesh
                arrow = o3d.geometry.TriangleMesh.create_arrow(
                    cylinder_radius=arrow_radius,
                    cone_radius=arrow_radius * 1.5,
                    cylinder_height=length * (1 - arrow_length_ratio),
                    cone_height=length * arrow_length_ratio
                )
                arrow.paint_uniform_color(colors[i])

                # Align arrow with vector v
                arrow.translate([0, 0, 0])
                R = arrow.get_rotation_matrix_from_xyz([0, 0, 0])  # identity placeholder
                z = np.array([0, 0, 1.0])
                v_norm = v / length
                axis = np.cross(z, v_norm)
                if np.linalg.norm(axis) > 1e-6:
                    axis /= np.linalg.norm(axis)
                    angle = np.arccos(np.dot(z, v_norm))
                    R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
                arrow.rotate(R, center=(0, 0, 0))
                arrow.translate(p0)
                geoms.append(arrow)
        else:
            # Use lines for undirected
            line = o3d.geometry.LineSet()
            line.points = o3d.utility.Vector3dVector(coords)
            line.lines = o3d.utility.Vector2iVector(np.array(g.get_edgelist()))
            line.paint_uniform_color(colors[i])
            geoms.append(line)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1080)
    for g in geoms:
        vis.add_geometry(g)
    vis.run()
    vis.destroy_window()


class MicrotubulesFactory(Factory, ModelUtils):
    """ Factory class for constructing Microtubule model objects. """

    DEFAULT_TOPOLOGY_TYPE = None
    DEFAULT_PARTICLE_TYPE = None
    DEFAULT_PARAMETERS = {
        "tags": ["boundary"],
        "flags": [],
        "radius": [0.05],
        "diffusion_constant": [0.0]
    }
    DEFAULT_VISUALIZATION_PROPERTIES = {
        "radii": {"all": DEFAULT_PARAMETERS["radius"]},
        "display_types": {"all": "FIBER"},
        "viz_types": {"all": 1001.0},
        "colors": {"all": "orchid"},
        "url": ""
    }

    def __init__(self, data_loader: DataLoader, model: Optional[Model] = None, *args, **kwargs):
        assert data_loader is not None, "Please provide a MitoTNTDataLoader object"
        super().__init__(data_loader)
        self.data_loader = data_loader
        self.model = model if model else self._get_default_model()

        self.g = None
        self.centrosome_coordinate = None

        self.nucleus = kwargs.get("nucleus", None)
        offset = kwargs.get("offset", None)
        # offset = kwargs.get("offset", np.array([7, 0, -2]))

        self.img_segmented = self.data_loader['microtubules_segmented']
        self.img_segmented = self.format_mask(self.img_segmented)
        if offset is not None:
            self.img_segmented = self.translate_mask(self.img_segmented, offset)
        self.img_unchanged = self.data_loader['microtubules']
        self.img_unchanged = self.format_mask(self.img_unchanged)
        if offset is not None:
            self.img_unchanged = self.translate_mask(self.img_segmented, offset)

        # Parameters
        # TODO: Set some of these up as properties
        self.centrosome_search_radius = 2
        self.centrosome_connect_radius = 2 # 6

        self.nuclear_distance_threshold = 10
        self.vertex_distance_threshold = 2.0
        self.vertex_degree_threshold = 3
        self.vertex_count_threshold = 5
        self._n_downsample = 2

        self._mask_membrane = True
        self._mask_nucleus = True
        self._save_skeleton = kwargs.get("save_skeleton", True)
        self._skeleton_img = None
        self.img_skeleton_path = None

        self.nucleus_mask = self.format_mask(self.data_loader['nucleus_mask'])
        self.membrane_mask = self.format_mask(self.data_loader['membrane_mask'])

        self.voxel_size, self.voxel_units = self.data_loader.get_voxel_scale()

    def run(self, filename: str, *args, **kwargs) -> Model:
        """ Construct and return a model."""
        if self._mask_membrane:
            # Invert the membrane mask and intersect with the microtubule mask
            # Format the mask to be boolean (black to True, white to False)
            mask = self.data_loader['membrane_mask'].astype(bool)
            self.img_segmented = self.img_segmented * self.format_mask(mask)

        if self._mask_nucleus:
            # intersect the nucleus mask with the microtubule mask
            mask = np.logical_not(self.data_loader['nucleus_mask'].astype(bool))
            self.img_segmented = self.img_segmented * self.format_mask(mask)

        self._get_skeleton_img()

        # self._get_skeleton_graph()
        # self._refine_graph()

        if self.nucleus is not None:
            self.locate_centrosome_with_nucleus()
        else:
            self.locate_centrosome()

        traj_name = filename # kwargs.get("filename", "/Users/earkfeld/Projects/mitosim/data/rmm_trajs/out")
        traj_file = traj_name + ".h5"

        overwrite = False
        if overwrite and os.path.exists(traj_file):
            os.remove(traj_file)
        
        # Build the model
        builder = ModelBuilder()
        builder.load(path=self.img_skeleton_path)
        self.box_origin = builder.box_origin
        
        if not os.path.exists(traj_file):
            builder.run(filename=traj_name, show_summary=False, visualize=True, remove_existing=True)
        
        # Extract the graph from the last frame of the trajectory
        g = extract_topology_graphs_from_frame(traj_file, frame_index=-1)

        coords = np.array(g.vs['coordinate'])
        print(np.mean(coords, axis=0))

        self.nucleus_mask = dilate_mask_3d(self.nucleus_mask, dilation_array=np.array([1, 1, 2]))

        mask = combine_masks([self.membrane_mask, self.nucleus_mask], inversion_flags=[False, True])
        self.g = mask_graph(g, mask)
        centrosome_coordinate = self._get_centrosome_coordinate(filename)
        self.g = polarize(self.g, centrosome_coordinate=centrosome_coordinate, mode="closest", min_vertices=3,)

        self.model.data = self.g

        return self.model
    
    def _get_centrosome_coordinate(self, filename):

        centrosome_indices = {
            "control": {
                0: np.array([152, 161, 40]),
                1: np.array([207, 150, 27]),
                2: np.array([109, 138, 32]),
                3: np.array([74, 113, 38]),
            },
            "nocodazole_30min": {
                0: np.array([161, 172, 32]),
                1: np.array([98, 177, 13]),
                2: np.array([123, 138, 38]),
                3: np.array([166, 104, 33]),
                4: np.array([110, 135, 28]),
            },
        }

        # Split at "cell" then remove trailing underscore
        basename = os.path.basename(filename)
        condition = basename.split("cell")[0].rstrip("_")
        cid = int(basename.split("cell")[1][1])
        if condition in centrosome_indices and cid in centrosome_indices[condition]:
            voxel_coord = centrosome_indices[condition][cid]
            print(f"{condition} cell {cid} centrosome voxel coord: {voxel_coord}")
            physical_coord = voxel_coord * self.voxel_size
            physical_coord += self.box_origin
            return physical_coord
        else:
            return None

    def _get_skeleton_img(self):
        """ Skeletonizes the binary image and returns a graph with vertices as pixel coordinates. """
        skeleton = skeletonize(self.img_segmented, method='lee')

        self._skeleton_img = skeleton

        if self._save_skeleton:
            self.save_skeleton_img(skeleton)

    def locate_centrosome_with_nucleus(self):
        """ Locates the centrosome via the brightest region in the nuclear region. """
        nucleus_coordinates = self.nucleus.data
        # nucleus_coordinates = self.nucleus
        voxel_scale, voxel_units = self.data_loader.get_voxel_scale()
        nucleus_coordinates = np.round(nucleus_coordinates / voxel_scale).astype(int)

        centrosome_intensity = 0
        reference_coordinate = np.zeros(3)

        nuc_dist_thresh = int(2 * self.nuclear_distance_threshold)
        cent_radius_thresh = int(2 * self.centrosome_search_radius)

        img_working_slice = np.zeros((nuc_dist_thresh,
                                      nuc_dist_thresh,
                                      nuc_dist_thresh))

        for coordinate in nucleus_coordinates:
            # Slice the image around the nucleus coordinate according to d_nucleus_coordinate
            x, y, z = coordinate
            z_min = max(z - nuc_dist_thresh, 0)
            z_max = min(z + nuc_dist_thresh, self.img_segmented.shape[2])
            y_min = max(y - nuc_dist_thresh, 0)
            y_max = min(y + nuc_dist_thresh, self.img_segmented.shape[1])
            x_min = max(x - nuc_dist_thresh, 0)
            x_max = min(x + nuc_dist_thresh, self.img_segmented.shape[0])

            img_slice = self.img_segmented[x_min:x_max, y_min:y_max, z_min:z_max]

            intensity_sum = uniform_filter(img_slice.astype(float), size=(cent_radius_thresh,
                                                                          cent_radius_thresh,
                                                                          cent_radius_thresh))

            if intensity_sum.shape[0] == 0:
                continue
            else:
                candidate_coordinate = np.unravel_index(np.argmax(intensity_sum), intensity_sum.shape)

                # Update the centrosome coordinate if the intensity is higher
                if intensity_sum[candidate_coordinate] > centrosome_intensity:
                    centrosome_intensity = intensity_sum[candidate_coordinate]
                    reference_coordinate = coordinate
                    img_working_slice = img_slice

        # Get the brightest region in the slice
        intensity_sum = uniform_filter(img_working_slice.astype(float), size=(cent_radius_thresh,
                                                                              cent_radius_thresh,
                                                                              cent_radius_thresh))
        slice_coordinate = np.unravel_index(np.argmax(intensity_sum), intensity_sum.shape)
        self.centrosome_coordinate = reference_coordinate - self.nuclear_distance_threshold + slice_coordinate

    def add_centrosome(self):
        """ Adds a centrosome node to the graph. """
        # Locate the centrosome and add to graph
        self.g.add_vertex(coordinate=self.centrosome_coordinate, type="C")
        print(f"Centrosome located at: {self.centrosome_coordinate}")

        # Get all coordinates of the microtubule vertices
        coordinates = np.array(self.g.vs['coordinate'])
        coordinates = np.delete(coordinates, -1, axis=0)
        centrosome_distances = np.linalg.norm(coordinates - self.centrosome_coordinate, axis=1)
        node_ids_within_dist = np.argwhere(centrosome_distances < self.centrosome_connect_radius).flatten()

        # Get the neighbors of the nodes within the distance threshold
        neighbors = []
        for node in node_ids_within_dist:
            candidate_neighbors = self.g.neighbors(node)
            for neighbor in candidate_neighbors:
                if neighbor not in node_ids_within_dist:
                    neighbors.append(neighbor)
        neighbors = list(set(neighbors))

        # Connect the centrosome to the neighbors & delete the nodes within the distance threshold
        centrosome_vs = self.g.vs[-1]
        for neighbor in neighbors:
            neighbor_v = self.g.vs[neighbor]
            self.g.add_edge(centrosome_vs, neighbor_v)
        self.g.delete_vertices(node_ids_within_dist)
        self.g.simplify()

    def rescale(self):
        """ Rescales graph coordinate attribute """
        voxel_scale, voxel_units = self.data_loader.get_voxel_scale()
        coordinates = np.array(self.g.vs['coordinate'])
        coordinates *= voxel_scale
        self.g.vs['coordinate'] = coordinates

    def locate_centrosome(self):
        """ Locate centrosome via spherical region with max intensity and returns centroid pixel indices. """

        # Identify candidate regions for the centrosome
        intensity_sum = uniform_filter(self.img_unchanged.astype(float),
                                       size=([self.centrosome_search_radius * 2]*3))

        # Find the region closest to the center of the image
        center = np.array(self.img_unchanged.shape) // 2
        distances = np.linalg.norm(np.argwhere(intensity_sum) - center, axis=1)
        centrosome_coordinate = np.argwhere(intensity_sum)[np.argmin(distances)]
        self.centrosome_coordinate = centrosome_coordinate


    def save_skeleton_img(self, img_skeleton):
        """ Save the skeletonized image to the parent microtubule frame directory. """
        img_segmented_path = self.data_loader.pathdict["microtubules_segmented"]
        self.img_skeleton_path = img_segmented_path.replace("segmented", "skeleton")

        # Reformat the skeleton image
        img = self.format_mask(img_skeleton) * 255

        # Save the skeleton image so true is white and false is black
        io.imsave(self.img_skeleton_path, img)

