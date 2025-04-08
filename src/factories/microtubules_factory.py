import numpy as np
import igraph as ig
from typing import Optional
from tqdm import tqdm
from scipy.ndimage import uniform_filter
from skimage import io
from skimage.morphology import skeletonize

from src.core.model import Model
from src.core.model_utils import ModelUtils
from src.core.data_loader import DataLoader
from src.factories.base_factory import Factory


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
        self.centrosome_connect_radius = 6

        self.nuclear_distance_threshold = 10
        self.vertex_distance_threshold = 2.0
        self.vertex_degree_threshold = 3
        self.vertex_count_threshold = 5
        self._n_downsample = 2

        self._mask_membrane = True
        self._mask_nucleus = True
        self._save_skeleton = kwargs.get("save_skeleton", True)

    def run(self, *args, **kwargs) -> Model:
        """ Construct and return a model."""
        # TODO: Fixme; "None" results in maximum downsamples
        n_downsample = kwargs.get("n_downsample", 2)
        assert n_downsample is None or isinstance(n_downsample, int), "n_downsample must either an integer or None"

        if self._mask_membrane:
            mask = self.data_loader['membrane_mask'].astype(bool)
            self.img_segmented = self.img_segmented * self.format_mask(mask)

        if self._mask_nucleus:
            mask = np.logical_not(self.data_loader['nucleus_mask'].astype(bool))
            self.img_segmented = self.img_segmented * self.format_mask(mask)

        self._get_skeleton_graph()
        self._refine_graph()

        if self.nucleus is not None:
            self.locate_centrosome_with_nucleus()
        else:
            self.locate_centrosome()

        self.add_centrosome()
        self.polarize()
        self.rescale()

        if n_downsample and n_downsample > 0:
            self.g = self.downsample_graph(self.g, n_downsample)

        self.model.data = self.g
        return self.model


    @staticmethod
    def downsample_graph(g, n):

        def mean_coordinate(attrs):
            return np.mean(attrs, axis=0)

        sequences = MicrotubulesFactory.get_degree_2_sequences(g)
        vs_contract = g.vs.indices

        combine_sequences = []
        if n is None:
            for seq in sequences:
                if len(seq) < 2:
                    continue
                else:
                    if all(v in vs_contract for v in seq):
                        v_keep = seq[0]
                        v_remove = seq[1:]
                        for v in v_remove:
                            v_idx = vs_contract.index(v)
                            vs_contract[v_idx] = v_keep
        else:
            for seq in sequences:
                if 2 <= len(seq) < n:
                    if all(v in vs_contract for v in seq):
                        v_keep = seq[0]
                        v_remove = seq[1:]
                        for v in v_remove:
                            v_idx = vs_contract.index(v)
                            vs_contract[v_idx] = v_keep
                        combine_sequences.append(seq)
                else:
                    n_subsequences = int(np.floor(len(seq) // n))

                    for i in range(n_subsequences - 1):
                        subseq = seq[i * n:(i + 1) * n]

                        if all(v in vs_contract for v in subseq):
                            v_keep = subseq[0]
                            v_remove = subseq[1:]
                            for v in v_remove:
                                v_idx = vs_contract.index(v)
                                vs_contract[v_idx] = v_keep
                            combine_sequences.append(subseq)

        n_unique = len(set(vs_contract))
        unique_idxs = list(range(n_unique))

        vs_working = vs_contract.copy()
        vs_contract_new = np.empty(shape=len(vs_contract), dtype=int)

        visited_indices = set()

        for idx in unique_idxs:
            current_min = -1
            for i, v in enumerate(vs_working):
                if v is not None and (current_min == -1 or v < current_min):
                    current_min = v

            min_indices = [i for i, x in enumerate(vs_working) if x == current_min]

            for min_idx in min_indices:
                vs_contract_new[min_idx] = idx
                visited_indices.add(min_idx)
                vs_working[min_idx] = None

        g.contract_vertices(vs_contract_new, combine_attrs={"coordinate": mean_coordinate, "path_length": "min"})
        return g.simplify()

    @staticmethod
    def get_degree_2_sequences(g):
        # Step 1: Identify degree-2 nodes
        degree_2_nodes = [v.index for v in g.vs if g.degree(v.index) == 2]

        # Step 2: Mark visited nodes
        visited = set()
        sequences = []

        # Step 3: Traverse connected sequences of degree-2 nodes
        for node in degree_2_nodes:
            if node not in visited:
                # Start a new sequence
                sequence = []
                current = node
                while current in degree_2_nodes and current not in visited:
                    sequence.append(current)
                    visited.add(current)
                    # Move to the next connected degree-2 neighbor
                    neighbors = [n for n in g.neighbors(current) if n in degree_2_nodes and n not in visited]
                    if neighbors:
                        current = neighbors[0]
                    else:
                        break
                if len(sequence) > 1:  # Only store sequences longer than a single node
                    sequences.append(sequence)
        return sequences

    def _get_skeleton_graph(self):
        """ Skeletonizes the binary image and returns a graph with vertices as pixel coordinates. """
        skeleton = skeletonize(self.img_segmented, method='lee')

        if self._save_skeleton:
            self.save_skeleton_img(skeleton)

        binary_skeleton = skeleton.astype(np.uint8)

        g = ig.Graph()
        pixel_coordinates = np.argwhere(binary_skeleton).astype(float)
        g.add_vertices(pixel_coordinates.shape[0], attributes={'coordinate': pixel_coordinates,
                                                               'type': "Microtubule"})

        distances = np.zeros((g.vcount(), g.vcount()))
        for i in range(g.vcount()):
            distances[i] = np.linalg.norm(pixel_coordinates - pixel_coordinates[i], axis=1)
        edges = np.argwhere(distances < self.vertex_distance_threshold)
        g.add_edges(edges)
        g.simplify()
        self.g = g

    def _refine_graph(self):
        """ Refines the graph via removing edges and vertices based on degree and vertex thresholds. """
        # Find edges between two degree 2 vertices and remove them
        g = self.g.copy()
        edge_list = g.get_edgelist()
        for edge in edge_list:
            v1 = g.vs[edge[0]]
            v2 = g.vs[edge[1]]
            if v1.degree() >= self.vertex_degree_threshold and v2.degree() >= self.vertex_degree_threshold:
                g.delete_edges(edge)

        subgraphs = g.decompose()
        filtered_subgraphs = [subgraph for subgraph in subgraphs if len(subgraph.vs) >= self.vertex_count_threshold]

        g_filtered = ig.Graph()
        for sg in filtered_subgraphs:
            g_filtered = g + sg
        g_filtered.simplify()
        self.g = g_filtered

    def locate_centrosome_with_nucleus(self):
        """ Locates the centrosome via the brightest region in the nuclear region. """
        nucleus_coordinates = self.nucleus.data
        voxel_scale, voxel_units = self.data_loader.get_voxel_scale()
        nucleus_coordinates = np.round(nucleus_coordinates / voxel_scale).astype(int)

        centrosome_intensity = 0
        reference_coordinate = np.zeros(3)

        # nuclear_distance_dims = ([int(2 * self.nuclear_distance_threshold)]*3)
        # centrosome_radius_dims = ([int(2 * self.centrosome_search_radius)]*3)

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

            # for coordinate in nucleus_coordinates:
            #     min_coords = np.maximum(coordinate - self.nuclear_distance_threshold, 0)
            #     max_coords = np.minimum(coordinate + self.nuclear_distance_threshold, self.img.shape)

            # img_slice = self.img[min_coords[0]:max_coords[0], min_coords[1]:max_coords[1], min_coords[2]:max_coords[2]]
            # intensity_sum = uniform_filter(img_slice.astype(float), size=centrosome_radius_dims)

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
        self.g.add_vertex(coordinate=self.centrosome_coordinate, type="Centrosome")

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

    def polarize(self):
        """ Adds polarity to an undirected/depolarized microtubule graph network. """
        subgraphs = self.g.decompose()
        g_centrosome = [sg for sg in subgraphs if any([v['type'] == "Centrosome" for v in sg.vs])][0]
        sg_list = [sg for sg in subgraphs if all([v['type'] != "Centrosome" for v in sg.vs])]

        sg_list_disjoint = []
        for i in tqdm(range(len(sg_list)), desc="Polarizing Microtubules", leave=False):
            sg = sg_list[i]
            vs_closest, vs_distance = self.get_closest_vertex_pair(g_centrosome, sg, None, None)
            if vs_distance is not None and vs_distance < self.vertex_distance_threshold:
                v1, v2 = vs_closest
                v1_coord = v1['coordinate']
                v2_coord = v2['coordinate']
                v1_degree = v1.degree()
                v2_degree = v2.degree()

                # Combine the two graphs
                combined_graph = g_centrosome.disjoint_union(sg)
                v1_new, _ = self.get_closest_vertex(combined_graph, v1_coord, v1_degree)
                v2_new, _ = self.get_closest_vertex(combined_graph, v2_coord, v2_degree)

                # Connect the new vertices
                combined_graph.add_edge(v1_new, v2_new)
                combined_graph = combined_graph.simplify()
                g_centrosome = combined_graph
            else:
                vs_deg1 = [v for v in sg.vs if v.degree() == 1]
                if len(vs_deg1) == 0:
                    continue
                vs_random = np.random.choice(vs_deg1)
                sg_directed = MicrotubulesFactory.redirect_graph(sg, vs_random)
                sg_list_disjoint.append(sg_directed)

        g = self.redirect_graph(g_centrosome, g_centrosome.vs.find(type="Centrosome"))
        for sg in sg_list_disjoint:
            if len(sg.vs) >= self.vertex_count_threshold:
                g_temp = g.disjoint_union(sg.as_directed())
                g = g_temp

        self.g = g.simplify()

    def rescale(self):
        """ Rescales graph coordinate attribute """
        voxel_scale, voxel_units = self.data_loader.get_voxel_scale()
        coordinates = np.array(self.g.vs['coordinate'])
        coordinates *= voxel_scale
        self.g.vs['coordinate'] = coordinates

    # @staticmethod
    # def redirect_graph(g, root_vertex):
    #     """Returns a directed graph with edges directed towards the centrosome."""
    #     g_undirected = g.as_undirected()
    #     g_directed = g.as_directed(mode="mutual")
    #     paths = g_undirected.get_shortest_paths(root_vertex, mode="in", output="vpath")
    #
    #     new_edges = set()
    #     for path in paths:
    #         vs_path = [g_undirected.vs[idx] for idx in path]
    #         for i in range(len(vs_path) - 1):
    #             new_edges.add((vs_path[i].index, vs_path[i + 1].index))
    #
    #     # Remove all edges in g_directed
    #     g_directed.delete_edges(g_directed.get_edgelist())
    #     g_directed.add_edges(list(new_edges))
    #     return g_directed.simplify()

    @staticmethod
    def redirect_graph(g, root_vertex):
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
                new_edges.add((vs_path[i].index, vs_path[i + 1].index))

        # Remove all edges in g_directed
        g_directed.delete_edges(g_directed.get_edgelist())
        g_directed.add_edges(list(new_edges))

        return g_directed.simplify()

    @staticmethod
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

    @staticmethod
    def get_closest_vertex_pair(g1, g2, v1_degree=None, v2_degree=None):
        """Gets the closest pair of vertices between two graphs with the given vertex degrees."""

        if v1_degree is None:
            v1_degree = range(max(v.degree() for v in g1.vs) + 1)

        if v2_degree is None:
            v2_degree = range(max(v.degree() for v in g2.vs) + 1)

        if isinstance(v1_degree, int):
            v1_degree = [v1_degree]
        if isinstance(v2_degree, int):
            v2_degree = [v2_degree]

        vs1 = [v for v in g1.vs if v.degree() in v1_degree]
        vs2 = [v for v in g2.vs if v.degree() in v2_degree]

        vs1_coords = np.array([v['coordinate'] for v in vs1])
        vs2_coords = np.array([v['coordinate'] for v in vs2])

        distances = np.linalg.norm(vs1_coords[:, np.newaxis] - vs2_coords, axis=2)

        if np.all(distances == 0):
            return None, None

        min_idx = np.unravel_index(np.argmin(distances), distances.shape)
        return (vs1[min_idx[0]], vs2[min_idx[1]]), distances[min_idx]

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

    # def _get_random_vectors(self, n=1000, membrane_only_fraction=0.8):
    #     """ Get random vectors in the membrane. """
    #
    #     nucleus_coordinates = self.models["nucleus"].get_coordinates()
    #     membrane_coordinates = self.models["membrane"].get_coordinates()
    #
    #     n_membrane_membrane = int(n * membrane_only_fraction)
    #     n_membrane_nucleus = n - n_membrane_membrane
    #
    #     # Pick random points on the membrane
    #     idx_list = list(range(0, membrane_coordinates.shape[0]))
    #     start_idxs = np.random.choice(idx_list, n_membrane_membrane)
    #     end_idxs = np.random.choice(idx_list, n_membrane_membrane)
    #
    #     start_coordinates = membrane_coordinates[start_idxs]
    #     end_coordinates = membrane_coordinates[end_idxs]
    #
    #     if n_membrane_nucleus > 0:
    #         membrane_start_idxs = np.random.choice(idx_list, n_membrane_nucleus)
    #         membrane_start_coordinates = membrane_coordinates[membrane_start_idxs]
    #
    #         # Finding proximal nucleus coordinate
    #         nucleus_end_coordinates = [
    #             nucleus_coordinates[np.argmin(np.linalg.norm(nucleus_coordinates - point, axis=1))] for point in
    #             membrane_start_coordinates]
    #
    #         start_coordinates = np.vstack([start_coordinates, membrane_start_coordinates])
    #         end_coordinates = np.vstack([end_coordinates, nucleus_end_coordinates])
    #
    #     cytoskeleton_coordinates = (np.array(start_coordinates), np.array(end_coordinates))
    #     return cytoskeleton_coordinates

    def save_skeleton_img(self, img_skeleton):
        """ Save the skeletonized image to the parent microtubule frame directory. """
        img_segmented_path = self.data_loader.pathdict["microtubules_segmented"]
        img_skeleton_path = img_segmented_path.replace("segmented", "skeleton")

        # Reformat the skeleton image
        img = self.format_mask(img_skeleton) * 255

        # Save the skeleton image so true is white and false is black
        io.imsave(img_skeleton_path, img)
