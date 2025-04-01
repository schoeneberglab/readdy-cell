import open3d as o3d
import readdy
from scipy.ndimage import binary_dilation
from skimage.feature import canny
import skimage as ski
import numpy as np
import igraph as ig
from typing import Optional, List

class ModelUtils:

    @staticmethod
    def translate(coordinates, translation_vector):
        """ Translate a model's coordinates according to a translation vector. """
        return coordinates + translation_vector

    @staticmethod
    def rotate(coordinates, rotation_matrix):
        """ Rotate a model's coordinates according to a rotation matrix. """
        return np.dot(coordinates, rotation_matrix)

    @staticmethod
    def rescale(coordinates, scale_factor):
        """ Scale a model's coordinates according to a scale factor. """
        centroid = ModelUtils.calculate_centroid(coordinates)
        coordinates = ModelUtils.translate(coordinates, -centroid)
        coordinates = coordinates * scale_factor
        coordinates = ModelUtils.translate(coordinates, centroid)
        return coordinates

    @staticmethod
    def calculate_centroid(coordinates):
        """ Calculate the centroid of a model. """
        return np.mean(coordinates, axis=0)


    def center_model(self, vector):
        """ Centers the model. """
        if isinstance(self.data, np.ndarray):
            self.data += vector
            print(len(self.data))
        else:
            if not isinstance(self.data, list):
                self.data = [self.data]

            centered_graphs = []
            for g in self.data:
                coordinates = np.array(g.vs['coordinate'])
                coordinates += vector
                for i, vs in enumerate(g.vs):
                    vs['coordinate'] = coordinates[i]
                centered_graphs.append(g)
            self.data = centered_graphs


    @staticmethod
    def get_max_node_degree(graphs):
        # Find the max degree of all nodes in all graphs (list)
        max_degrees = [max(g.degree()) for g in graphs]
        return max(max_degrees)

    @staticmethod
    def get_graph_coords(graphs):
        if isinstance(graphs, ig.Graph):
            graphs = [graphs]
        coords = np.empty((0, 3))
        for g in graphs:
            coords = np.vstack((coords, np.array(g.vs['coordinate'])))
        return coords

    @staticmethod
    def update_graph_coords(graphs, new_coords):
        old_coords = np.empty((0, 3))
        for g in graphs:
            old_coords = np.vstack((old_coords, np.array(g.vs['coordinate'])))
        new_coords = np.array(new_coords)

        i = 0
        for g in graphs:
            for j in range(len(g.vs)):
                g.vs[j]['coordinate'] = new_coords[i]
                i += 1
        return graphs

    @staticmethod
    def format_mask(img):
        """ Reformats the membrane mask and returns it. """
        if img.ndim != 3:
            raise ValueError("Input mask data must be 3D.")
        img = np.flip(img, axis=1)
        img = np.swapaxes(img, 0, 2)
        img_array = np.array(img > 0, dtype=np.uint8)
        return img_array

    @staticmethod
    def translate_mask(mask, translation_vector):
        """ Shift all pixel values according to the translation vector. """
        mask = np.roll(mask, translation_vector, axis=(0, 1, 2))
        return mask

    @staticmethod
    def _canny_3d(data):
        """An algorithm which performs edge detection in 3D using the canny algorithm."""
        # Copy data and permute axes to rotate
        data_arrays = ModelUtils.permute_array(data)
        canny_arrays = []
        for array in data_arrays:
            edges = np.zeros(array.shape)
            for i in range(0, array.shape[0]):
                edges[i, :, :] = canny(array[i, :, :],
                                       sigma=1,
                                       low_threshold=0.1,
                                       high_threshold=0.2).astype(np.uint8)
            canny_arrays.append(edges)

        # Recombine the edge arrays to a single arrays and convert to N,3 array of coordinates
        edge_data = ModelUtils.combine_permuted_arrays(canny_arrays)
        return edge_data

    @staticmethod
    def _pad_mask_3d(data, padding_array):
        """Pad a 3D mask with zeros."""
        permuted_arrays = ModelUtils.permute_array(data)
        for i in range(len(permuted_arrays)):
            permuted_arrays[i] = np.pad(permuted_arrays[i], padding_array[i], mode='constant', constant_values=0)
        padded_data = ModelUtils.combine_permuted_arrays(permuted_arrays)
        return padded_data

    @staticmethod
    def _dilate_mask_3d(data, dilation_array):
        """Dilate a 3D mask."""
        permuted_arrays = ModelUtils.permute_array(data)
        for axis, permuted_array in enumerate(permuted_arrays):
            iterations = dilation_array[axis]
            structure = np.ones((2 * iterations + 1,) * 2)
            for slice_idx in range(permuted_array.shape[0]):
                permuted_array[slice_idx] = binary_dilation(permuted_array[slice_idx], structure=structure)

        dilated_data = ModelUtils.combine_permuted_arrays(permuted_arrays)
        return dilated_data

    @staticmethod
    def _calculate_downsample_factor(coordinates, radius, buffer_factor):
        """ Calculates the downsample factor for a given radius that is required to maintain a closed surface. """
        current_particle_count = coordinates.shape[0]
        circle_area = np.pi * radius ** 2
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coordinates)
        pcd.estimate_normals()
        hull, _ = pcd.compute_convex_hull()
        surface_area = hull.get_surface_area()
        downsample_factor = int(current_particle_count / ((surface_area / circle_area) * buffer_factor))
        return downsample_factor

    @staticmethod
    def _calculate_dilation_array(radius, voxel_scale):
        """ Calculates dilation array for a given radius and voxel scale. """
        return np.array([int(radius / voxel_scale[0]), int(radius / voxel_scale[1]), int(radius / voxel_scale[2])])

    @staticmethod
    def _downsample_coordinates(coordinates, downsample_factor):
        """Downsample the coordinates of a membrane using Open3D's farthest point sampling."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coordinates)
        ds_pcd = pcd.farthest_point_down_sample(int(coordinates.shape[0] / downsample_factor))
        ds_pcd.estimate_normals()
        return np.array(ds_pcd.points)

    @staticmethod
    def permute_array(data):
        return [np.copy(data), np.rot90(data, axes=(0, 1)), np.rot90(data, axes=(0, 2))]

    @staticmethod
    def combine_permuted_arrays(data):
        data[1] = np.rot90(data[1], axes=(1, 0))
        data[2] = np.rot90(data[2], axes=(2, 0))
        return np.logical_or.reduce(data)

    @staticmethod
    def coordinates_to_pcd(coordinates):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coordinates)
        pcd.estimate_normals()
        return pcd

    @staticmethod
    def pcd_to_coordinates(pcd):
        return np.asarray(pcd.points)

    @staticmethod
    def get_downsample_factor(coordinates, radius, buffer_factor):
        current_particle_count = coordinates.shape[0]
        circle_area = np.pi * radius ** 2
        pcd = ModelUtils.coordinates_to_pcd(coordinates)
        hull, _ = pcd.compute_convex_hull()
        surface_area = hull.get_surface_area()
        downsample_factor = int(current_particle_count / ((surface_area / circle_area) * buffer_factor))
        return downsample_factor

    @staticmethod
    def add_to_simulation(model,
                          simulation: readdy.Simulation,
                          *args, **kwargs):
        """ Adds a model to a simulation. """

        assert model is not None, "Please provide a Model object."
        assert simulation is not None, "Please provide a readdy.Simulation object."

        topology_type = kwargs.get("topology_type", model.topology_type)
        particle_type = kwargs.get("particle_type", model.particle_type)

        if isinstance(model.data, ig.Graph):
            model.data = [model.data]
        if topology_type is not None:
            ModelUtils._add_topology_species_to_simulation(model.data, simulation, topology_type, particle_type)
        else:
            ModelUtils._add_particle_species_to_simulation(model.data, simulation, particle_type)

    @staticmethod
    def _add_topology_species_to_simulation(graphs: List[ig.Graph],
                                            simulation: readdy.Simulation,
                                            topology_type: str,
                                            particle_type: str):
        """ Adds the topology/topologies to the simulation. """
        for g in graphs:
            g_coords = np.empty((0, 3))
            for vs in g.vs:
                vs_coords = np.array(vs["coordinate"])
                g_coords = np.vstack((g_coords, vs_coords)).reshape(-1, 3)

            particle_types = [particle_type] * len(g_coords)

            top = simulation.add_topology(topology_type=topology_type,
                                          particle_types=particle_types,
                                          positions=g_coords)
            for es in g.es:
                top.get_graph().add_edge(es.source, es.target)

    @staticmethod
    def _add_particle_species_to_simulation(coordinates: np.ndarray,
                                            simulation: readdy.Simulation,
                                            particle_type: str):
        """ Adds the particle species to the simulation. """
        simulation.add_particles(particle_type, coordinates)

    @staticmethod
    def save_mask_as_binary_tif(data):
        """ Save a mask as a binary tif file. """
        ski.io.imsave('mask.tif', data.astype(np.uint16))


