import re
import itertools
import numpy as np


class ReactionUtils:
    @staticmethod
    def get_reaction_topology_products(reactive_tags, tag_to_topology):
        unique_tags = ReactionUtils._get_unique_tags(reactive_tags)
        tag_combinations = ReactionUtils._get_tag_combinations(unique_tags)
        tag_permutations = ReactionUtils._get_tag_permutations(tag_combinations)
        valid_combinations = ReactionUtils._get_valid_tag_combinations(tag_permutations, reactive_tags)
        valid_topology_combinations = ReactionUtils._get_valid_topology_combinations(valid_combinations,
                                                                                     tag_to_topology)
        topology_products = []
        for topology_combination in valid_topology_combinations:
            topology_products.append('-'.join(topology_combination))
        return topology_products

    @staticmethod
    def _get_unique_tags(reactive_tags):
        unique_tags = set()
        for reactive_pair in reactive_tags:
            for tag in reactive_pair:
                unique_tags.add(tag)
        return list(unique_tags)

    @staticmethod
    def _get_tag_combinations(unique_tags):
        # get all possible combinations of unique tags possible
        tag_combinations = []
        for i in range(1, len(unique_tags) + 1):
            tag_combinations.extend(itertools.combinations(unique_tags, i))
        tag_combinations = [list(comb) for comb in tag_combinations]
        return tag_combinations

    @staticmethod
    def _get_tag_permutations(tag_combinations):
        # Get all possible permutations of the tag combinations
        tag_permutations = []
        for comb in tag_combinations:
            tag_permutations.extend(itertools.permutations(comb))
        tag_permutations = [list(perm) for perm in tag_permutations]
        return tag_permutations

    @staticmethod
    def _get_valid_tag_combinations(tag_permutations, reactive_tags):
        candidate_combinations = []
        for perm in tag_permutations:
            if len(perm) > 1:
                candidate_combinations.append(perm)
        check_pairs = [tuple(pair) for pair in reactive_tags]

        valid_combinations = []
        for candidate in candidate_combinations:
            valid = True
            for i in range(len(candidate) - 1):
                if tuple([candidate[i], candidate[i + 1]]) not in check_pairs:
                    valid = False
            if valid:
                valid_combinations.append(candidate)
        return valid_combinations

    @staticmethod
    def _get_valid_topology_combinations(valid_combinations, tag_to_topology):
        topology_combinations = []
        for combination in valid_combinations:
            topology_combination = [tag_to_topology[tag] for tag in combination]
            topology_combinations.extend(list(itertools.product(*topology_combination)))

        return topology_combinations

    # Active Transport Utility Functions
    @staticmethod
    def get_cytoskeleton_distances(point):
        """
        Find the closest coordinates on the line segments to the given point and the distances.

        Parameters:
        point (array-like): The point [x, y, z] as a numpy array or list.

        Returns:
        tuple: A tuple containing:
            - numpy.ndarray: The closest coordinates on each line segment to the given point.
            - numpy.ndarray: The distances from the point to each line segment.
        """

        point = np.array(point)
        segments_start, segments_end = parameters["CytoskeletonCoordinates"]
        segment_vectors = segments_end - segments_start
        point_vectors = point - segments_start
        segment_lengths_squared = np.sum(segment_vectors ** 2, axis=1)
        projections = np.einsum('ij,ij->i', point_vectors, segment_vectors) / segment_lengths_squared
        clamped_projections = np.clip(projections, 0, 1)
        proximal_coordinates = segments_start + (clamped_projections[:, np.newaxis] * segment_vectors)
        segment_distances = np.linalg.norm(proximal_coordinates - point, axis=1)
        return segment_distances, proximal_coordinates

    @staticmethod
    def is_point_between(point, vector_id):
        """
        Determine if a test point lies between two coordinates on the same line segment.

        Parameters:
        point (array-like): The test point [x, y, z] as a numpy array or list.
        segment_start (array-like): The start point of the segment [x1, y1, z1].
        segment_end (array-like): The end point of the segment [x2, y2, z2].

        Returns:
        bool: True if the test point lies between the segment start and end, False otherwise.
        """
        # Convert inputs to numpy arrays for easy manipulation
        point = np.array(point)
        segment_start, segment_end = parameters["Cytoskeleton"][str(vector_id)]["coordinates"]
        if parameters["Common"]["use_cytoskeleton_end_check"]:
            tolerance = parameters["Common"]["cytoskeleton_end_check_tolerance"]
            # Shorten the segment by the tolerance
            segment_start = segment_start + (
                    tolerance * parameters["Cytoskeleton"][str(vector_id)]["positive_step_unit_vector"])
            segment_end = segment_end - (
                    tolerance * parameters["Cytoskeleton"][str(vector_id)]["positive_step_unit_vector"])

        return np.all(np.logical_and(segment_start <= point, point <= segment_end))

    @staticmethod
    def get_gaussian_distribution(mean, variance, size):
        """ Returns an array of gaussian-distributed values according to mean and variance """
        return np.random.normal(mean, np.sqrt(variance), size)

    @staticmethod
    def get_poisson_distribution(mean, value_range, size=1000):
        """ Returns an array of n poisson-distributed values according a mean and range """
        values = np.random.poisson(mean, size)
        return values[(value_range[0] <= values) & (values <= value_range[1])]

