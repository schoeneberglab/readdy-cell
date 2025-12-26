import numpy as np
from igraph import Graph
import igraph as ig

# TODO: Pull in chain generator functions into this
class TopologyGraphGenerator:
    """ Class for generating iGraph representations of topologies for use in ReaDDy simulations. """
    def __init__(self, particle_type, topology_type, spacing=0.3, box_size=np.array([10.]*3), with_flags=True):
        self._ptype = particle_type
        self._ttype = topology_type
        self._spacing = spacing
        self._box_size = box_size
        self._with_flags = with_flags

    def generate_network_erdos_renyi(self, total_num_nodes, nodes_per_iter, mean_degree=1.0):
        actual_num_nodes = 0
        ratios = []
        all_mito = []
        mean_degrees = np.array([])

        for it in range(
                int(total_num_nodes / nodes_per_iter * 2.2)):  # multiply by a factor that accounts for deleted lone nodes
            rand_mito = ig.Graph.Erdos_Renyi(nodes_per_iter, mean_degree / nodes_per_iter)  # use E-R random network
            lone_nodes = [vs.index for vs in rand_mito.vs if vs.degree() == 0]
            rand_mito.delete_vertices(lone_nodes)

            rand_frag_size_list = []
            rand_mito.es['distance'] = 0

            for frag in rand_mito.components():
                contracted_frag = self.contract_edges(rand_mito.induced_subgraph(frag))  # remove degree-2 nodes
                rand_frag_size_list.append(len(contracted_frag.vs))
                mean_degrees = np.append(mean_degrees, [v.degree() for v in contracted_frag.vs])

            total_frag_size = sum(rand_frag_size_list)
            if total_frag_size > 0:
                actual_num_nodes += total_frag_size  # count the number of nodes generated
                ratios.append(nodes_per_iter / total_frag_size)
                all_mito.append(rand_mito)  # add the nodes and continue until target_num_nodes is reached
            else:
                # Handle the case when no fragments are left
                print(f"Iteration {it}: All fragments were removed. Skipping this iteration.")

        gs = self.assign_coordinates_to_graphs(all_mito)
        return gs

    @staticmethod
    def contract_edges(frag):
        """ Method which contracts all nodes with degree 2 into a single edge"""
        skeleton_nodes = []
        edge_weights = []
        all_filaments = []

        root = 0
        for node in frag.vs:
            if node.degree() == 1:
                root = node
                break

        last_node = -1
        for node in frag.dfsiter(root):
            n = node.index
            degree = node.degree()

            if n != root:
                if degree == 2:
                    # first node on a new branch after concluding a branch
                    if last_node == -1:
                        skeleton_nodes.append(n)
                        last_node = n

                    # sometimes in large graph a new branch is visited without reaching a degree!=2 node
                    else:
                        # this may fail when reach end of one branch and jump to the start of another branch
                        try:
                            weight = frag.es[frag.get_eid(n, last_node)]['distance']

                        # when it fails just start another skeleton
                        except Exception:
                            if len(skeleton_nodes) != 0:
                                all_filaments.append([skeleton_nodes, sum(edge_weights)])
                                skeleton_nodes = []
                                edge_weights = []

                                skeleton_nodes.append(n)
                                last_node = n

                        # when the two nodes are on skeleton we can just append distance and node
                        else:
                            edge_weights.append(weight)
                            skeleton_nodes.append(n)
                            last_node = n

                # conclude the branch when reached a terminal or branching point
                else:
                    if len(skeleton_nodes) != 0:
                        all_filaments.append([skeleton_nodes, sum(edge_weights)])
                        skeleton_nodes = []
                        edge_weights = []
                        last_node = -1

        # add edges and delete nodes
        edge_nodes = []
        for f in all_filaments:
            nodes = f[0]
            weight = f[1]

            if len(nodes) == 1:
                ends = frag.neighbors(nodes[0])
            else:
                neighs_a = frag.neighbors(nodes[0])
                neighs_b = frag.neighbors(nodes[-1])
                end_a = [n for n in neighs_a if n not in nodes]
                end_b = [n for n in neighs_b if n not in nodes]
                ends = end_a + end_b

            if len(ends) != 2:
                raise Exception('Invalid pairs to connect')
            else:
                # add edges that connect braching nodes to skeleton nodes
                weight += frag.es[frag.get_eid(nodes[0], ends[0])]['distance']
                weight += frag.es[frag.get_eid(nodes[-1], ends[1])]['distance']
                frag.add_edge(ends[0], ends[1], distance=weight)
                edge_nodes = edge_nodes + nodes
        frag.delete_vertices(edge_nodes)
        frag.simplify(combine_edges='sum')
        return frag

    def assign_coordinates_to_graphs(self, graph_list):
        """
        Assigns coordinates to the nodes of each graph such that the length of each edge equals the spacing parameter.

        Parameters:
        - graph_list (list): List of graphs (igraph.Graph) to assign coordinates to.

        Returns:
        - list: List of graphs with coordinates assigned to each node.
        """

        modified_graphs = []

        for graph in graph_list:
            num_nodes = graph.vcount()

            # Compute the layout using the graphopt layout algorithm
            layout = graph.layout_graphopt(dim=3, edge_length=self._spacing, niter=1000)

            # Extract positions
            positions = np.array(layout.coords)

            # Assign the positions to the nodes
            for i, v in enumerate(graph.vs):
                v["coordinate"] = positions[i]
                v["ptype"] = self._ptype
                v["ttype"] = self._ttype

            if self._with_flags:
                graph = self._assign_flags(graph)
            modified_graphs.append(graph)
        return modified_graphs

    def assign_coordinates_to_graphs(self, graph_list):
        """
        Assigns random coordinates to the nodes of each graph and returns the modified graphs.

        Parameters:
        - graph_list (list): List of graphs (igraph.Graph) to assign coordinates to.
        - particle_radius (float): Radius of each particle.
        - spacing (float): Desired spacing factor for node placement.
        - box_dimensions (tuple): Dimensions of the box as (x_size, y_size, z_size).

        Returns:
        - list: List of graphs with coordinates assigned to each node.
        """

        modified_graphs = []

        for graph in graph_list:
            num_nodes = graph.vcount()
            positions = np.zeros((num_nodes, 3))

            # Assign random coordinates to each node
            for node_id in range(num_nodes):
                range_factor = 2 * np.random.uniform(size=(1, 3)) - 1
                positions[node_id] = 1.5 * self._spacing * self._box_size * range_factor

            # Assign the coordinates as a node attribute
            for i, v in enumerate(graph.vs):
                v["coordinate"] = positions[i]
                v["ptype"] = self._ptype
                v["ttype"] = self._ttype
            if self._with_flags:
                graph = self._assign_flags(graph)
            modified_graphs.append(graph)
        return modified_graphs

    def _get_segment_graph(self, n_nodes, coordinates):
        """ Constructs a segment graph with the specified number of nodes and coordinates. """
        g = ig.Graph(directed=False)
        ptypes = [self._ptype] * n_nodes
        ttypes = [self._ttype] * n_nodes
        g.add_vertices(n_nodes)
        g.vs["ptype"] = ptypes
        g.vs["ttype"] = ttypes
        g.vs["coordinate"] = coordinates

        for i in range(n_nodes - 1):
            g.add_edge(i, i + 1)

        if self._with_flags:
            g = self._assign_flags(g)
        return g

    def _assign_flags(self, g):
        """ Assigns flags to the nodes of the graph. """
        for v in g.vs:
            if v.degree() == 1:
                v["ptype"] += "#terminal"
            else:
                v["ptype"] += "#internal"
        return g

    def get_random_segment(self, n_nodes, spacing,  centroid=np.array([0., 0., 0.]), particle_type="mitochondria", topology_type="Mitochondria", with_flags=True):
        """ Constructs a segment graph with randomly placed nodes. """
        coords = [np.zeros(3)]
        for i in range(1, n_nodes):
            displacement = np.random.normal(size=(3))
            displacement *= spacing / np.linalg.norm(displacement)
            coords.append(coords[i - 1] + displacement)
        coords = np.array(coords)
        coordinates = coords - np.mean(coords, axis=0) + centroid
        g = self._get_segment_graph(n_nodes, coordinates)
        return g

    def get_linear_segment(self, n_nodes, spacing, centroid=np.array([0., 0., 0.]), particle_type="mitochondria", topology_type="Mitochondria", with_flags=True):
        """ Constructs a linear segment graph. """
        coords = np.zeros(shape=(n_nodes, 3))
        for i in range(1, n_nodes):
            coords[i] = coords[i - 1] + np.array([spacing, 0., 0.])
        coords -= np.mean(coords, axis=0)
        coordinates = coords + centroid
        g = self._get_segment_graph(n_nodes, coordinates)
        return g


class ToyGraphGenerator:
    def __init__(self):
        pass

    @staticmethod
    def construct_circular_directed_graph(node_spacing: float,
                                          r: float) -> Graph:
        """Constructs a circular directed graph with a specified node_spacing.
        :param node_spacing: The spacing between nodes.
        :param r: The radius of the circle.
        :return: An igraph Graph object.
        """
        circumference = 2 * np.pi * r
        n_vertices = int(circumference / node_spacing)  # Determine number of vertices based on spacing
        graph = ig.Graph(directed=True)
        graph.add_vertices(n_vertices)
        angles = np.linspace(0, 2 * np.pi, n_vertices, endpoint=False)
        coordinates = [(r * np.cos(angle), r * np.sin(angle), 0.) for angle in angles]
        for i, coord in enumerate(coordinates):
            graph.vs[i]["coordinate"] = coord

        edges = [(i, (i + 1) % n_vertices) for i in range(n_vertices)]
        graph.add_edges(edges)

        return graph.simplify()

    @staticmethod
    def construct_figure8_directed_graph(node_spacing: float,
                                         r: float) -> Graph:
        """
        Constructs a figure-8 shaped directed graph.
        :param node_spacing: The spacing between nodes.
        :param r: The radius of the figure-8 shape from intersection to the farthest point.
        :return: An igraph Graph object.
        """
        # Approximate the total length of the figure-8 shape
        approximate_length = 4 * r  # Rough estimate of the path length of the figure-8
        n_vertices = int(approximate_length / node_spacing)  # Determine number of vertices based on spacing
        graph = ig.Graph(directed=True)
        graph.add_vertices(n_vertices)

        t = np.linspace(0, 2 * np.pi, n_vertices, endpoint=False)
        coordinates = [(r * np.sin(ti) / (1 + np.cos(ti)**2),
                        r * np.sin(ti) * np.cos(ti) / (1 + np.cos(ti)**2), 0.)
                       for ti in t]

        for i, coord in enumerate(coordinates):
            graph.vs[i]["coordinate"] = coord

        edges = [(i, (i + 1) % n_vertices) for i in range(n_vertices)]
        graph.add_edges(edges)

        return graph.simplify()

    @staticmethod
    def construct_line_directed_graph(node_spacing: float,
                                      r: float) -> Graph:
        """
        Constructs a linear directed graph with a specified node_spacing.
        :param node_spacing: The spacing between nodes.
        :param r: The length of the line.
        :return: An igraph Graph object.
        """
        total_length = 2 * r  # The length of the line (from -r to +r)
        n_vertices = int(total_length / node_spacing) + 1  # Number of vertices based on node_spacing
        graph = ig.Graph(directed=True)
        graph.add_vertices(n_vertices)

        x = np.linspace(-r, r, n_vertices)
        coordinates = [(xi, 0., 0.) for xi in x]

        for i, coord in enumerate(coordinates):
            graph.vs[i]["coordinate"] = coord

        edges = [(i, i + 1) for i in range(n_vertices - 1)]
        graph.add_edges(edges)

        return graph.simplify()

    @staticmethod
    def construct_spiral_directed_graph(node_spacing: float,
                                        r_start: float,
                                        r_end: float,
                                        n_turns: int = 5) -> Graph:
        """
        Constructs a spiral directed graph with a specified node_spacing.
        :param node_spacing: The spacing between nodes.
        :param r_start: The starting radius of the spiral.
        :param r_end: The ending radius of the spiral.
        :param n_turns: The number of turns in the spiral.
        """
        spiral_length = n_turns * 2 * np.pi * (r_start + r_end) / 2  # Approximate the length of the spiral
        n_vertices = int(spiral_length / node_spacing) + 1  # Determine number of vertices based on spacing
        graph = ig.Graph(directed=True)
        graph.add_vertices(n_vertices)
        radii = np.linspace(r_start, r_end, n_vertices)
        angles = np.linspace(0, 2 * np.pi * n_turns, n_vertices, endpoint=False)
        coordinates = [(r * np.cos(angle), r * np.sin(angle), 0.) for r, angle in zip(radii, angles)]
        for i, coord in enumerate(coordinates):
            graph.vs[i]["coordinate"] = coord
        edges = [(i, i + 1) for i in range(n_vertices - 1)]
        graph.add_edges(edges)
        return graph.simplify()

    @staticmethod
    def construct_whirlpool_graph(node_spacing, r_start, r_end, z_extent, n_turns=5, closed=True):
        """Constructs a whirlpool-like graph with a specified node_spacing and optional closed structure."""
        spiral_length = n_turns * 2 * np.pi * (r_start + r_end) / 2  # Approximate the length of the spiral
        vertical_length = z_extent  # The vertical length is the height of the whirlpool
        total_length = spiral_length + vertical_length  # Total length is a combination of horizontal and vertical distance
        n_vertices = int(total_length / node_spacing)  # Determine number of vertices based on spacing
        if closed:
            n_vertices += 1  # Account for the closing edge

        graph = ig.Graph(directed=True)
        graph.add_vertices(n_vertices)

        radii = np.linspace(r_start, r_end, n_vertices)
        z_values = np.linspace(+z_extent / 2, -z_extent / 2, n_vertices)
        angles = np.linspace(0, 2 * np.pi * n_turns, n_vertices, endpoint=False)

        coordinates = [(r * np.cos(angle), r * np.sin(angle), z) for r, angle, z in zip(radii, angles, z_values)]
        graph.vs["coordinate"] = coordinates

        edges = [(i, i + 1) for i in range(n_vertices - 1)]
        if closed:
            edges.append((n_vertices - 1, 0))

        graph.add_edges(edges)

        return graph.simplify()

    @staticmethod
    def construct_random_directed_graph(n_segments: int,
                                        node_spacing: float,
                                        box_size: np.ndarray,) -> Graph:
        """
        Constructs random directed graph segments with a specified node_spacing and box_size.
        :param n_segments: Number of segments to generate.
        :param node_spacing: Spacing between nodes.
        :param box_size: Size of the box in which the segments are generated.
        :return: A random directed graph.
        """

        graph = ig.Graph(directed=True)
        for i in range(n_segments):
            g = ig.Graph(directed=True)
            start_coordinate = np.random.uniform(-box_size / 2, box_size / 2, 3)
            end_coordinate = np.random.uniform(-box_size / 2, box_size / 2, 3)
            n_nodes = int(np.linalg.norm(start_coordinate - end_coordinate) / node_spacing)
            coordinates = np.linspace(start_coordinate, end_coordinate, n_nodes)
            g.add_vertices(n_nodes)
            for j, coord in enumerate(coordinates):
                g.vs[j]["coordinate"] = coord
            edges = [(k, k + 1) for k in range(n_nodes - 1)]
            g.add_edges(edges)
            graph += g
        return graph.simplify()

if __name__ == "__main__":
    tgg = TopologyGraphGenerator("mitochondria", "Mitochondria")
    gs = tgg.generate_network_erdos_renyi(600, 50)
    count = 0
    edge_lengths = []
    for g in gs:
        count += g.vcount()
        edges = g.get_edgelist()

        for edge in edges:
            v1_coord = np.array(g.vs[edge[0]]["coordinate"])
            v2_coord = np.array(g.vs[edge[1]]["coordinate"])
            edge_length = np.linalg.norm(v2_coord - v1_coord)
            edge_lengths.append(edge_length)
    print(np.mean(edge_lengths))
    print(count)
