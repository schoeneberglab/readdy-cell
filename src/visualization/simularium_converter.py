import json
import readdy
import igraph as ig
from typing import Optional
import numpy as np
# from simulariumio.readdy import ReaddyConverter, ReaddyData
from simulariumio.readdy import ReaddyData
from simulariumio import (
    UnitData,
    MetaData,
    DisplayData,
    DISPLAY_TYPE,
    ModelMetaData,  # TODO: Add metadata to model classes?
    TrajectoryConverter,
    CameraData
)

from src.visualization.agent_data_custom import AgentData
from src.visualization.trajectory_data_custom import TrajectoryData
from src.visualization.readdy_converter_custom import ReaddyConverter
from src.analysis import TopologyGraphs

# TODO: Find bug which causes incorrect fiber position array dims when sequentially converting a set of trajectories
class SimulariumConverter:
    DEFAULT_PROPERTIES = {
        "mitochondria#internal": {"radius": 0.15,
                                  "color": "green"},
        "mitochondria#terminal#AT": {"radius": 0.15,
                                     "color": "green"},
        "mitochondria#terminal": {"radius": 0.15,
                                  "color": "green"},
        "mitochondria#AT": {"radius": 0.15,
                            "color": "green"},
        "mitochondria#AT-IM1": {"radius": 0.15,
                                "color": "green"},
        "mitochondria#AT-IM2": {"radius": 0.15,
                                "color": "green"},
        # "mitochondria#AT1-IM1": {"radius": 0.15,
        #                          "color": "green"},
        # "mitochondria#AT1-IM2": {"radius": 0.15,
        #                          "color": "green"},
        # "mitochondria#AT2-IM1": {"radius": 0.15,
        #                          "color": "green"},
        # "mitochondria#AT2-IM2": {"radius": 0.15,
        #                          "color": "green"},
        "mitochondria#Fusion-IM1": {"radius": 0.15,
                                    "color": "green"},
        "mitochondria#Fusion-IM2": {"radius": 0.15,
                                    "color": "green"},
        "mitochondria#Fission-IM1": {"radius": 0.15,
                                     "color": "green"},
        "mitochondria#Fission-IM2": {"radius": 0.15,
                                     "color": "green"},
        "nucleus": {"radius": 0.25,
                    "color": "blue"},
        "membrane": {"radius": 0.25,
                     "color": "red"},
        "microtubule": {"color": "magenta",
                        "radius": 0.03},
        # "motor#kinesin": {"radius": 0.05,
        #                   "color": "white"},
        # "motor#dynein": {"radius": 0.05,
        #                  "color": "gray"},
        # "motor#decay": {"radius": 0.05,
        #                 "color": "black"},
        # "motor#IM": {"radius": 0.05,
        #              "color": "black"},
        "motor#kinesin": {"radius": 0.05,
                          "color": "orange"},
        "motor#dynein": {"radius": 0.05,
                         "color": "yellow"},
        "motor#decay": {"radius": 0.05,
                        "color": "black"},
        "motor#IM": {"radius": 0.05,
                     "color": "black"},
    }
    # COLORS_PATH = "/Users/earkfeld/PycharmProjects/mitosim/src/visualization/colors.json"
    COLORS_PATH = "/home/earkfeld/PycharmProjects/mitosim/colors.json"

    def __init__(self, readdy_h5_path: str, **kwargs):

        self.readdy_h5_path = readdy_h5_path
        self.timestep = kwargs.get("timestep", 5.e-3)
        self.fiber_graph = kwargs.get("fiber_graph", None)
        self.fiber_coordinates = kwargs.get("fiber_coordinates", None)
        self.strided_fiber_coordinates = kwargs.get("strided_fiber_coordinates", False)
        self._ignore_types = kwargs.get("ignore_types", ["membrane"])

        self.time_units = UnitData(kwargs.get("time_units", "s"))
        self.spatial_units = UnitData(kwargs.get("spatial_units", "um"))
        self.fov_degrees = kwargs.get("fov_degrees", 75)
        self.system_properties = kwargs.get("system_properties", self.DEFAULT_PROPERTIES)

        # self.colors_path = "/Users/earkfeld/PycharmProjects/mitosim/src/visualization/colors.json"
        self.color_map = self.load_color_mappings()
        self.trajectory = readdy.Trajectory(readdy_h5_path)

        # Derived attributes
        self.box_size = self.trajectory.box_size

        print(self.box_size)

        # Observables
        self.time, self.particle_types, self.particle_ids, self.positions = self.trajectory.read_observable_particles()
        # self.time, self.positions = self.trajectory.read_observable_particle_positions()  # Replaces particle observable
        _, self.topologies = self.trajectory.read_observable_topologies()
        self.stride = self.time[1] - self.time[0]
        self.timestep *= self.stride  # Updating timestep to account for time per trajectory stride

        # Derived attributes
        self.box_size = self.trajectory.box_size
        self.particle_id_to_type_map = {value: key for key, value in self.trajectory.particle_types.items()}
        self.n_steps = len(self.time)

        self._with_fibers = kwargs.get("with_fibers", True)
        self.fiber_type = kwargs.get("fiber_type", "tubulin")
        if self.fiber_type in self.trajectory.particle_types.keys() and self._with_fibers:
            print("Extracting fiber data...")
            self._extract_fiber_data()

        self.max_n_subpoints = None
        self.subpoint_array = None
        self.particle_agent_data = None
        self.fiber_agent_data = None

    def _extract_fiber_data(self):
        """ Extracts fiber data from the trajectory """
        tg = TopologyGraphs(self.readdy_h5_path, timestep=self.timestep)
        tg.run(self.fiber_type)

        # gs = tg.results["all_topology_graphs"][0]
        # g_mt = ig.Graph()
        # gs_mt = []
        # for g in gs:
        #     ptypes = [v["type"] for v in g.vs]
        #     if "tubulin" not in ptypes:
        #         continue
        #     else:
        #         g_new = g.copy()
        #         vs_remove = []
        #         for v in g_new.vs:
        #             if "tubulin" not in v["type"]:
        #                 vs_remove.append(v)
        #         g_new.delete_vertices(vs_remove)
        #         g_mt += g_new.simplify()
        #         gs_mt.append(g_new.simplify())

        # Testing
        gs = tg.results[self.fiber_type][0]
        g_mt = ig.Graph()
        gs_mt = []
        for i, g in enumerate(gs):
            ptypes = [v["type"] for v in g.vs]
            if "tubulin" not in ptypes:
                continue
            else:
                g_new = g.copy()
                vs_remove = []
                for v in g_new.vs:
                    if "tubulin" not in v["type"]:
                        vs_remove.append(v)
                g_new.delete_vertices(vs_remove)
                g_mt += g_new.simplify()
                gs_mt.append(g_new.simplify())

        puids = []
        for i, g in enumerate(gs_mt):
            puids.extend(g.vs["uid"])

        times, types, ids, positions = self.trajectory.read_observable_particles()

        coordinates = np.zeros(shape=(len(times), len(puids), 3))
        for t, frame_pos in enumerate(positions):
            for i, pos in enumerate(frame_pos):
                if "tubulin" in self.trajectory._inverse_types_map[types[t][i]]:
                    puid = ids[t][i]
                    coordinates[t, i] = pos
                    # coordinates[t, puid] = pos

        self.fiber_graph = g_mt
        self.fiber_coordinates = coordinates
        self.strided_fiber_coordinates = True

    def load_color_mappings(self):
        with open(self.COLORS_PATH, 'r') as file:
            colors = json.load(file)
        return {color["name"].lower(): color["hex"] for color in colors}

    def get_all_topology_types(self):
        return [str(key).split("\'")[1] for key, value in self.trajectory.topology_types.items()]

    def get_all_particle_types(self):
        return [str(key).split("\'")[1] for key, value in self.trajectory.particle_types.items()]

    def construct_particle_display_data(self):
        display_data = {}
        for particle_type in self.system_properties.keys():
            display_data[particle_type] = DisplayData(
                name=particle_type,
                color=self.color_map[self.system_properties[particle_type]["color"]],
                radius=self.system_properties[particle_type]["radius"],
                display_type=DISPLAY_TYPE.SPHERE,
            )
        return display_data

    def create_readdy_data(self):
        display_data = self.construct_particle_display_data()
        readdy_data = ReaddyData(
            meta_data=MetaData(
                box_size=self.box_size,
                trajectory_title="",
                model_meta_data=None
            ),
            timestep=self.timestep,
            path_to_readdy_h5=self.readdy_h5_path,
            display_data=display_data,
            ignore_types=["tubulin"],
            time_units=self.time_units,
            spatial_units=self.spatial_units
        )
        return readdy_data

    def get_fiber_agent_data(self):
        if self.fiber_coordinates is not None:
            print("Getting Agent Data for Dynamic Fibers...")
            self.fiber_agent_data = self._get_dynamic_fiber_agent_data()
        else:
            self.fiber_agent_data = self._get_static_fiber_agent_data()

    def generate_fiber_graph_trajectory(self):
        """ Generates a graph for each frame in the trajectory """
        gs = []
        for t in range(self.n_steps):
            # Copy the original graph and update the coordinates
            g = self.fiber_graph.copy()
            for i, v in enumerate(g.vs):
                v["coordinate"] = self.fiber_coordinates[t][i]
            gs.append(g)
        return gs

    # def _get_static_fiber_agent_data(self, type_name="microtubule"):
    #     """ Constructs agent data for per-segment fibers """
    #     gs = SimulariumConverter.get_segments(self.fiber_graph)
    #     self.max_n_subpoints = max(g.vcount() for g in gs) * 3
    #
    #     # Retrieve properties only once
    #     microtubule_properties = self.system_properties[type_name]
    #     radius = microtubule_properties['radius']
    #     color_code = self.color_map[microtubule_properties['color']]
    #
    #     # Prepare data structures efficiently
    #     _times = np.arange(self.n_steps, dtype=np.float32) * self.timestep
    #     n_agents = len(gs)
    #     _n_agents = [n_agents] * self.n_steps
    #     _viz_types = np.full((self.n_steps, n_agents), 1001, dtype=np.int32).tolist()
    #     _unique_ids = np.tile(np.arange(n_agents), (self.n_steps, 1)).tolist()
    #     _subpoints = [[None] * n_agents for _ in range(self.n_steps)]
    #     _n_subpoints = [[0] * n_agents for _ in range(self.n_steps)]
    #     _radii = [[radius] * n_agents for _ in range(self.n_steps)]
    #     _type_names = [[type_name] * n_agents for _ in range(self.n_steps)]
    #     _positions = np.zeros((self.n_steps, n_agents, 3), dtype=np.float32)
    #
    #     for t in range(self.n_steps):
    #         for i, g in enumerate(gs):
    #             g_ordered = SimulariumConverter.reorder_vertices(g)
    #             g_coordinates = np.array(g_ordered.vs["coordinate"], dtype=np.float32).flatten()
    #             _subpoints[t][i] = g_coordinates
    #             _n_subpoints[t][i] = g_coordinates.size
    #
    #     agent_data = AgentData(
    #         times=_times,
    #         n_agents=_n_agents,
    #         types=_type_names,
    #         positions=_positions,
    #         viz_types=_viz_types,
    #         unique_ids=_unique_ids,
    #         subpoints=_subpoints,
    #         n_subpoints=_n_subpoints,
    #         radii=_radii,
    #     )
    #
    #     agent_data.display_data[type_name] = DisplayData(
    #         name=type_name,
    #         color=color_code,
    #         display_type=DISPLAY_TYPE.FIBER,
    #     )
    #     return agent_data
    #
    # def _get_dynamic_fiber_agent_data(self, type_name="microtubule"):
    #     """ Constructs agent data for dynamic fibers """
    #     if isinstance(self.fiber_graph, list):
    #         print("_get_dynamic_fiber_agent_data: fiber_graph is a list")
    #         gs = []
    #         frame_indices = np.arange(0, len(self.fiber_coordinates), self.stride, dtype=int)
    #         for i in frame_indices:
    #             gs_frame = []
    #             start_idx = 0
    #             for g_segment in self.fiber_graph:
    #                 n_vs = g_segment.vcount()
    #                 end_idx = start_idx + n_vs
    #
    #                 g = ig.Graph(directed=False)
    #                 g.add_vertices(n_vs)
    #
    #                 # Update the coordinates of the vertices
    #                 g.vs["coordinate"] = self.fiber_coordinates[i][start_idx:end_idx]
    #
    #                 # Add the edges
    #                 edges = g_segment.get_edgelist()
    #                 g.add_edges(edges)
    #                 gs_frame.append(g)
    #                 start_idx = end_idx
    #             gs.append(gs_frame)
    #
    #         self.max_n_subpoints = sum([g.vcount() for g in gs[0]])
    #         n_agents = len(gs[0])
    #
    #     if isinstance(self.fiber_graph, ig.Graph):
    #         print("_get_dynamic_fiber_agent_data: fiber_graph is not a list")
    #         gs = []
    #         frame_indices = np.arange(0, len(self.fiber_coordinates), self.stride, dtype=int)
    #         for i in frame_indices:
    #             n_vs = self.fiber_graph.vcount()
    #             # start_idx = 0
    #
    #             g_frame = ig.Graph(directed=False)
    #             g_frame.add_vertices(n_vs)
    #
    #             # Update the coordinates of the vertices
    #             g_frame.vs["coordinate"] = self.fiber_coordinates[i]
    #
    #             # Add the edges
    #             edges = self.fiber_graph.get_edgelist()
    #             g_frame.add_edges(edges)
    #             gs.append(g_frame)
    #
    #         is_branching = False
    #         for g_frame in gs:
    #             components = g_frame.connected_components(mode="weak")
    #             components = [g_frame.subgraph(component) for component in components]
    #
    #             for g in components:
    #                 # Count the number of vertices of degree 1
    #                 n_deg1 = len([v for v in g.vs if v.degree() == 1])
    #                 if n_deg1 != 2:
    #                     print("Branching detected!")
    #                     is_branching = True
    #                     break
    #
    #             if is_branching:
    #                 break
    #             else:
    #                 continue
    #
    #         segments = []
    #         for g in components:
    #             segments.extend(self.get_segments(g))
    #
    #         vcount_segments = [g.vcount() for g in segments]
    #         self.max_n_subpoints = max(vcount_segments) * 3
    #         n_agents = len(segments)
    #
    #     # Retrieve properties only once
    #     microtubule_properties = self.system_properties[type_name]
    #     radius = microtubule_properties['radius']
    #     color_code = self.color_map[microtubule_properties['color']]
    #
    #     # Prepare data structures efficiently
    #     _times = np.arange(self.n_steps, dtype=np.float32) * self.timestep
    #     _n_agents = [n_agents] * self.n_steps
    #     _viz_types = np.full((self.n_steps, n_agents), 1001, dtype=np.int32).tolist()
    #     _unique_ids = np.tile(np.arange(n_agents), (self.n_steps, 1)).tolist()
    #     _subpoints = [[None] * n_agents for _ in range(self.n_steps)]
    #     _n_subpoints = [[0] * n_agents for _ in range(self.n_steps)]
    #     _radii = [[radius] * n_agents for _ in range(self.n_steps)]
    #     _type_names = [[type_name] * n_agents for _ in range(self.n_steps)]
    #     _positions = np.zeros((self.n_steps, n_agents, 3), dtype=np.float32)
    #
    #     # Iterate over each frame
    #     for frame_idx, g_frame in enumerate(gs):
    #         # Retrieve the graph for the current frame
    #         if isinstance(g_frame, list):
    #             gs_segments = g_frame
    #         else:
    #             # gs_segments = SimulariumConverter.get_segments(g_frame)
    #             components = g_frame.connected_components(mode="weak")
    #             gs_segments = []
    #             for component in components:
    #                 g_frame_component = g_frame.subgraph(component)
    #                 gs_segments.extend(self.get_segments(g_frame_component))
    #                 # gs_segments.append(g_frame.subgraph(component))
    #         for i, g in enumerate(gs_segments):
    #             # g_ordered = self.reorder_vertices(g)
    #             g_coordinates = np.array(g.vs["coordinate"], dtype=np.float32).flatten()
    #             _subpoints[frame_idx][i] = g_coordinates
    #             _n_subpoints[frame_idx][i] = g_coordinates.size
    #
    #     agent_data = AgentData(
    #         times=_times,
    #         n_agents=_n_agents,
    #         types=_type_names,
    #         positions=_positions,
    #         viz_types=_viz_types,
    #         unique_ids=_unique_ids,
    #         subpoints=_subpoints,
    #         n_subpoints=_n_subpoints,
    #         radii=_radii,
    #     )
    #
    #     agent_data.display_data[type_name] = DisplayData(
    #         name=type_name,
    #         color=color_code,
    #         display_type=DISPLAY_TYPE.FIBER,
    #     )
    #     return agent_data

    def _get_dynamic_fiber_agent_data(self, type_name="microtubule"):
        """ Constructs agent data for dynamic fibers """
        if isinstance(self.fiber_graph, list):
            print("_get_dynamic_fiber_agent_data: fiber_graph is a list")
            gs = []
            frame_indices = np.arange(0, len(self.fiber_coordinates), self.stride, dtype=int)
            for i in frame_indices:
                gs_frame = []
                start_idx = 0
                for g_segment in self.fiber_graph:
                    n_vs = g_segment.vcount()
                    end_idx = start_idx + n_vs

                    g = ig.Graph(directed=False)
                    g.add_vertices(n_vs)

                    # Update the coordinates of the vertices
                    g.vs["coordinate"] = self.fiber_coordinates[i][start_idx:end_idx]

                    # Add the edges
                    edges = g_segment.get_edgelist()
                    g.add_edges(edges)
                    gs_frame.append(g)
                    start_idx = end_idx
                gs.append(gs_frame)

            self.max_n_subpoints = sum([g.vcount() for g in gs[0]]) * 3  # Ensuring the correct calculation
            n_agents = len(gs[0])

        elif isinstance(self.fiber_graph, ig.Graph):
            print("_get_dynamic_fiber_agent_data: fiber_graph is not a list")
            gs = []
            if self.strided_fiber_coordinates:
                # Provided coordinates are already strided and should be used as is
                frame_indices = np.arange(0, len(self.fiber_coordinates), dtype=int)
            else:
                # Provided coordinates need to be strided
                frame_indices = np.arange(0, len(self.fiber_coordinates), self.stride, dtype=int)


            for i in frame_indices:
                n_vs = self.fiber_graph.vcount()

                g_frame = ig.Graph(directed=False)
                g_frame.add_vertices(n_vs)

                # Update the coordinates of the vertices
                g_frame.vs["coordinate"] = self.fiber_coordinates[i]

                # Add the edges
                edges = self.fiber_graph.get_edgelist()
                g_frame.add_edges(edges)
                gs.append(g_frame)

            # Ensure non-branching check
            is_branching = False
            for g_frame in gs:
                components = g_frame.connected_components(mode="weak")
                components = [g_frame.subgraph(component) for component in components]

                for g in components:
                    # Count the number of vertices of degree 1
                    n_deg1 = len([v for v in g.vs if v.degree() == 1])
                    if n_deg1 != 2:
                        print("Branching detected!")
                        is_branching = True
                        break

                if is_branching:
                    break

            segments = []
            for g in components:
                segments.extend(self.get_segments_with_anchor_nodes(g))

            vcount_segments = [g.vcount() for g in segments]
            self.max_n_subpoints = max(vcount_segments) * 3  # Correct calculation for max subpoints
            n_agents = len(segments)

        # Retrieve properties only once
        microtubule_properties = self.system_properties[type_name]
        radius = microtubule_properties['radius']
        color_code = self.color_map[microtubule_properties['color']]

        # Prepare data structures efficiently
        _times = np.arange(self.n_steps, dtype=np.float32) * self.timestep
        _n_agents = [n_agents] * self.n_steps
        _viz_types = np.full((self.n_steps, n_agents), 1001, dtype=np.int32).tolist()
        _unique_ids = np.tile(np.arange(n_agents), (self.n_steps, 1)).tolist()
        _subpoints = np.zeros((self.n_steps, n_agents, self.max_n_subpoints),
                              dtype=np.float32)  # Updated initialization
        _n_subpoints = np.zeros((self.n_steps, n_agents), dtype=int)
        _radii = [[radius] * n_agents for _ in range(self.n_steps)]
        _type_names = [[type_name] * n_agents for _ in range(self.n_steps)]
        _positions = np.zeros((self.n_steps, n_agents, 3), dtype=np.float32)

        # Iterate over each frame
        for frame_idx, g_frame in enumerate(gs):
            # Retrieve the graph for the current frame
            if isinstance(g_frame, list):
                gs_segments = g_frame
            else:
                components = g_frame.connected_components(mode="weak")
                gs_segments = []
                for component in components:
                    g_frame_component = g_frame.subgraph(component)
                    gs_segments.extend(self.get_segments_with_anchor_nodes(g_frame_component))

            for i, g in enumerate(gs_segments):
                g_coordinates = np.array(g.vs["coordinate"], dtype=np.float32).flatten()

                # Resize or pad g_coordinates as needed
                if g_coordinates.size < self.max_n_subpoints:
                    padded_coordinates = np.zeros(self.max_n_subpoints, dtype=np.float32)
                    padded_coordinates[:g_coordinates.size] = g_coordinates
                else:
                    padded_coordinates = g_coordinates[:self.max_n_subpoints]

                _subpoints[frame_idx, i, :] = padded_coordinates  # Assign with consistent size
                _n_subpoints[frame_idx, i] = g_coordinates.size

        agent_data = AgentData(
            times=_times,
            n_agents=_n_agents,
            types=_type_names,
            positions=_positions,
            viz_types=_viz_types,
            unique_ids=_unique_ids,
            subpoints=_subpoints,
            n_subpoints=_n_subpoints,
            radii=_radii,
        )

        agent_data.display_data[type_name] = DisplayData(
            name=type_name,
            color=color_code,
            display_type=DISPLAY_TYPE.FIBER,
        )
        return agent_data

    @staticmethod
    def get_segments_with_anchor_nodes(g, as_undirected=True):
        """ Returns linear segments from a graph object."""
        if as_undirected:
            g = g.as_undirected()

        # Remove any degree 0 nodes
        g.delete_vertices([v.index for v in g.vs if v.degree() == 0])

        branch_nodes = [v.index for v in g.vs if v.degree() > 2]
        terminal_nodes = [v.index for v in g.vs if v.degree() == 1]
        boundary_nodes = branch_nodes + terminal_nodes

        # Add a type attribute to the branch and terminal nodes
        for node in branch_nodes:
            g.vs[node]["type"] = "branch"
        for node in terminal_nodes:
            g.vs[node]["type"] = "terminal"

        segment_sequences = []
        for boundary_node in boundary_nodes:
            neighbors = g.neighbors(boundary_node)
            for neighbor in neighbors:
                segment = [boundary_node, neighbor]
                while True:
                    next_neighbor = [n for n in g.neighbors(neighbor) if n not in segment]
                    if not next_neighbor or (len(segment) > 2 and next_neighbor[0] == segment[0]):
                        break
                    neighbor = next_neighbor[0]
                    segment.append(neighbor)
                    if g.vs[neighbor].degree() > 2 or g.vs[neighbor].degree() == 1:
                        break
                if segment in segment_sequences or list(reversed(segment)) in segment_sequences:
                    continue
                segment_sequences.append(segment)

        graph_segments = []
        for segment in segment_sequences:
            # Create a new graph for each segment
            subgraph = g.subgraph(segment)
            graph_segments.append(subgraph)
        # return graph_segments

        ordered_segments = []
        for g in graph_segments:
            ordered_segments.append(SimulariumConverter.reorder_vertices(g))
        return ordered_segments

    @staticmethod
    def reorder_vertices(g):
        """ Reorders vertices according to connectivity """
        v_end1, v_end2 = g.vs.select(_degree=1)
        v_path = g.get_shortest_paths(v_end1, v_end2)[0]
        g_new = ig.Graph(directed=False)
        g_new.add_vertex()
        g_new.vs[0]['coordinate'] = g.vs[v_path[0]]['coordinate']
        for i in range(1, len(v_path)):
            v = g.vs[v_path[i]]  # get vertex from other graph
            g_new.add_vertex()
            v_current = g_new.vs[-1]
            v_current['coordinate'] = np.array(v['coordinate'])
            g_new.add_edge(v_current.index - 1, v_current.index)
        return g_new.simplify()

    def get_camera_data(self, focus_point=None, view_type='front'):
        """ Method which calculates the camera position for a given view type and field of view (fov) """
        focus_point = np.array([0.] * 3) if focus_point is None else focus_point
        view_type = 'oblique' if view_type is None else view_type
        fov_radians = np.radians(self.fov_degrees)

        dist_to_focus_point = np.max(self.box_size) / (2 * np.tan(fov_radians / 2))
        view_dict = {'oblique': (np.array([1., 1., 1.]) / np.sqrt(3)),
                     'top': np.array([0., 0., 1.]),
                     'bottom': np.array([0., 0., -1.]),
                     'front': np.array([1., 0., 0.]),
                     'back': np.array([-1., 0., 0.]),
                     'left': np.array([0., -1., 0.]),
                     'right': np.array([0., 1., 0.])}

        camera_position = (focus_point + dist_to_focus_point * view_dict[view_type]) * 2
        return CameraData(position=camera_position, look_at_position=focus_point, fov_degrees=self.fov_degrees)

    def save(self, outfile="out"):
        """ Saves the trajectory data to a file """
        if self.fiber_graph is not None:
            self.get_fiber_agent_data()

            model_meta_data = ModelMetaData(title="Whole-Cell Digital Twin of Cal27 Cell",
                                            authors="Arkfeld, et al.")


            try:
                readdy_data = self.create_readdy_data()
                self.particle_agent_data = ReaddyConverter(readdy_data)._get_agent_data(readdy_data)[0]
                self.particle_agent_data.display_data = readdy_data.display_data

                # camera_data = self.get_camera_data()
                # self.box_size *= np.array([100., 100., 100.])
                camera_data = None
                trajectory_data = TrajectoryData(
                    meta_data=MetaData(
                        box_size=self.box_size,
                        # box_size=np.array([0., 0., 0.]),
                        trajectory_title="Whole-Cell Digital Twin of Cal27 Cell",
                        model_meta_data=model_meta_data,
                        camera_defaults=camera_data
                    ),
                    agent_data=self.fiber_agent_data,
                    time_units=self.time_units,
                    spatial_units=self.spatial_units,
                )
                trajectory_data.append_agents_with_subpoints_optimized(self.particle_agent_data)
            except:
                print("Error in readdy data; Saving fiber data only...")
                # camera_data = self.get_camera_data()
                camera_data = None
                trajectory_data = TrajectoryData(
                    meta_data=MetaData(
                        box_size=self.box_size,
                        trajectory_title="Whole-Cell Digital Twin of Cal27 Cell",
                        model_meta_data=model_meta_data,
                        camera_defaults=camera_data
                    ),
                    agent_data=self.fiber_agent_data,
                    time_units=self.time_units,
                    spatial_units=self.spatial_units,
                )
            TrajectoryConverter(trajectory_data).save(outfile)

        else:
            print("No fiber graph provided; Saving particle data only...")
            readdy_data = self.create_readdy_data()
            ReaddyConverter(readdy_data).save(outfile)

if __name__ == "__main__":
    # Example usage
    data_dir_fn = lambda tid, condition, cid: f"/home/earkfeld/PycharmProjects/mitosim/data/trajectories/production_runs/t{tid}/{condition}/cell_{cid}/"
    infile_fn = lambda condition, cid, rid: f"{condition}_c{cid}_{rid}.h5"
    outfile_fn = lambda condition, cid, rid: f"{condition}_c{cid}_{rid}_vis"

    condition = "control"
    tid = 9
    cid = 1
    rid = 0

    infile = data_dir_fn(tid, condition, cid) + infile_fn(condition, cid, rid)
    # outfile = data_dir_fn(tid, condition, cid) + outfile_fn(condition, cid, rid)
    outfile = f"./{condition}"

    processor = SimulariumConverter(
        readdy_h5_path=infile,
        with_fibers=False,
    )

    processor.save(outfile=outfile)

    """
    Note: Modified Simularium Files:
    trajectory_data.py
    agent_data.py
    """
