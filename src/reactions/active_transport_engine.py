"""
Decentralized Active Transport Engine
"""

import readdy
import pickle
from typing import Union, List
import numpy as np
import igraph as ig
from collections import defaultdict
from readdy.api.utils import vec3_of
from dataclasses import dataclass, field
from tqdm import trange
from src.core.readdy_utils import ReaddyUtils
from readdy.api.experimental.action_factory import BreakConfig, ReactionConfig

ut = readdy.units

def from_vec3(vec):
    if not isinstance(vec, np.ndarray):
        vec = np.array([vec[0], vec[1], vec[2]])
    return vec

def get_projection_coordinate(pos, v1, v2):
    """
    Calculates the projection of the vector from v1 to pos onto the vector defined by points v1 and v2,
    ensuring the coordinate lies along the segment between v1 and v2.

    Parameters:
    pos (np.ndarray): Position to project.
    v1 (np.ndarray): Starting point of the vector defining the segment.
    v2 (np.ndarray): Ending point of the vector defining the segment.

    Returns:
    np.ndarray: The projected coordinate constrained to lie between v1 and v2.
    """
    a = pos - v1  # Vector from v1 to pos
    b = v2 - v1  # Vector from v1 to v2 (the direction of the line segment)
    proj = np.dot(a, b) / np.dot(b, b) * b
    proj_clipped = np.clip(proj, 0, np.linalg.norm(b)) * (b / np.linalg.norm(b))
    coordinate = v1 + proj_clipped
    return coordinate

def rate_to_probability(rate, timestep, **kwargs):
    time_unit = kwargs.get("time_unit", "s")

    assert type(rate) in [float, ut.Quantity], "Rate must be a float, or Quantity object."
    assert type(timestep) in [float, ut.Quantity], "Timestep must be a float, or Quantity object."

    if isinstance(rate, ut.Quantity):
        rate = rate.to(f"{time_unit}^-1").magnitude

    if isinstance(timestep, ut.Quantity):
        timestep = timestep.to(time_unit).magnitude
    print(f"(rate_to_probability) Rate: {rate}, Timestep: {timestep}")
    return 1 - np.exp(-rate * timestep)

def calculate_motor_step_probability(target_velocity, step_length, timestep, **kwargs):
    """ Calculates the probability of a motor taking a step based on the target velocity. """
    if not isinstance(target_velocity, ut.Quantity):
        target_velocity = target_velocity * ut.um / ut.s
    if not isinstance(step_length, ut.Quantity):
        step_length = step_length * ut.um

    # Calculate the step rate based on the target velocity
    step_rate = target_velocity / step_length
    p_step = rate_to_probability(step_rate, timestep, **kwargs)
    return p_step

def reformat_vec3_trajectory(vec):
    vec_new = np.zeros(shape=(len(vec), len(vec[0]), 3), dtype=np.float32)
    for i in range(len(vec)):
        for j in range(len(vec[i])):
            vec_new[i][j] = np.array([vec[i][j][0], vec[i][j][1], vec[i][j][2]])
    return vec_new

# TODO: Clean up unused parameters
@dataclass
class Motor:
    motor_type: str = None
    polarity: int = 1
    radius: float = 0.05
    binding_distance: float = 0.5
    path_selection: str = "random"
    step_length: float = 0.05
    velocity: float = 1.0
    step_rate: float = 1.0
    binding_rate: float = 1.0
    unbinding_rate: float = 1.0
    activation_rate: float = 1.0
    deactivation_rate: float = 1.0
    p_step: float = 1.0
    p_binding: float = 1.0
    p_unbind: float = 1.0
    p_active: float = 1.0
    p_activate: float = 1.0
    p_deactivate: float = 1.0

class ActiveTransportEngine:
    def __init__(self, timestep, stride, **kwargs):
        self._timestep = timestep
        self._stride = stride

        self.uid_to_vertex_map = {}
        self.vertex_to_uid_map = {}
        self.buffer_map = {}
        self.cargo_to_motor_map = {}
        self.iterator_registry = {}
        self.motor_registry = {}

        self.vertex_coordinates = None
        self.edge_data = {}
        self.edge_vertex_map = defaultdict(dict)
        self.coupled_iterators = {}
        self._counts = {'lone_motor': 0}
        self._step_counts = {}
        self._event_counts = {}

        self.break_config = BreakConfig()
        self.reaction_config = ReactionConfig()

        self._count_steps = kwargs.get("count_steps", False)
        self._count_events = kwargs.get("count_events", True)
        self._single_motor = kwargs.get("single_motor", False) # Disables 2nd set of Add AT reactions
        self._end_behavior = kwargs.get("end_behavior", "remove")
        self._record_trajectory = kwargs.get("record_trajectory", False)

        self._end_behavior_options = ["remove_all", "remove", "invert", "invert_all"]
        self._it = 0

        assert self._end_behavior in self._end_behavior_options, \
            f"Invalid end behavior: {self._end_behavior}. \nOptions: {self._end_behavior_options}"

    def register_with_simulation(self,
                                 gs: any,
                                 simulation: readdy.Simulation):
        """ Sets up the graph data and registers the graph with the simulation. """
        if isinstance(gs, ig.Graph):
            # Split the graph into a list of connected components
            components = gs.connected_components(mode="weak")
            gs = [gs.subgraph(c) for c in components]

        index_buffer = 0
        for g in gs:
            coordinates = np.array(g.vs["coordinate"])
            ptypes = g.vs["ptype"] if "ptype" in g.vs.attributes() else ["tubulin"] * g.vcount()
            ttype = g.vs[0]["ttype"] if "ttype" in g.vs.attributes() else "Microtubule"
            top = simulation.add_topology(ttype, ptypes, coordinates)
            for edge in g.get_edgelist():
                top.get_graph().add_edge(edge[0], edge[1])

            # Set up the vertex map
            vs_top = top.get_graph().get_vertices()
            for i, v_top in enumerate(vs_top):
                puid = str(top.particle_id_of_vertex(v_top))
                v_idx = str(i + index_buffer)
                self.uid_to_vertex_map[puid] = v_idx

                # Initialize entry in the edge_vertex_map
                self.edge_vertex_map[v_idx] = {"outgoing": [], "incoming": []}

            # Set up the edge data
            adjlist_out = g.get_adjlist(mode="out")
            adjlist_in = g.get_adjlist(mode="in")

            for i in range(len(adjlist_out)):
                v_current_idx = i + index_buffer
                vs_outgoing = adjlist_out[i]
                vs_outgoing_idxs = [v + index_buffer for v in vs_outgoing]
                self.edge_vertex_map[str(v_current_idx)]["outgoing"] = vs_outgoing_idxs

            for i in range(len(adjlist_in)):
                v_current_idx = i + index_buffer
                vs_incoming = adjlist_in[i]
                vs_incoming_idxs = [v + index_buffer for v in vs_incoming]
                self.edge_vertex_map[str(v_current_idx)]["incoming"] = vs_incoming_idxs
            index_buffer += len(g.vs)

    def get_motor_event_probabilities(self, parameters):
        """ Returns the motor event probabilities based on the given parameters. """
        probability_rate_map = {"p_binding": "binding_rate",
                                "p_unbind": "unbinding_rate",
                                "p_activate": "activation_rate",
                                "p_deactivate": "deactivation_rate"}

        for pkey, rvalue in probability_rate_map.items():
            if pkey not in parameters:
                pvalue = rate_to_probability(parameters[rvalue], self._timestep)
                parameters[pkey] = pvalue

        # Calculate the step probability
        if "p_step" not in parameters.keys():
            step_length = parameters["step_length"]
            if not isinstance(step_length, ut.Quantity):
                step_length = step_length * ut.um

            if "velocity" in parameters.keys():
                velocity = parameters["velocity"]
                print(f"(get_motor_event_probabilities) Velocity: {velocity}")
                if not isinstance(velocity, ut.Quantity):
                    velocity = velocity * ut.um / ut.s

                step_rate = velocity / step_length
                print(f"(get_motor_event_probabilities) Step Rate: {step_rate}")
                p_step = rate_to_probability(step_rate, self._timestep)
                print(f"(get_motor_event_probabilities) Step Probability: {p_step}")
            else:
                p_step = rate_to_probability(parameters["step_rate"], self._timestep)
            parameters["p_step"] = p_step

        print(f"(get_motor_event_probabilities) Motor Parameters:")
        for key, value in parameters.items():
            print(f"  {key}: {value}")
        return parameters

    def register_motor_type(self, motor_type, parameters, **kwargs):
        """Registers a new motor type with default properties."""
        motor_cls = kwargs.get("motor_module", None)
        if not motor_cls:
            parameters = self.get_motor_event_probabilities(parameters)
            motor_cls = Motor(motor_type, **parameters)
        self.motor_registry[motor_type] = motor_cls

        print(f"(register_motor_type) Registered motor: {motor_cls}")

        if self._count_steps and motor_type not in self._step_counts:
                self._step_counts[motor_type] = {}
        if self._count_events and motor_type not in self._event_counts:
                self._event_counts[motor_type] = {}

    def update_motor_parameter(self, uid, update_data):
        motor = self.motor_registry[uid]
        for key, value in update_data.items():
            setattr(motor, key, value)

    def delete_motor(self, uid):
        if uid in self.iterator_registry:
            del self.iterator_registry[uid]

    def set_buffer(self, site_uid, cargo_uid, cargo_uvec):
        if not isinstance(cargo_uid, str):
            cargo_uid = str(cargo_uid)
        if not isinstance(site_uid, str):
            site_uid = str(site_uid)

        if cargo_uvec is not None:
            motor_type = self._set_motor_type(site_uid, cargo_uvec)
        else:
            # Pick a random motor type
            motor_type = np.random.choice(["motor#kinesin", "motor#dynein"])
        self.buffer_map[cargo_uid] = (site_uid, motor_type)

    def delete_buffer(self, cargo_uid):
        if not isinstance(cargo_uid, str):
            cargo_uid = str(cargo_uid)
        if cargo_uid in self.buffer_map:
            del self.buffer_map[cargo_uid]

    # Iterator management methods
    def create_iterator(self, cargo_uid, motor_uid, motor_position, **kwargs):
        motor_type = kwargs.get("motor_type", None)
        if not isinstance(cargo_uid, str):
            cargo_uid = str(cargo_uid)
        if not isinstance(motor_uid, str):
            motor_uid = str(motor_uid)

        # cargo_uid --> site_uid --> source_vertex
        if motor_type is None:
            site_uid, motor_type = self.buffer_map[cargo_uid]
        else:
            site_uid, _ = self.buffer_map[cargo_uid]
        source_vertex = self.uid_to_vertex_map[site_uid]

        motor_position = from_vec3(motor_position)
        motor_obj = self.motor_registry[motor_type]
        self.iterator_registry[motor_uid] = ActiveTransportIterator(self,
                                                                    source_vertex,
                                                                    motor_position,
                                                                    motor_uid,
                                                                    motor_obj)

        if self._count_steps:
            self._step_counts[motor_type][motor_uid] = {"step": 0, "no_step": 0, "running": []}
        if self._count_events:
            self._event_counts[motor_type][motor_uid] = {"activate": 0, "deactivate": 0}

    def delete_iterator(self, motor_uid):
        if not isinstance(motor_uid, str):
            motor_uid = str(motor_uid)
        try:
            del self.iterator_registry[motor_uid]
        except Exception as e:
            print(f"(delete_iterator) Error: {e}")

    def is_primary_iterator(self, motor_uid):
        if not isinstance(motor_uid, str):
            motor_uid = str(motor_uid)
        return motor_uid in self.coupled_iterators.keys()

    def change_motor_type(self, uid, new_type):
        if not isinstance(uid, str):
            uid = str(uid)

        iterator = self.iterator_registry[uid]
        new_motor = self.motor_registry[new_type]
        old_motor = iterator.motor

        if new_motor.polarity != old_motor.polarity:
            new_v_target = iterator.source_vertex
            new_v_source = iterator.target_vertex
            iterator.target_vertex = new_v_target
            iterator.source_vertex = new_v_source
            iterator.motor = new_motor
        self.iterator_registry[uid] = new_motor

    def step(self, motor_uid):
        if not isinstance(motor_uid, str):
            motor_uid = str(motor_uid)
        iterator = self.iterator_registry[motor_uid]
        # TODO: This is probably redundant to some part of iterator's __next__ method
        if iterator:
            try:
                return next(iterator)
            except StopIteration:
                return None
        return None

    def update_site_data(self, coordinates):
        """ Updates the graph data with new site coordinates. """
        self.vertex_coordinates = coordinates

    def get_vertex_position(self, index):
        try:
            v_pos = from_vec3(self.vertex_coordinates[int(index)])
            return v_pos
        except Exception as e:
            return None

    def get_vector_data(self, source, target):
        return self.edge_data.get((source, target))

    def get_closest_vertex(self, position):
        distances = np.linalg.norm(self.vertex_coordinates - position, axis=1)
        return np.argmin(distances)

    def select_next_vertex(self, source_vertex, polarity, strategy="random"):
        """ Select the next vertex based on the chosen strategy using polarity directly. """
        # Get the next vertex or vertices using the polarity
        if polarity > 0:
            vs_next = self.edge_vertex_map[str(source_vertex)].get("incoming", [])
        else:
            vs_next = self.edge_vertex_map[str(source_vertex)].get("outgoing", [])

        # Apply the selection strategy
        if not vs_next:
            return None
        elif strategy == "all":
            return vs_next
        elif len(vs_next) == 1:
            return vs_next[0]
        elif strategy == "first":
            return vs_next[0]
        elif strategy == "random":
            return np.random.choice(vs_next)
        else:
            return None

    def get_motor_type(self, cargo_uid):
        if not isinstance(cargo_uid, str):
            cargo_uid = str(cargo_uid)
        site_uid, motor_type = self.buffer_map[cargo_uid]
        return motor_type

    def _set_motor_type(self, site_uid, cargo_uvec):
        """ Returns the motor type based on the cargo unit vector and site's directed edges. """
        # print(f"(set_motor_type) Called")
        if not isinstance(site_uid, str):
            site_uid = str(site_uid)

        v_source = self.uid_to_vertex_map[site_uid]
        vs_target_pos = [self.select_next_vertex(v_source, 1, strategy="random")]
        vs_target_neg = [self.select_next_vertex(v_source, -1, strategy="random")]

        # Get the orientation weights for all edges from the source vertex
        pos_vecs = []
        if vs_target_pos is not None and len(vs_target_pos) > 0:
            for v_target in vs_target_pos:
                weight = self.get_orientation_weight(v_source, v_target, cargo_uvec)
                if weight is not None:
                    pos_vecs.append(weight)

        neg_vecs = []
        if vs_target_neg is not None:
            for v_target in vs_target_neg:
                weight = self.get_orientation_weight(v_source, v_target, cargo_uvec)
                if weight is not None:
                    neg_vecs.append(weight)

        # Get the average orientation weight for all edges from the source vertex
        pos_weight = -1
        neg_weight = -1
        if len(pos_vecs) == 0 and len(neg_vecs) == 0:
            return None
        else:
            if len(pos_vecs) > 0:
                pos_weight = np.mean(pos_vecs)
            if len(neg_vecs) > 0:
                neg_weight = np.mean(neg_vecs)

        if pos_weight > neg_weight and pos_weight > 0:
            motor_type = "motor#kinesin"
        elif pos_weight < neg_weight and neg_weight > 0:
            motor_type = "motor#dynein"
        else:
            motor_type = None
        return motor_type

    def get_orientation_weight(self, v_source, v_target, uvec_cargo):
        try:
            source_pos = self.get_vertex_position(v_source)
            target_pos = self.get_vertex_position(v_target)
            vec_edge = target_pos - source_pos
            uvec_edge = vec_edge / (np.linalg.norm(vec_edge) + 1.e-32)
            dot_prod = np.dot(uvec_cargo, uvec_edge)
            return dot_prod
        except Exception as e:
            return None

    def save_state(self, file_path: str):
        """Saves the engine's state to a file using pickle."""
        state_data = {
            "uid_to_vertex_map": self.uid_to_vertex_map,
            "vertex_to_uid_map": self.vertex_to_uid_map,
            "buffer_map": self.buffer_map,
            "cargo_to_motor_map": self.cargo_to_motor_map,
            "iterator_registry": self.iterator_registry,
            "motor_registry": self.motor_registry,
            "vertex_coordinates": self.vertex_coordinates,
            "edge_data": self.edge_data,
            "edge_vertex_map": self.edge_vertex_map,
            "coupled_iterators": self.coupled_iterators,
            "_counts": self._counts,
            "_step_counts": self._step_counts,
            "_event_counts": self._event_counts,
            "_it": self._it,
        }

        with open(file_path, 'wb') as f:
            pickle.dump(state_data, f)
        print(f"State saved to {file_path}")

    def load_state(self, file_path: str):
        """Loads the engine's state from a file using pickle."""
        try:
            with open(file_path, 'rb') as f:
                state_data = pickle.load(f)

            # Restore state
            self.uid_to_vertex_map = state_data["uid_to_vertex_map"]
            self.vertex_to_uid_map = state_data["vertex_to_uid_map"]
            self.buffer_map = state_data["buffer_map"]
            self.cargo_to_motor_map = state_data["cargo_to_motor_map"]
            self.iterator_registry = state_data["iterator_registry"]
            self.motor_registry = state_data["motor_registry"]
            self.vertex_coordinates = state_data["vertex_coordinates"]
            self.edge_data = state_data["edge_data"]
            self.edge_vertex_map = state_data["edge_vertex_map"]
            self.coupled_iterators = state_data["coupled_iterators"]
            self._counts = state_data["_counts"]
            self._step_counts = state_data["_step_counts"]
            self._event_counts = state_data["_event_counts"]
            self._it = state_data["_it"]
            print(f"State loaded from {file_path}")
        except FileNotFoundError:
            print(f"Error: File {file_path} not found.")
        except Exception as e:
            print(f"Error loading state: {e}")

    def get_custom_loop(self,
                        system: readdy.ReactionDiffusionSystem,
                        simulation: readdy.Simulation,
                        timestep: any,
                        n_steps: int,
                        break_config: BreakConfig,
                        reaction_config: ReactionConfig,
                        with_active_transport: bool = True):

        _n_steps = n_steps
        _internal_timestep = timestep.magnitude
        _actions = simulation._actions
        _sim = simulation._simulation
        _system = system
        _break_config = break_config
        _reaction_config = reaction_config

        if self._count_events:
            arr = np.zeros(n_steps)
            for key in self.motor_registry.keys():
                self._event_counts[key] = {"activate": np.zeros_like(arr), "deactivate": np.zeros_like(arr), "unbind": np.zeros_like(arr)}
            print(self._count_events)

        def active_transport_simulation_loop():
            readdy_actions = _actions
            init = readdy_actions.initialize_kernel()
            diffuse = readdy_actions.integrator_euler_brownian_dynamics(_internal_timestep)
            calculate_forces = readdy_actions.calculate_forces()
            break_bonds = readdy_actions.break_bonds(_internal_timestep, _break_config)
            perform_reaction = readdy_actions.action_reaction(_reaction_config)
            create_nl = readdy_actions.create_neighbor_list(_system.calculate_max_cutoff().magnitude)
            update_nl = readdy_actions.update_neighbor_list()
            react_particles = readdy_actions.reaction_handler_gillespie(_internal_timestep)
            react_topologies = readdy_actions.topology_reaction_handler(_internal_timestep)
            observe = readdy_actions.evaluate_observables()

            # Main Loop
            init()
            create_nl()
            update_nl()
            calculate_forces()
            self.update_site_data(_sim.get_particle_positions("tubulin"))
            observe(0)

            for t in trange(1, _n_steps + 1):
                diffuse()
                update_nl()
                self.update_site_data(_sim.get_particle_positions("tubulin"))
                perform_reaction()
                break_bonds()
                react_particles()
                react_topologies()
                update_nl()
                calculate_forces()
                observe(t)
                self._it += 1

        def passive_transport_simulation_loop():
            readdy_actions = _actions
            init = readdy_actions.initialize_kernel()
            diffuse = readdy_actions.integrator_euler_brownian_dynamics(_internal_timestep)
            calculate_forces = readdy_actions.calculate_forces()
            create_nl = readdy_actions.create_neighbor_list(_system.calculate_max_cutoff().magnitude)
            update_nl = readdy_actions.update_neighbor_list()
            react_particles = readdy_actions.reaction_handler_gillespie(_internal_timestep)
            react_topologies = readdy_actions.topology_reaction_handler(_internal_timestep)
            observe = readdy_actions.evaluate_observables()

            # Main Loop
            init()
            create_nl()
            update_nl()
            calculate_forces()
            observe(0)
            for t in trange(1, _n_steps + 1):
                diffuse()
                update_nl()
                react_particles()
                react_topologies()
                update_nl()
                calculate_forces()
                observe(t)
                self._it += 1

        if with_active_transport:
            return active_transport_simulation_loop
        else:
            return passive_transport_simulation_loop

    @property
    def end_behavior(self):
        return self._end_behavior

    @end_behavior.setter
    def end_behavior(self, value):
        assert value in self._end_behavior_options, \
            f"Invalid end behavior: {value}. \nOptions: {self._end_behavior_options}"
        self._end_behavior = value

    @property
    def single_motor(self):
        return self._single_motor

    @single_motor.setter
    def single_motor(self, value):
        assert type(value) == bool, "Must be boolean."
        self._single_motor = value

    @property
    def count_steps(self):
        return self._count_steps

    @count_steps.setter
    def count_steps(self, value):
        assert type(value) == bool, "Must be boolean."
        self._count_steps = value

    @property
    def step_counts(self):
        return self._step_counts

    @property
    def count_events(self):
        return self._count_events

    @count_events.setter
    def count_events(self, value):
        assert type(value) == bool, "Must be boolean."
        self._count_events = value

    @property
    def event_counts(self):
        return self._event_counts

    @property
    def counts(self):
        return self._counts

    @property
    def record_trajectory(self):
        return self._record_trajectory

    @record_trajectory.setter
    def record_trajectory(self, value):
        assert type(value) == bool, "Must be boolean."
        self._record_trajectory = value


class ActiveTransportIterator:
    def __init__(self,
                 engine: ActiveTransportEngine,
                 source_vertex: int,
                 position: np.ndarray,
                 uid: str,
                 motor: Motor,
                 **kwargs):

        self.engine = engine
        self.position = position
        self.uid = uid
        self.motor = motor
        self.source_vertex = int(source_vertex)
        self.target_vertex = None
        self.vector_to_target = None
        self.unit_vector_to_target = None
        self.uvec_source_target = None
        self.dist_from_source = 0.
        self.old_source_pos = None
        self.old_target_pos = None
        self._is_active = np.random.choice([True, False],
                                           p=[self.motor.p_active, 1 - self.motor.p_active])

    @property
    def is_active(self):
        return self._is_active

    def calculate_distance_from_source(self):
        if self.dist_from_source is None:
            pos_source = from_vec3(self.engine.vertex_coordinates[self.source_vertex])
            self.dist_from_source = np.linalg.norm(self.position - pos_source)

    def update_target_vertex(self):
        self.target_vertex = int(self.engine.select_next_vertex(self.source_vertex,
                                                                self.motor.polarity,
                                                                self.motor.path_selection))

        self.vector_to_target = from_vec3(self.engine.vertex_coordinates[self.target_vertex]) - self.position
        self.unit_vector_to_target = self.vector_to_target / (np.linalg.norm(self.vector_to_target) + 1.e-32)

    def update_active_motor_position(self):
        """ Updates the motor position along the edge. """
        if np.random.choice([True, False], p=[self.motor.p_step, 1 - self.motor.p_step]):
            self.position += self.motor.step_length * self.unit_vector_to_target
            # print(f"(ATI) Motor Step: {self.uid}")
            if self.engine._count_steps:
                self.engine._step_counts[self.motor.motor_type][self.uid]["step"] += 1
                self.engine._step_counts[self.motor.motor_type][self.uid]["running"].append(1)
        else:
            # print(f"(ATI) No Motor Step: {self.uid}")
            if self.engine._count_steps:
                self.engine._step_counts[self.motor.motor_type][self.uid]["no_step"] += 1

        new_vector_to_target = from_vec3(self.engine.vertex_coordinates[self.target_vertex]) - self.position
        new_unit_vector_to_target = new_vector_to_target / np.linalg.norm(new_vector_to_target + 1.e-32)

        # Iterate source/target vertices if the motor has passed the target vertex
        if np.dot(self.unit_vector_to_target, new_unit_vector_to_target) < 0:
            self.source_vertex = self.target_vertex
            self.target_vertex = None
            self.vector_to_target = None
            self.old_source_pos = self.old_target_pos
            self.old_target_pos = None

        self.unit_vector_to_target = new_unit_vector_to_target
        self.vector_to_target = new_vector_to_target

    # Original
    def apply_edge_drift_correction(self):
        """Calculates a more accurate drift correction for the motor along the edge,
           taking into account the diffusion of the source and target vertices."""

        # # Case 1: No position history; applied only for first time step after binding
        if self.old_source_pos is None and self.old_target_pos is None:
            self.old_source_pos = self.engine.get_vertex_position(self.source_vertex)
            self.old_target_pos = self.engine.get_vertex_position(self.target_vertex)
            vec_to_source = self.old_source_pos - self.position
            self.position += vec_to_source

        # Case 2: Source/Target vertices were just updated
        elif self.old_target_pos is None and self.old_source_pos is not None:
            self.position = self.old_source_pos
            self.old_target_pos = self.engine.get_vertex_position(self.target_vertex)
            self.old_source_pos = self.engine.get_vertex_position(self.source_vertex)

        elif self.old_target_pos is not None and self.old_source_pos is not None:
            # Case 2: Have position history; applied for all time steps after the first
            pos_source = self.engine.get_vertex_position(self.source_vertex)
            pos_target = self.engine.get_vertex_position(self.target_vertex)

            # Calculate the displacement vectors
            vec_source = pos_source - self.old_source_pos
            vec_target = pos_target - self.old_target_pos

            # Calculate the relative position of the motor along the edge (0 at source, 1 at target)
            edge_vector = self.old_target_pos - self.old_source_pos
            edge_length = np.linalg.norm(edge_vector)
            if edge_length != 0:
                motor_projection = np.dot(self.position - self.old_source_pos, edge_vector) / (edge_length ** 2)
                motor_projection = np.clip(motor_projection, 0, 1)  # Ensure it lies within [0, 1]
            else:
                motor_projection = 0.5  # Default to the midpoint if the edge length is zero

            # Apply a weighted correction based on the motor's position along the edge
            drift_correction = (1 - motor_projection) * vec_source + motor_projection * vec_target
            self.position += drift_correction

            # Cache the updated positions of the source and target vertices
            self.old_source_pos = pos_source
            self.old_target_pos = pos_target
        else:
            raise ValueError("(ATI - apply_edge_drift_correction) Invalid state.")


    def __next__(self):
        if self.source_vertex is None:
            return None

        if self.target_vertex is None:
            try:
                self.update_target_vertex()
            except TypeError:
                return None  # end of the line

        # >>> BEGIN MODIFIED V2>>>
        # Evaluate P(Unbinding) and return None if unbinding event occurs to trigger disassembly
        if np.random.choice([True, False], p=[self.motor.p_unbind, 1 - self.motor.p_unbind]):
            if self.engine._event_counts:
                self.engine._event_counts[self.motor.motor_type]["unbind"][self.engine._it] += 1
            return None  # Equivalent to the end of the line

        # Correct for drift regardless of (possible) events to follow
        self.apply_edge_drift_correction()

        # Evaluate P(Event | Active)'s
        if self._is_active:
            # Deactivation event
            if np.random.choice([True, False], p=[self.motor.p_deactivate, 1 - self.motor.p_deactivate]):
                self._is_active = False
                if self.engine.count_events:
                    self.engine._event_counts[self.motor.motor_type]["deactivate"][self.engine._it] += 1
                # print(f"(ATI) Deactivated motor: {self.uid}")
            else:
                self.update_active_motor_position()

        # Evaluate P(Event | Inactive)'s
        else:
            # Activation event
            if np.random.choice([True, False], p=[self.motor.p_activate, 1 - self.motor.p_activate]):
                self._is_active = True
                if self.engine.count_events:
                    self.engine._event_counts[self.motor.motor_type]["activate"][self.engine._it] += 1
                    # print(f"(ATI) Activated motor: {self.uid}")
        return vec3_of(self.position)
