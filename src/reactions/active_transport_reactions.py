import readdy
import itertools
import re
from pprint import pprint
import numpy as np
from .active_transport_engine import ActiveTransportEngine, from_vec3
from src.core.readdy_utils import ReaddyUtils

ut = readdy.units


def set_active_transport_engine(e):
    global engine
    engine = e


def convert_units(d, time_unit='s', length_unit='um'):
    for key, value in d.items():
        if isinstance(value, dict):
            convert_units(value)
        elif isinstance(value, ut.Quantity):
            if value.dimensionality == ut.Unit(f'1/{time_unit}').dimensionality:
                d[key] = value.to(f'1/{time_unit}').magnitude
            elif value.dimensionality == ut.Unit(length_unit).dimensionality:
                d[key] = value.to(length_unit).magnitude
        else:
            continue
    return d

class ActiveTransportReactions:
    """ Reaction Network Class for Active Transport """

    REACTION_FLAGS = ["#AT_IM",
                      "#AT_active",
                      "#AT_inactive"]

    def __init__(self, parameters, **kwargs):
        time_unit = kwargs.get("time_unit", "s")
        length_unit = kwargs.get("length_unit", "um")
        self.parameters = convert_units(parameters, time_unit, length_unit)
        # self.parameters = parameters
        engine = kwargs.get("engine", None)
        set_active_transport_engine(engine)
        # print("Engine set!")
        self._debug = False

    def register(self,
                 system: readdy.ReactionDiffusionSystem):
        """ Registers all active transport reactions with the system. """

        # Non-specific Reaction Functions
        ## Remove lone motor particles
        system.reactions.add_decay(name="motor decay",
                                   particle_type="motor#decay",
                                   rate=1.e32)

        ## Add active transport reactions (Motor selection handled by engine)
        # TODO: Set global rate rather than just using kinesin's

        # Case 1: Adding motor to Cargo (N=0)
        system.topologies.add_spatial_reaction(
            descriptor="Add_AT_R1a: Mitochondria(mitochondria#terminal) + Microtubule(tubulin) --> Microtubule#AT-IM1(mitochondria#AT-IM1--tubulin)",
            rate=self.parameters[f"motor#kinesin"]["binding_rate"],
            radius=self.parameters["motor#kinesin"]["binding_distance"],
        )

        # Case 2: Adding motor to AT Cargo complex (N>=1); Req'd due to "#AT" flag
        if not engine.single_motor:
            print("Adding multimotor AT reactions")
            system.topologies.add_spatial_reaction(
                descriptor=f"Add_AT_R1b: Mitochondria#AT(mitochondria#terminal) + Microtubule(tubulin) --> Microtubule#AT-IM1(mitochondria#AT-IM1--tubulin)",
                rate=self.parameters[f"motor#kinesin"]["binding_rate"],
                radius=self.parameters["motor#kinesin"]["binding_distance"],
            )

        system.topologies.add_structural_reaction("Add_AT_R2",
                                                  topology_type="Microtubule#AT-IM1",
                                                  reaction_function=self.get_add_active_transport2_reaction_function(),
                                                  rate_function=lambda x: 1.e32,
                                                  expect_connected=False)

        system.topologies.add_structural_reaction("Add_AT_R3",
                                                  topology_type="Microtubule#AT-IM2",
                                                  reaction_function=self.get_add_active_transport3_reaction_function(),
                                                  rate_function=lambda x: 1.e32,
                                                  expect_connected=False)

        # TODO: Set this rate very high & update step function so the event probabilities are calculated per-motor
        # TODO: Process motor unbinding event probabilities in engine
        # system.topologies.add_structural_reaction(f'Step AT',
        #                                           topology_type=f'Mitochondria#AT',
        #                                           reaction_function=self.reaction_function_step,
        #                                           rate_function=lambda x: 1.e6,
        #                                           expect_connected=False)

        system.topologies.add_structural_reaction(f'Step AT',
                                                  topology_type=f'Mitochondria#AT',
                                                  reaction_function=self.reaction_function_step,
                                                  rate_function=lambda x: 1.e-32,
                                                  expect_connected=False)

        # self._register_configs(system)

    def _register_configs(self, system):
        """ Registers breakable configurations and reaction configurations. """
        motor_types = [key for key in self.parameters.keys() if "motor#" in key]
        ptype_map = system._context.particle_types.type_mapping
        print(ptype_map) if self._debug else None
        motor_ptype_ids = [ptype_map[mtype] for mtype in motor_types]
        mito_at_ptype_id = ptype_map["mitochondria#terminal#AT"]
        for motor_ptype_id in motor_ptype_ids:
            engine.break_config.add_breakable_pair(
                type1=mito_at_ptype_id,
                type2=motor_ptype_id,
                threshold_energy=self.parameters["transport_parameters"]["bond_energy_threshold"],
                rate=self.parameters["transport_parameters"]["rc_cleave_bond"])

    def test_reaction_function(self, topology):
        """ Test reaction function for debugging. """
        recipe = readdy.StructuralReactionRecipe(topology)
        print(f"Test Reaction called for Topology: {topology.type}")
        return recipe

    def get_rate_function(self, *args):
        """ Returns the rate function for the active transport reactions using provided parameter dictionary keys. """
        def rate_function(topology):
            return self.parameters[args[0]][args[1]]
        return rate_function

    def get_add_active_transport2_reaction_function(self):
        """ Returns the second reaction function for adding active transport. """
        def reaction_function_add_active_transport2(topology):
            """
            Adds a motor particle to the topology and sets engine buffer for creating a motor iterator.
            Note:
                Generalized for N-motors

            Reaction Description:
                T#IM1{m#AT-IM1--t} --> T#AT-IM2{m#AT-IM2--x#IM} + T#AT-IM2{t}
            """
            print(f"(Add_AT_2) Called") if self._debug else None
            recipe = readdy.StructuralReactionRecipe(topology)
            try:
                v_mito1 = ReaddyUtils.get_vertex_of_type(topology, "mitochondria#AT-IM1", exact_match=True)
                v_mito1_uid = ReaddyUtils.vertex_to_uid(topology, v_mito1)
            except Exception as e:
                # TODO: Set up for TypeError when trying to get uid for None
                print(ReaddyUtils.topology_to_string(topology)) if self._debug else None
                return recipe
            # v_mito1 = ReaddyUtils.get_vertex_of_type(topology, "mitochondria#AT-IM1", exact_match=True)
            try:
                v_mito2 = ReaddyUtils.get_neighbor_of_type(topology, v_mito1, "mitochondria#", exclude_vertices=[], exact_match=False)
            # TODO: Possible transient bug here when adding AT to single-particle topology
            except Exception as e:
                v_mito2 = None

            # Case 1: [Y] mitochondria#terminal, [Y] mitochondria#internal --> Oriented Binding
            if v_mito2 is not None:
                # Get the positions of the mito particles
                pos_mito1 = ReaddyUtils.get_vertex_position(topology, v_mito1)
                pos_mito2 = ReaddyUtils.get_vertex_position(topology, v_mito2)
                vec_mito21 = from_vec3(pos_mito1) - from_vec3(pos_mito2)
                uvec_mito = vec_mito21 / np.linalg.norm(vec_mito21)

            # Case 2: [Y] mitochondria#terminal, [N] mitochondria#internal --> Non-oriented Binding
            else:
                uvec_mito = None

            # Add motor, set buffer, update topology + particle types
            v_mt = ReaddyUtils.get_neighbor_of_type(topology,v_mito1, "tubulin", exclude_vertices=[], exact_match=True)
            v_mt_uid = ReaddyUtils.vertex_to_uid(topology, v_mt)
            pos_mt = from_vec3(ReaddyUtils.get_vertex_position(topology, v_mt))
            recipe.append_particle([v_mito1], "motor#IM", pos_mt)
            engine.set_buffer(site_uid=v_mt_uid, cargo_uid=v_mito1_uid, cargo_uvec=uvec_mito)
            recipe.change_particle_type(v_mito1, "mitochondria#AT-IM2")
            recipe.remove_edge(v_mito1, v_mt)
            recipe.change_topology_type("Microtubule#AT-IM2")
            return recipe
        return reaction_function_add_active_transport2

    def get_add_active_transport3_reaction_function(self):
        """ Returns the third reaction function for adding active transport. """
        def reaction_function_add_active_transport3(topology):
            """ Sets motor type + iterator for cargo or resets topology for microtubules. """

            print(f"(Add_AT_3) Called") if self._debug else None
            recipe = readdy.StructuralReactionRecipe(topology)
            vs_mito = ReaddyUtils.get_vertices_of_type(topology, "mitochondria#", exact_match=False)
            vs_motor = ReaddyUtils.get_vertices_of_type(topology, "motor#", exact_match=False)
            vs_tubulin = ReaddyUtils.get_vertices_of_type(topology, "tubulin", exact_match=True)

            n_motors = len(vs_motor)
            n_mito = len(vs_mito)
            n_tubulin = len(vs_tubulin)

            # Case 1:  [Y] Tubulin, [N] Mitochondria, [n] Motor --> Found microtubule --> Reset to base topology
            if n_mito == n_motors == 0 and n_tubulin > 0:

                print(f"(Add_AT_3) Case 1: Found microtubule") if self._debug else None
                recipe.change_topology_type("Microtubule")

            # Case 2: [N] Tubulin, [N] Mitochondria, [Y] Motor --> Lone Motor Particle --> Flag for removal
            elif n_mito == n_tubulin == 0 and n_motors == 1:
                # print(f"(Add_AT_3) Case 2: Lone Motor Particle")
                v_motor = vs_motor[0]
                v_motor_uid = ReaddyUtils.vertex_to_uid(topology, v_motor)
                engine.delete_iterator(v_motor_uid)
                recipe.change_particle_type(v_motor, "motor#decay")
                recipe.separate_vertex(v_motor)


            # Case 3: [N] Tubulin, [Y] Mitochondria, [Y] Motor --> Mitochondria#AT Complex
            elif n_mito > 0 and n_motors > 0 and n_tubulin == 0:
                # TODO: Add check for missing motors here!
                if n_motors > 1:
                    vs_mito_at = ReaddyUtils.get_vertices_of_type(topology, vertex_type="mitochondria#terminal#AT", exact_match=True)
                    vs_mito_IM2 = ReaddyUtils.get_vertices_of_type(topology, vertex_type="mitochondria#AT-IM2", exact_match=True)
                    n_mito_terminal = len(vs_mito_at + vs_mito_IM2)
                    if n_mito_terminal > n_motors:
                        for v_mito_at in vs_mito_at:
                            v_motor_at = ReaddyUtils.get_neighbor_of_type(topology, v_mito_at, "motor#", exact_match=False)
                            if v_motor_at is None:
                                v_mito_at_uid = ReaddyUtils.vertex_to_uid(topology, v_mito_at)
                                engine.delete_buffer(v_mito_at_uid)
                                recipe.change_particle_type(v_mito_at, "mitochondria#terminal")

                # Get the motor and cargo vertices
                v_mito_IM2 = ReaddyUtils.get_vertex_of_type(topology, "mitochondria#AT-IM2", exact_match=True)
                v_mito_IM2_uid = ReaddyUtils.vertex_to_uid(topology, v_mito_IM2)
                v_motor_IM = ReaddyUtils.get_neighbor_of_type(topology, v_mito_IM2, "motor#IM", exact_match=True)

                # TODO: Not sure if this is necessary
                # if v_motor_IM is None:
                #     print("Add_AT_R3: No valid motor found. ")
                #     recipe.change_particle_type(v_mito_IM2, "mitochondria#terminal")
                #     recipe.change_topology_type("Mitochondria")
                #     return recipe

                # Set up motor info
                v_motor_IM_uid = ReaddyUtils.vertex_to_uid(topology, v_motor_IM)
                v_motor_IM_pos = ReaddyUtils.get_vertex_position(topology, v_motor_IM)

                # TODO: Implement motor binding rates here if necessary (remove oriented binding?)
                # Could have some funny behavior here b/c of pre-setting motor type?
                if n_motors == 1:
                    new_motor_type = engine.get_motor_type(v_mito_IM2_uid)
                    # TODO: ^^^Not sure if oriented binding is necessary for decentralized approach
                else:
                    new_motor_type = np.random.choice(["motor#kinesin", "motor#dynein"], p=[0.5, 0.5])

                # Case 3a: Valid motor type (returns None if no valid motor type is found)
                if new_motor_type is not None:
                    # p_active = self.parameters[motor_type]["p_active"]
                    # motor_state = np.random.choice(["active", "inactive"], p=[p_active, 1. - p_active])
                    new_topology_type = f"Mitochondria#AT"
                    new_mito_type = f"mitochondria#terminal#AT"

                    # TODO: Set up decentralized iterator creation in engine
                    engine.create_iterator(cargo_uid=v_mito_IM2_uid,
                                           motor_uid=v_motor_IM_uid,
                                           motor_position=v_motor_IM_pos,
                                           motor_type=new_motor_type)

                # Case 3b: No valid motor type
                else:
                    new_topology_type = f"Mitochondria"
                    new_mito_type = f"mitochondria#terminal"
                    new_motor_type = "motor#decay"
                    recipe.separate_vertex(v_motor_IM)
                    engine.delete_buffer(v_mito_IM2_uid)

                print(f"(Add_AT_3) Case 3 New Motor Type: {new_motor_type}") if self._debug else None
                print(f"(Add_AT_3) Case 3 New Topology Type: {new_topology_type}") if self._debug else None
                print(f"(Add_AT_3) Case 3 New Mitochondria Type: {new_mito_type}") if self._debug else None

                # Update the topology and particle types accordingly
                recipe.change_particle_type(v_mito_IM2, new_mito_type)
                recipe.change_particle_type(v_motor_IM, new_motor_type)
                recipe.change_topology_type(new_topology_type)

            # Case 4: Edge case
            else:
                raise Exception(f"(Add_AT_3) Case 4: Edge Case - This shouldn't have happened..."
                                f"\n{ReaddyUtils.topology_to_string(topology)}")
            return recipe
        return reaction_function_add_active_transport3

    def reaction_function_step(self, topology):
        print(f"(AT_step) Topology: {ReaddyUtils.topology_to_string(topology)}") if self._debug else None
        recipe = readdy.StructuralReactionRecipe(topology)
        vs_motor = ReaddyUtils.get_vertices_of_type(topology, vertex_type="motor#", exact_match=False)
        vs_mito = ReaddyUtils.get_vertices_of_type(topology, vertex_type="mitochondria#", exact_match=False)
        vs_mito_at = ReaddyUtils.get_vertices_of_type(topology, vertex_type="mitochondria#terminal#AT", exact_match=True)
        # vs_mito_im = ReaddyUtils.get_vertices_of_type(topology, vertex_type="mitochondria#AT-IM2", exact_match=False)
        n_motors = len(vs_motor)
        n_mito_at = len(vs_mito_at)
        n_mito = len(vs_mito)
        print(f"(AT_step) n_motors: {n_motors}") if self._debug else None
        print(f"(AT_step) n_mito_at: {n_mito_at}") if self._debug else None
        print(f"(AT_step) n_mito: {n_mito}") if self._debug else None

        # Case 1: Mitochondria#AT Complex
        if n_mito_at > 0 and n_motors > 0:
            print(f"(AT_step) Case 1: Mitochondria#AT Complex") if self._debug else None
            print(f"(AT_step) Topology: {ReaddyUtils.topology_to_string(topology)}") if self._debug else None
            for v_mito_at in vs_mito_at:
                v_mito_at_mito_neighbors = ReaddyUtils.get_neighbors_of_type(topology, v_mito_at, "mitochondria#", exact_match=False)
                v_mito_at_motor_neighbors = ReaddyUtils.get_neighbors_of_type(topology, v_mito_at, "motor#", exact_match=False)

                n_mito_neighbors = len(v_mito_at_mito_neighbors)
                n_motor_neighbors = len(v_mito_at_motor_neighbors)
                # Check for missing motor and reset particle type if so.
                if n_motor_neighbors == 0 and n_mito_neighbors >= 0:
                    # print(f"(AT Step) Found mito without motor.") if self._debug else None
                    v_mito_at_uid = ReaddyUtils.vertex_to_uid(topology, v_mito_at)
                    engine.delete_buffer(v_mito_at_uid)
                    recipe.change_particle_type(v_mito_at, "mitochondria#terminal")
                elif n_motor_neighbors == 1:
                    v_motor = v_mito_at_motor_neighbors[0]
                    v_motor_uid = ReaddyUtils.vertex_to_uid(topology, v_motor)
                    new_position = engine.step(v_motor_uid)

                    # Position update
                    if new_position is not None:
                        recipe.change_particle_position(v_motor, new_position)

                    # No valid position found
                    else:
                        if engine.end_behavior == "remove_all":
                            recipe = self.remove_all_motors(recipe, topology, vs_motor, vs_mito_at)
                            n_motors = 0
                        elif engine.end_behavior == "invert_all":
                            recipe = self.invert_motors(recipe, topology)
                        elif engine.end_behavior == "invert":
                            recipe = self.remove_motor(recipe, topology, v_motor, v_mito_at)
                            recipe = self.invert_motors(recipe, topology)
                            n_motors -= 1
                        else: # Default behavior
                            if engine.end_behavior == "remove":
                                recipe = self.remove_motor(recipe, topology, v_motor, v_mito_at)
                                n_motors -= 1

            # Reset to base topology if all motors have been removed
            if n_motors == 0:
                recipe.change_topology_type("Mitochondria")

        # Case 2: Lone Motor Particle (from energy-dependent bond breaking)
        elif n_motors == 1 and n_mito_at == 0:
            print(f"(AT_step) Case 2: Lone Motor Particle") if self._debug else None
            print(f"Topology: {ReaddyUtils.topology_to_string(topology)}") if self._debug else None
            v_motor = vs_motor[0]
            v_motor_uid = ReaddyUtils.vertex_to_uid(topology, v_motor)
            engine.delete_iterator(v_motor_uid)
            engine._counts["lone_motor"] += 1
            recipe.change_particle_type(v_motor, "motor#decay")

        # Case 2: Mitochondria#AT Complex with no motors
        elif n_mito_at > 0 and n_motors == 0:
            print(f"(AT_step) Case 3: Mitochondria#AT Complex with no motors") if self._debug else None
            print(f"Topology: {ReaddyUtils.topology_to_string(topology)}") if self._debug else None
            # Reset particle types
            for v_mito_at in vs_mito_at:
                v_mito_at_uid = ReaddyUtils.vertex_to_uid(topology, v_mito_at)
                engine.delete_buffer(v_mito_at_uid)
                recipe.change_particle_type(v_mito_at, "mitochondria#terminal")

            # Reset topology to base type
            recipe.change_topology_type("Mitochondria")

        # Case 3: Edge case
        else:
            raise Exception(f"(reaction_function_step) Case 4: Edge Case - This shouldn't have happened..."
                            f"\n{ReaddyUtils.topology_to_string(topology)}")
        return recipe

    def remove_motor(self, recipe, topology, v_motor, v_cargo):
        """ Removes a motor from the topology. """
        print(f"(Remove Motor) v_motor: {v_motor}") if self._debug else None
        v_motor_uid = ReaddyUtils.vertex_to_uid(topology, v_motor)
        v_cargo_uid = ReaddyUtils.vertex_to_uid(topology, v_cargo)
        engine.delete_iterator(v_motor_uid)
        engine.delete_buffer(v_cargo_uid)
        recipe.change_particle_type(v_motor, "motor#decay")
        recipe.separate_vertex(v_motor)
        recipe.change_particle_type(v_cargo, "mitochondria#terminal")
        return recipe

    def remove_all_motors(self, recipe, topology, vs_motors, vs_cargo):
        """ Removes all motors from the topology. """
        print(f"(Remove All Motors) Called") if self._debug else None
        for v_motor, v_cargo in zip(vs_motors, vs_cargo):
            recipe = self.remove_motor(recipe, topology, v_motor, v_cargo)
        recipe.change_topology_type("Mitochondria")
        return recipe

    def invert_motors(self, recipe, topology):
        """ Inverts all motors types in the topology, e.g. kinesin -> dynein & vice versa. """
        raise NotImplemented
