import readdy
import itertools
import numpy as np
from typing import Optional
from src.core.readdy_utils import ReaddyUtils
from src.reactions.active_transport_engine import ActiveTransportEngine

ut = readdy.units

def set_parameters(p):
    global parameters
    parameters.update(p)

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

class MitochondrialDynamics:
    """ Mitochondrial Dynamics Reaction Network """

    REACTIVE_TOPOLOGIES = {
        "fusion1": ["Mitochondria",
                    "Mitochondria#AT"],
        "fusion2": ["Mitochondria#Fusion-IM1",
                    "Mitochondria#AT#Fusion-IM1"],
        "fission1": ["Mitochondria",
                     "Mitochondria#AT"],
        "fission2": ["Mitochondria#Fission-IM1",
                     "Mitochondria#AT#Fission-IM1"],
        "fission3": ["Mitochondria#Fission-IM2",
                     "Mitochondria#AT#Fission-IM2"],
    }

    REACTIVE_PARTICLES = {
        "fusion1": ["mitochondria#internal",
                    "mitochondria#terminal",
                    "mitochondria#terminal#AT"],
    }

    def __init__(self,
                 parameters: dict = None,
                 engine: Optional[ActiveTransportEngine] = None,
                 **kwargs):
        assert parameters is not None, "Parameters must be provided"

        time_unit = kwargs.get('time_unit', 's')
        length_unit = kwargs.get('length_unit', 'um')
        self.parameters = convert_units(parameters, time_unit, length_unit)

        if engine:
            self._with_active_transport = True
            set_active_transport_engine(engine)
        else:
            self._with_active_transport = False

        self.registry = self.REACTIVE_TOPOLOGIES
        self._debug = False

    def _get_reactive_particle_pairs(self):
        """ Returns the reactive particle types """
        ptype_pairs = list(itertools.product(self.REACTIVE_PARTICLES["fusion1"], repeat=2))
        # TODO: There's likely a better place for this conditional parameter but it's here for now
        if not self.parameters["enable_ss_fusion"]:
            ptype_pairs = [pair for pair in ptype_pairs if pair != ("mitochondria#internal", "mitochondria#internal")]
        return ptype_pairs

    def register(self, system: readdy.ReactionDiffusionSystem, *args, **kwargs):
        """ Register the reaction network for mitochondrial dynamics """
        if self._with_active_transport:
            self.add_fusion_reactions_with_at(system)
            self.add_fission_reactions_with_at(system)
        else:
            self.add_fusion_reactions_without_at(system)
            self.add_fission_reactions_without_at(system)

    def fission1_rate_function(self, topology):
        n = len(ReaddyUtils.get_vertices_of_type(topology, vertex_type="mitochondria#", exact_match=False))
        return self.parameters["mitochondria_fission_rate"] * n

    def add_fission_reactions_with_at(self, system: readdy.ReactionDiffusionSystem):
        """ Add mitochondrial fission reactions to the system """
        for reactive_topology in self.REACTIVE_TOPOLOGIES["fission1"]:
            reaction_name = f"Fission Reaction 1 ({reactive_topology})"
            system.topologies.add_structural_reaction(name=reaction_name,
                                                      topology_type=reactive_topology,
                                                      reaction_function=self.reaction_function_fission1_with_at,
                                                      rate_function=self.fission1_rate_function,
                                                      expect_connected=False)

        system.topologies.add_structural_reaction(name="Fission Reaction 2",
                                                  topology_type="Mitochondria#Fission-IM1",
                                                  reaction_function=self.reaction_function_fission2_with_at,
                                                  rate_function=lambda x: 1.e32,
                                                  expect_connected=False)

    # ORIGINAL
    # def reaction_function_fission1_with_at(self, topology):
    #     """ Fission Reaction 1/2 with Active Transport """
    #     print(f"(Fission 1+AT) Initial Topology: \n", ReaddyUtils.topology_to_string(topology)) if self._debug else None
    #     recipe = readdy.StructuralReactionRecipe(topology)
    #
    #     # Get the number of particles in the topology
    #     n_particles = len(topology.particles)
    #     if n_particles == 1:
    #         return recipe
    #
    #     vs_mito_int = ReaddyUtils.get_vertices_of_type(topology, vertex_type="mitochondria#internal", exact_match=True)
    #     if len(vs_mito_int) == 0:
    #         return recipe
    #     else:
    #         v_mito = vs_mito_int[0]
    #         vs_mito_neighbors = ReaddyUtils.get_neighbors_of_type(topology, v_mito, vertex_type="mitochondria#", exact_match=False)
    #         if len(vs_mito_neighbors) == 0:
    #             return recipe
    #
    #         v_mito_neighbor = np.random.choice(vs_mito_neighbors)
    #         recipe.change_particle_type(v_mito, "mitochondria#Fission-IM1")
    #         recipe.change_particle_type(v_mito_neighbor, "mitochondria#Fission-IM1")
    #         recipe.change_topology_type("Mitochondria#Fission-IM1")
    #         recipe.remove_edge(v_mito, v_mito_neighbor)
    #     return recipe

    # MODIFIED - Added fixed vertex selection
    def reaction_function_fission1_with_at(self, topology):
        """ Fission Reaction 1/2 with Active Transport """
        print(f"(Fission 1+AT) Initial Topology: \n", ReaddyUtils.topology_to_string(topology)) if self._debug else None
        recipe = readdy.StructuralReactionRecipe(topology)

        # Filtering detached motors from the topology
        # TODO: Not sure if this is required...
        n_particles = len(topology.particles)
        if n_particles == 1:
            return recipe

        vs_mito = ReaddyUtils.get_vertices_of_type(topology, vertex_type="mitochondria#", exact_match=False)
        if len(vs_mito) == 1:
            return recipe
        else:
            v1_split = np.random.choice(vs_mito)
            vs_mito_neighbors = ReaddyUtils.get_neighbors_of_type(topology, v1_split, vertex_type="mitochondria#", exact_match=False)
            if len(vs_mito_neighbors) == 0:
                return recipe

            v2_split = np.random.choice(vs_mito_neighbors)
            recipe.change_particle_type(v1_split, "mitochondria#Fission-IM1")
            recipe.change_particle_type(v2_split, "mitochondria#Fission-IM1")
            recipe.change_topology_type("Mitochondria#Fission-IM1")
            recipe.remove_edge(v1_split, v2_split)
        return recipe

    def reaction_function_fission2_with_at(self, topology):
        """ Fission Reaction 2/2 with Active Transport """
        print(f"(Fission 2+AT) Initial Topology: \n", ReaddyUtils.topology_to_string(topology)) if self._debug else None
        recipe = readdy.StructuralReactionRecipe(topology)
        vs_mito_im = ReaddyUtils.get_vertices_of_type(topology,"mitochondria#Fission-IM1", exact_match=True)
        vs_motors = ReaddyUtils.get_vertices_of_type(topology, "motor#", exact_match=False)
        for v_mito_im in vs_mito_im:
            vs_mito_neighbors = ReaddyUtils.get_neighbors_of_type(topology, v_mito_im, vertex_type="mitochondria#", exact_match=False)
            vs_motor_neighbors = ReaddyUtils.get_neighbors_of_type(topology, v_mito_im, vertex_type="motor#", exact_match=False)

            # Case 1: [0] Motors; [>1] Mito Neighbors --> Internal, No AT
            if len(vs_motor_neighbors) == 0 and len(vs_mito_neighbors) > 1:
                recipe.change_particle_type(v_mito_im, "mitochondria#internal")

            # Case 2: [1] Motor; [1] Mito --> Terminal + AT
            elif len(vs_motor_neighbors) == 1 and len(vs_mito_neighbors) == 1:
                recipe.change_particle_type(v_mito_im, "mitochondria#terminal#AT")

            # Case 3: [1] Motor; [0] Mito --> Terminal, + AT
            elif len(vs_motor_neighbors) == 1 and len(vs_mito_neighbors) == 0:
                recipe.change_particle_type(v_mito_im, "mitochondria#terminal#AT")

            # Case 4: [0] Motors; [1] Mito --> Terminal, No AT
            elif len(vs_motor_neighbors) == 0 and len(vs_mito_neighbors) == 1:
                recipe.change_particle_type(v_mito_im, "mitochondria#terminal")

            # Case 5: [0] Motors; [0] Mito --> Terminal, No AT
            elif len(vs_motor_neighbors) == 0 and len(vs_mito_neighbors) == 0:
                recipe.change_particle_type(v_mito_im, "mitochondria#terminal")
            else:
                raise Exception("(Fission 2+AT) Unexpected case encountered. This shouldn't have happened...")

        # Reset to topology with AT if there are motors
        if len(vs_motors) > 0:
            recipe.change_topology_type("Mitochondria#AT")
        else:
            # Reset to base topology type
            recipe.change_topology_type("Mitochondria")
        return recipe

    def add_fusion_reactions_with_at(self, system: readdy.ReactionDiffusionSystem):
        """ Adds fusion reactions to the system """
        reactive_topology_pairs = list(itertools.product(self.REACTIVE_TOPOLOGIES["fusion1"], repeat=2))
        reactive_particle_pairs = self._get_reactive_particle_pairs()
        rxn_name_template = "Mitochondria Fusion Reaction 1 {T1}({P1}) + {T2}({P2})"
        rxn_string_descriptor = "{RXN_NAME}: {T1}({P1}) + {T2}({P2}) -> {T3}({P3_1}--{P3_2})"

        i = 0
        for top_pair in reactive_topology_pairs:
            for particle_pair in reactive_particle_pairs:
                p1, p2 = particle_pair
                t1, t2 = top_pair
                t3 = "Mitochondria#Fusion-IM1"
                p3_1 = "mitochondria#Fusion-IM1"
                p3_2 = "mitochondria#Fusion-IM1"
                rxn_name = rxn_name_template.format(T1=t1, T2=t2, P1=p1, P2=p2)
                rxn_descriptor = rxn_string_descriptor.format(
                    RXN_NAME=rxn_name, T1=t1, T2=t2, T3=t3, P1=p1, P2=p2, P3_1=p3_1, P3_2=p3_2
                )
                system.topologies.add_spatial_reaction(rxn_descriptor,
                                                       rate=self.parameters["mitochondria_fusion_rate"],
                                                       radius=self.parameters["mitochondria_fusion_radius"])
                i += 1

        # Add second fusion reaction (Clean up the topology)
        reaction_name = f"Mitochondria Fusion Reaction 2"
        reaction_function = self.reaction_function_fusion2_with_at
        system.topologies.add_structural_reaction(name=reaction_name,
                                                  topology_type="Mitochondria#Fusion-IM1",
                                                  reaction_function=reaction_function,
                                                  rate_function=lambda x: 1.e32,
                                                  expect_connected=False)


    def reaction_function_fusion2_with_at(self, topology):
        """ 2nd reaction in fusion mechanism """
        print(f"(Fusion 2) Input Topology: \n", ReaddyUtils.topology_to_string(topology)) if self._debug else None
        recipe = readdy.StructuralReactionRecipe(topology)
        if len(topology.particles) == 1:
            print(f"(Fusion 2) Single Particle Topology: {ReaddyUtils.topology_to_string(topology)}") if self._debug else None
            try:
                v_motor = ReaddyUtils.get_vertices_of_type(topology, vertex_type="motor#", exact_match=False)[0]
                if v_motor:
                    recipe.change_particle_type(v_motor, "motor#decay")
                    recipe.separate_vertex(v_motor)
                    return recipe
            except IndexError:
                print(f"(Fusion 2+AT) This shouldn't have happened...")
        else:
            vs_mito_fusion = ReaddyUtils.get_vertices_of_type(topology,
                                                              vertex_type="mitochondria#Fusion-IM1",
                                                              exact_match=True)

            vs_motors_total = ReaddyUtils.get_vertices_of_type(topology,
                                                               vertex_type="motor#",
                                                               exact_match=False)
            n_motors = len(vs_motors_total)
            print(f"(Fusion 2+AT) Initial Number of Motors: {n_motors}") if self._debug else None
            for v in vs_mito_fusion:
                vs_mito_neighbors = ReaddyUtils.get_neighbors_of_type(topology, v, "mitochondria#", exact_match=False)
                vs_motor_neighbors = ReaddyUtils.get_neighbors_of_type(topology, v, "motor#", exact_match=False)

                # TODO: Edge case; Retain motor single mito particle with AT fuses with another fragment?
                # Remove motors from the new particles
                if len(vs_motor_neighbors) != 0:
                    for v_motor in vs_motor_neighbors:
                        print(f"(Fusion 2+AT) Removing Motor: {ReaddyUtils.vertex_to_type(topology, v_motor)}") if self._debug else None
                        recipe.change_particle_type(v_motor, "motor#decay")
                        recipe.separate_vertex(v_motor)
                        n_motors -= 1

                degree = len(vs_mito_neighbors)
                # Edge case w/ single mito particle fusion (it's still terminal after reacting)
                if degree == 1:
                    recipe.change_particle_type(v, "mitochondria#terminal")
                else:
                    recipe.change_particle_type(v, "mitochondria#internal")

            if n_motors > 0:
                recipe.change_topology_type("Mitochondria#AT")
            else:
                recipe.change_topology_type("Mitochondria")
            return recipe

    def add_fission_reactions_without_at(self, system: readdy.ReactionDiffusionSystem):
        """ Add mitochondrial fission reactions to the system """
        for reactive_topology in self.REACTIVE_TOPOLOGIES["fission1"]:
            reaction_name = f"Fission Reaction 1 ({reactive_topology})"
            system.topologies.add_structural_reaction(name=reaction_name,
                                                      topology_type=reactive_topology,
                                                      reaction_function=self.reaction_function_fission1_without_at,
                                                      rate_function=self.fission1_rate_function,
                                                      expect_connected=False)

    def reaction_function_fission1_without_at(self, topology):
        """ Fission Reaction 1/1 without Active Transport """
        print(f"(Fission 1-AT) Initial Topology: \n", ReaddyUtils.topology_to_string(topology)) if self._debug else None
        recipe = readdy.StructuralReactionRecipe(topology)

        # Get the number of particles in the topology
        n_particles = len(topology.particles)
        if n_particles == 1:
            return recipe

        vs_mito = ReaddyUtils.get_vertices_of_type(topology, vertex_type="mitochondria#", exact_match=False)
        if len(vs_mito) > 1:
            v1_split = np.random.choice(vs_mito)
            vs_neighbors = ReaddyUtils.get_neighbors_of_type(topology, v1_split, vertex_type="mitochondria#",
                                                             exact_match=False)
            if len(vs_neighbors) == 1:
                v2_split = vs_neighbors[0]
            else:
                v2_split = np.random.choice(vs_neighbors)

            for v in [v1_split, v2_split]:
                n_neighbors = len(ReaddyUtils.get_neighbors_of_type(topology, v, vertex_type="mitochondria#", exact_match=False))
                if (n_neighbors - 1) >= 1:
                    recipe.change_particle_type(v, "mitochondria#terminal")
                else:
                    recipe.change_particle_type(v, "mitochondria#internal")
            recipe.remove_edge(v1_split, v2_split)
        return recipe

    @staticmethod
    def reaction_function_fusion2_without_at(topology):
        """ 2nd reaction in fusion mechanism """
        recipe = readdy.StructuralReactionRecipe(topology)
        vs_mito_fusion = ReaddyUtils.get_vertices_of_type(topology, vertex_type="Fusion-#IM1", exact_match=False)

        for v in vs_mito_fusion:
            n_neighbors = len(ReaddyUtils.get_neighbors_of_type(topology, v, vertex_type="mitochondria#", exact_match=False))
            if n_neighbors == 1:
                recipe.change_particle_type(v, "mitochondria#terminal")
            else:
                recipe.change_particle_type(v, "mitochondria#internal")
        recipe.change_topology_type("Mitochondria")
        return recipe

    def add_fusion_reactions_without_at(self, system: readdy.ReactionDiffusionSystem):
        """ Adds fusion reactions to the system (without active transport) """
        reactive_particle_pairs = self._get_reactive_particle_pairs()

        rxn_name_template = "Mitochondria Fusion Reaction 1 {T12}({P1}) + {T12}({P2})"
        rxn_string_descriptor = "{RXN_NAME}: {T12}({P1}) + {T12}({P2}) -> {T3}({P3}--{P3})"

        for particle_pair in reactive_particle_pairs:
            p1, p2 = particle_pair
            t12 = "Mitochondria"
            t3 = "Mitochondria#Fusion-IM1"
            p3 = "mitochondria#Fusion-IM1"
            rxn_name = rxn_name_template.format(T12=t12, P1=p1, P2=p2)
            rxn_descriptor = rxn_string_descriptor.format(RXN_NAME=rxn_name, T12=t12, T3=t3, P1=p1, P2=p2, P3=p3)
            system.topologies.add_spatial_reaction(rxn_descriptor,
                                                   rate=self.parameters["mitochondria_fusion_rate"],
                                                   radius=self.parameters["mitochondria_fusion_radius"])

        # Add second fusion reaction (Clean up the topology)
        reaction_name = f"Mitochondria Fusion Reaction 2"
        reaction_function = self.reaction_function_fusion2_without_at
        system.topologies.add_structural_reaction(name=reaction_name,
                                                  topology_type="Mitochondria#Fusion-IM1",
                                                  reaction_function=reaction_function,
                                                  rate_function=lambda x: 1.e32,
                                                  expect_connected=False)
