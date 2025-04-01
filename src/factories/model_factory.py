import os
import readdy
import numpy as np
import json
import pickle
import igraph as ig
from itertools import product
from src.core import DataLoader
from src.factories import *
from src.reactions import ActiveTransportEngine, ActiveTransportReactions, MitochondrialDynamics
from readdy.api.experimental.action_factory import BreakConfig, ReactionConfig
from src.analysis import Velocity, TopologyGraphs
from src.visualization import SimulariumConverter

# Initialize units for readdy
ut = readdy.units
np.random.seed(42)


# TODO: Finish up routine for measuring reactions and then run the optimization

class ModelFactory:
    def __init__(self, *args, **kwargs):
        self.parameters = {
            "motor#kinesin": {
                "binding_distance": 0.2 * ut.um,  # ARBITRARY + CURRENTLY SHARED
                "binding_rate": 3.75 / ut.s,  # D'Souza [LIT]; Average rates from Kinesin and Dynein (CURRENTLY SHARED)
                # "binding_rate": 5. / ut.s,  # 1st bound or 2nd Kinesin; 1.25 /s if existing Dynein (NOT IMPLEMENTED)
                "unbinding_rate": 1.27 / ut.s,  # D'Souza [EXP];
                # "activation_rate": 0.56 / ut.s, # D'Souza [MY CALC]; Single-Motor Vesicle t_pause=1.8 -> 1/t_pause = 0.56
                "activation_rate": 0.06735,  # <<< CALIBRATED >>>; --> p_active = p_activate = 4.051e-4 for dt=0.005
                # "p_activate": 3.367e-4,  # <<< CALIBRATED >>>; OVERRIDES activation_rate
                # "p_activate": 0.0025,
                "p_activate": 0.0006,
                # "p_activate": 0.0008,
                "deactivation_rate": 0.31 / ut.s,  # D'Souza [EXP]; Single-Motor Vesicle Pause Frequency
                "velocity": 0.78 * ut.um / ut.s,  # Cai [EXP]
                # "velocity": 0.1 * ut.um / ut.s,  # Cai [EXP]
                "step_length": 0.05,  # um (CURRENTLY SHARED)
                "polarity": 1,
                # "p_active": 0.8, # From D'Souza, et. Al.
                # "p_active":  3.367e-4,  # <<< CALIBRATED >>>
                # "p_active": 0.0025,
                "p_active": 0.0006,
                # "p_active": 0.0008,
                "path_selection": "random",
                "radius": 0.05,  # um
            },
            "motor#dynein": {
                "binding_distance": 0.2 * ut.um,  # ARBITRARY + CURRENTLY SHARED
                "binding_rate": 3.75 / ut.s,  # NOT IMPLEMENTED; RATE SHARED WITH KINESIN; VALUE IS AVERAGE!
                # "binding_rate": 2.5 / ut.s,  # D'Souza [LIT];
                "unbinding_rate": 0.44 / ut.s,  # D'Souza [EXP];
                # "activation_rate": 0.44 / ut.s, # D'Souza [EXP] ; Single-Motor Vesicle t_pause=2.3 -> 1/t_pause = 0.44
                "activation_rate": 0.06735,  # <<< CALIBRATED >>>; --> p_active = p_activate = 4.051e-4 for dt=0.005
                # "p_activate": 3.367e-4,  # <<< CALIBRATED >>>; OVERRIDES activation_rate
                # "p_activate": 0.0008,
                "p_activate": 0.0006,
                "deactivation_rate": 0.25 / ut.s,  # D'Souza [EXP]; Single-Motor Vesicle Pause Frequency
                "velocity": 1.2 * ut.um / ut.s,  # Fenton [EXP]; Mean +-directed velocity **MITOCHONDRIA**
                # "velocity": 0.1 * ut.um / ut.s,
                "step_length": 0.05,  # um
                "polarity": -1,
                # "p_active": 0.9, # D'Souza[EXP]
                # "p_active": 3.367e-4,  # <<< CALIBRATED >>>
                # "p_active": 0.0025,
                "p_active": 0.0006,
                # "p_active": 0.0008,
                "path_selection": "random",
                "radius": 0.05,  # um
            },
            "mitochondria": {
                "radius": 0.15,  # um
                # "diffusion_constant": 0.0032507691082437193 * ut.um ** 2 / ut.s, #Calibrated to D_mean from Nocodazole 60min
                # "diffusion_constant": 0.0088 * ut.um ** 2 / ut.s,
                "diffusion_constant": 0.0048 * ut.um ** 2 / ut.s,
                "kc_repulsion": 0.005 * ut.kilojoule / ut.mol / ut.nm ** 2,
                "d_repulsion": 0.3,  # um
                "kc_bond": 0.01 * ut.kilojoule / ut.mol / ut.nm ** 2,
                "eq_bond": 0.3,
                "kc_angle": 9.12 * ut.kilojoule / ut.mol / ut.rad ** 2,
                "eq_angle": np.pi,

                # "mitochondria_fission_rate": 0.002963 / ut.s, # normed lambda off from rate for Noco60
                # "mitochondria_fusion_rate": 0.000569748 / ut.s,  # Calibrated to R_rxn = 0.6
                # "mitochondria_fusion_rate": 0.001098 / ut.s,  # Estimated lambda_on from rate for Noco60

                # "mitochondria_fission_rate": 0.00043 / ut.s, # lambda_off for Noco60; Normed wrt n_nodes per frame (first 20 tracked frames)
                # "mitochondria_fusion_rate": 0.0008934116258985479 / ut.s,  # tuned lambda_on for Noco60 c1 relative to normed lambda_off (above)
                "mitochondria_fission_rate": 0.00044 / ut.s,
                # "mitochondria_fusion_rate": 0.0008934116258985479 / ut.s,
                # "mitochondria_fusion_rate": 0.0225 / ut.s, # Tuned w/ side-side reactions enabled
                # "mitochondria_fusion_rate": 0.08 / ut.s,  # Tuned w/ side-side reactions disabled
                # "mitochondria_fusion_rate": 0.1 / ut.s,  # Testing
                "mitochondria_fusion_rate": 0.12 / ut.s,  # Testing
                "mitochondria_fusion_radius": 0.3,  # um
                "enable_ss_fusion": False,
            },
            "tubulin": {
                "radius": 0.05,  # um
                "diffusion_constant": 0.010078746846453792 * ut.um ** 2 / ut.s,  # dt=5.e-3 s (PRODUCTION)
                "kc_repulsion": 0.005 * ut.kilojoule / ut.mol / ut.nm ** 2,
                "d_repulsion": 0.05,  # um
                "kc_bond": 0.01 * ut.kilojoule / ut.mol / ut.nm ** 2,
                "eq_bond": 0.2 * ut.um,
                "kc_angle": 50. * ut.kilojoule / ut.mol / ut.rad ** 2,
                "eq_angle": np.pi,
            },
            "nucleus": {
                "radius": 0.25
            },
            "membrane": {
                "radius": 0.25
            },
            "transport_parameters": {
                "motor_radius": 0.05,  # um
                "rc_reverse_direction": 0.,  # /s
                "kc_bond": 0.01 * ut.kilojoule / ut.mol / ut.nm ** 2,
                "bond_energy_threshold": 6.5e2,  # Threshold Energy for breaking mito-motor bond
                "rc_cleave_bond": 1.e32,  # Rate constant for breaking mito-motor bond
            },
            "run_parameters": {
                "flags": {
                    "plot_results": True,
                    "save_trajectory": True,
                    "save_results": True,
                    "enable_active_transport": True,
                    "single_motor": False,
                    "count_steps": True,
                    "count_events": True,
                    "enable_mitochondrial_dynamics": True,
                },
            }
        }

        self.parameters["run_parameters"]["flags"]["enable_active_transport"] = kwargs.get("enable_active_transport",
                                                                                           True)
        self.parameters["run_parameters"]["flags"]["enable_mitochondrial_dynamics"] = kwargs.get(
            "enable_mitochondrial_dynamics", True)

        cell_idx = kwargs.get("cell_idx", None)
        condition = kwargs.get("condition", None)

        datadir = "data/mitosim_dataset_v6/{}/cell_{}".format(condition, cell_idx)
        print(f"Loading data from {datadir}")
        dl = DataLoader(datadir)

        self._temperature = 310.15 * ut.kelvin
        self._timestep = 5.e-3 * ut.s
        self._skin = 1.0

        self._n_steps = kwargs.get("n_steps", int(10))
        self._stride = kwargs.get("stride", 1)
        self._kernel = kwargs.get("kernel", "CPU")
        self._reaction_handler = "Gillespie"

        self._run_index = kwargs.get("run_index", None)
        self._outdir = kwargs.get("outdir",
                                  f"../../data/processed_data/models/")
        self._outfile = f"{condition}_c{cell_idx}"

        p_active_activate = kwargs.get("p_active_activate", None)
        if p_active_activate is not None:
            self.parameters["motor#kinesin"]["p_active"] = p_active_activate
            self.parameters["motor#dynein"]["p_active"] = p_active_activate
            self.parameters["motor#kinesin"]["p_activate"] = p_active_activate
            self.parameters["motor#dynein"]["p_activate"] = p_active_activate

        self._replace_trials = True

        print(f"Output directory: {self._outdir}")
        print(f"Output file: {self._outfile}")

        # Check if the output directory exists
        if not os.path.exists(self._outdir):
            os.makedirs(self._outdir)
        self.simbox_dims = None

        # Run parameters
        self._membrane_fraction = kwargs.get("membrane_fraction", None)
        # self._box_size_multiplier = kwargs.get("box_size_multiplier", None)
        self._plot_results = self.parameters["run_parameters"]["flags"]["plot_results"]
        self._save_trajectory = self.parameters["run_parameters"]["flags"]["save_trajectory"]
        self._save_results = self.parameters["run_parameters"]["flags"]["save_results"]
        self._enable_active_transport = self.parameters["run_parameters"]["flags"]["enable_active_transport"]
        self._single_motor = self.parameters["run_parameters"]["flags"]["single_motor"]
        self._count_steps = self.parameters["run_parameters"]["flags"]["count_steps"]
        self._count_events = self.parameters["run_parameters"]["flags"]["count_events"]
        self._enable_mitochondrial_dynamics = self.parameters["run_parameters"]["flags"][
            "enable_mitochondrial_dynamics"]

        # Private Vars
        self._system = None
        self._simulation = None
        self._engine = None
        self._trajectory = None
        self._break_config = None
        self._reaction_config = None
        self._motor_types = [key for key in self.parameters.keys() if "motor" in key]

        if condition == "control":
            if cell_idx == 0:
                nuc_offset = np.array([0., 0.3, -0.3])  # Control C0
            if cell_idx == 1:
                nuc_offset = np.array([-0.1, 0.5, -0.65])  # Control C1 v5
                # nuc_offset = np.array([-0.1, 0.5, -0.5])  # Control C1 v6
            else:
                nuc_offset = np.array([0., 0., 0.])

        elif condition == "nocodazole_60min":
            if cell_idx == 3:
                nuc_offset = np.array([-0.15, -0.15, 0.5])
                # nuc_offset = np.array([0., 0., 0.5]) # Noco60 C3
            elif cell_idx == 1:
                nuc_offset = np.array([-0.6, -1.2, 0.3]) # Noco60 C1 Re-segmented 01/29/25
            else:
                nuc_offset = np.array([0., 0., 0.])
        else:
            nuc_offset = np.array([0., 0., 0.])

        self._membrane = MembraneFactory(dl).run()
        self._nucleus = NucleusFactory(dl).run()
        self._mitochondria = MitochondriaFactory(dl).run()

        if self._enable_active_transport:
            self._microtubules = MicrotubulesFactory(dl).run(n_downsample=2)

        self._nucleus.data += nuc_offset
        centering_vector = -1 * (np.max(self._membrane.data, axis=0) - np.min(self._membrane.data, axis=0)) / 2

        self._membrane.center_model(centering_vector)
        self._nucleus.center_model(centering_vector)
        self._mitochondria.center_model(centering_vector)
        if self._enable_active_transport:
            self._microtubules.center_model(centering_vector)

        max_coord = np.max(self._membrane.data, axis=0)
        min_coord = np.min(self._membrane.data, axis=0)
        self.simbox_dims = max_coord - min_coord


        if self._membrane_fraction:
            # z_fraction = 0.7
            # z_threshold = min_coord[2] + z_fraction * (max_coord[2] - min_coord[2])
            # Filter out the membrane particles that are above
            # self._membrane.data = self._membrane.data[self._membrane.data[:, 2] < z_threshold]

            # y_fraction = 0.5
            # y_threshold = min_coord[1] + y_fraction * (max_coord[1] - min_coord[1])
            # self._membrane.data = self._membrane.data[self._membrane.data[:, 1] < y_threshold]

            # x_fraction = -0.55 # Control Cell 1

            if self._membrane_fraction < 0:
                fraction = abs(self._membrane_fraction)
                threshold = min_coord[0] + fraction * (max_coord[0] - min_coord[0])
                self._membrane.data = self._membrane.data[self._membrane.data[:, 0] > threshold]
            else:
                fraction = self._membrane_fraction
                threshold = min_coord[0] + fraction * (max_coord[0] - min_coord[0])
                self._membrane.data = self._membrane.data[self._membrane.data[:, 0] < threshold]

        # count = 0
        # for g in self._mitochondria.data:
        #     count += g.vcount()
        # print(f"Number of Mitochondria: {count}")

        # Create a dictionary to store all models
        self.models = {
            "nucleus": self._nucleus,
            "membrane": self._membrane,
            "mitochondria": self._mitochondria,
        }
        if self._enable_active_transport:
            self.models["microtubules"] = self._microtubules

    def save_models(self):
        # Save the models to a pickle file
        outfile = self._outdir + self._outfile + "_models.pkl"
        with open(outfile, "wb") as f:
            pickle.dump(self.models, f)
        print(f"Models saved to: {outfile}")

    @staticmethod
    def get_model_from_trajectory(traj_file):
        tg = TopologyGraphs(trajectory_file=traj_file, timestep=5.e-3 * ut.s)
        tg.run(particle_types=["mitochondria"], resort=False)
        gs_mito = tg.results["mitochondria"][-1]

        # Reformat the graph
        for g in gs_mito:
            g.vs["ttype"] = ["Mitochondria"] * g.vcount()
            for v in g.vs:
                n_mito_neighbors = 0
                for n in g.neighbors(v):
                    if "mitochondria" in g.vs[n]["type"]:
                        n_mito_neighbors += 1
                if n_mito_neighbors > 1:
                    v["ptype"] = "mitochondria#internal"
                else:
                    v["ptype"] = "mitochondria#terminal"
            del g.vs["type"]
        return gs_mito

    def add_topology_to_simulation(self, model):
        graphs = model.data
        topology_type = model.topology_type
        particle_type = model.particle_type

        for i, g in enumerate(graphs):
            coordinates = np.array(g.vs["coordinate"])
            edges = g.get_edgelist()
            for v in g.vs:
                degree = g.degree(v)
                if degree == 1:
                    v["ptype"] = f"{particle_type}#terminal"
                else:
                    v["ptype"] = f"{particle_type}#internal"
            ptypes = g.vs["ptype"]
            ttype = "Mitochondria"
            top = self._simulation.add_topology(ttype, ptypes, coordinates)
            for edge in edges:
                top.get_graph().add_edge(edge[0], edge[1])

    def add_particles_to_simulation(self, model):
        coordinates = np.array(model.data)
        particle_type = model.particle_type
        self._simulation.add_particles(particle_type, coordinates)

    def _save_parameters(self):
        def process_value(value):
            """Process a value, converting it to a JSON-serializable format."""
            if hasattr(value, "units"):
                # Handle pint quantities: extract magnitude and unit
                return {"value": value.magnitude, "unit": str(value.units)}
            elif isinstance(value, (int, float)):
                # Return plain numerical values as-is
                return {"value": value}
            elif value == np.pi:
                # Handle np.pi specifically
                return {"value": np.pi, "unit": "rad"}
            else:
                # Return any other values as-is (e.g., strings)
                return value

        def convert_parameters(params):
            """Recursively convert a dictionary to the desired JSON format."""
            result = {}
            for key, value in params.items():
                if isinstance(value, dict):
                    # Recursively process nested dictionaries
                    result[key] = convert_parameters(value)
                else:
                    result[key] = process_value(value)
            return result

        # Convert the parameters dictionary
        converted_parameters = convert_parameters(self.parameters)

        outfile = self._outdir + self._outfile + "_params.json"
        with open(outfile, "w") as f:
            json.dump(converted_parameters, f, indent=4)

    def _register_species(self):
        self._break_config = BreakConfig()
        self._reaction_config = ReactionConfig()

        # Topologies
        topology_flags = ["", "#AT", "#AT-IM1", "#AT-IM2", "#Fusion-IM1", "#Fusion-IM2", "#Fission-IM1", "#Fission-IM2"]
        topology_types = ["Microtubule", "Mitochondria"]
        mito_flags = ["", "#AT-IM1", "#AT-IM2", "#terminal#AT", "#terminal", "#internal", "#Fusion-IM1",
                      "#Fusion-IM2", "#Fission-IM1", "#Fission-IM2"]
        motor_flags = ["#IM", "#decay", "#kinesin", "#dynein"]
        tubulin_flags = ["", "#AT-IM1", "#AT-IM2"]

        for ttype in topology_types:
            for flag in topology_flags:
                top_type = f"{ttype}{flag}"
                self._system.topologies.add_type(top_type)

        # Particles
        for flag in mito_flags:
            self._system.add_topology_species(
                f"mitochondria{flag}",
                self.parameters["mitochondria"]["diffusion_constant"])

        for flag in motor_flags:
            if "decay" in flag:
                self._system.add_species(f"motor{flag}", 0.0)
            else:
                self._system.add_topology_species(f"motor{flag}", 0.)

        for flag in tubulin_flags:
            self._system.add_topology_species(f"tubulin{flag}", self.parameters["tubulin"]["diffusion_constant"])

        self._system.add_species("nucleus", 0.)
        self._system.add_species("membrane", 0.)

        mito_flag_singles = mito_flags
        mito_flag_doubles = list(product(mito_flags, repeat=2))
        mito_flag_triples = list(product(mito_flags, repeat=3))

        tubulin_singles = tubulin_flags
        tubulin_doubles = list(product(tubulin_flags, repeat=2))
        tubulin_triples = list(product(tubulin_flags, repeat=3))
        # tubulin_quads = list(product(tubulin_flags, repeat=4))

        for mito_flag in mito_flag_singles:
            # Mitochondria-Nucleus/Membrane
            self._system.potentials.add_harmonic_repulsion(
                f"mitochondria{mito_flag}",
                "nucleus",
                force_constant=self.parameters["mitochondria"]["kc_repulsion"],
                interaction_distance=(self.parameters["mitochondria"]["radius"] + self.parameters["nucleus"]["radius"])
            )

            self._system.potentials.add_harmonic_repulsion(
                f"mitochondria{mito_flag}",
                "membrane",
                force_constant=self.parameters["mitochondria"]["kc_repulsion"],
                interaction_distance=(self.parameters["mitochondria"]["radius"] + self.parameters["membrane"]["radius"])
            )

            # self._system.potentials.add_box(
            #     f"mitochondria{mito_flag}",
            #     force_constant=self.kc_boxpot,
            #     origin=self.boxpot_origin,
            #     extent=self.boxpot_extent,
            # )

            # Mito-Motor
            for motor_flag in motor_flags:
                self._system.topologies.configure_harmonic_bond(
                    f"mitochondria{mito_flag}",
                    f"motor{motor_flag}",
                    force_constant=self.parameters["mitochondria"]["kc_bond"],
                    length=(self.parameters["mitochondria"]["radius"] + self.parameters[f"motor#kinesin"]["radius"])
                )

            # Mito-Tubulin
            for tubulin_flag in tubulin_flags:
                self._system.topologies.configure_harmonic_bond(
                    f"mitochondria{mito_flag}",
                    f"tubulin{tubulin_flag}",
                    force_constant=self.parameters["mitochondria"]["kc_bond"],
                    length=(self.parameters["mitochondria"]["radius"] + self.parameters["tubulin"]["radius"])
                )

        # Mito-Mito
        for mito_flag1, mito_flag2 in mito_flag_doubles:
            self._system.potentials.add_harmonic_repulsion(
                f"mitochondria{mito_flag1}",
                f"mitochondria{mito_flag2}",
                force_constant=self.parameters["mitochondria"]["kc_repulsion"],
                interaction_distance=self.parameters["mitochondria"]["radius"] * 2)

            self._system.topologies.configure_harmonic_bond(
                f"mitochondria{mito_flag1}",
                f"mitochondria{mito_flag2}",
                force_constant=self.parameters["mitochondria"]["kc_bond"],
                length=self.parameters["mitochondria"]["eq_bond"]
            )

        # Mito-Mito-Mito
        for mito_flag1, mito_flag2, mito_flag3 in mito_flag_triples:
            self._system.topologies.configure_harmonic_angle(
                f"mitochondria{mito_flag1}",
                f"mitochondria{mito_flag2}",
                f"mitochondria{mito_flag3}",
                force_constant=self.parameters["mitochondria"]["kc_angle"],
                equilibrium_angle=np.pi
            )

        for tubulin_flag in tubulin_singles:
            # Tubulin-Nucleus/Membrane
            self._system.potentials.add_harmonic_repulsion(
                f"tubulin{tubulin_flag}",
                "nucleus",
                force_constant=self.parameters["mitochondria"]["kc_repulsion"],
                interaction_distance=(self.parameters["nucleus"]["radius"] + self.parameters["tubulin"]["radius"])
            )

            self._system.potentials.add_harmonic_repulsion(
                f"tubulin{tubulin_flag}",
                "membrane",
                force_constant=self.parameters["mitochondria"]["kc_repulsion"],
                interaction_distance=(self.parameters["membrane"]["radius"] + self.parameters["tubulin"]["radius"])
            )

            # self._system.potentials.add_box(
            #     f"tubulin{tubulin_flag}",
            #     force_constant=self.kc_boxpot,
            #     origin=self.boxpot_origin,
            #     extent=self.boxpot_extent,
            # )

            # Tubulin-Motor
            for motor_type in self._motor_types:
                self._system.topologies.configure_harmonic_bond(
                    f"tubulin{tubulin_flag}",
                    motor_type,
                    force_constant=self.parameters["tubulin"]["kc_bond"],
                    length=(self.parameters["tubulin"]["radius"] + self.parameters[motor_type]["radius"])
                )
            # for motor_flag in motor_flags:
            #     self._system.topologies.configure_harmonic_bond(
            #         f"tubulin{tubulin_flag}",
            #         f"motor{motor_flag}",
            #         force_constant=self.parameters["tubulin"]["kc_bond"],
            #         length=(self.parameters["tubulin"]["radius"] + self.parameters[f"motor{motor_flag}"]["radius"])
            #     )

        # Tubulin-Tubulin
        for tubulin_flag1, tubulin_flag2 in tubulin_doubles:
            self._system.topologies.configure_harmonic_bond(
                f"tubulin{tubulin_flag1}",
                f"tubulin{tubulin_flag2}",
                force_constant=self.parameters["tubulin"]["kc_bond"],
                length=self.parameters["tubulin"]["eq_bond"]
            )

        # Tubulin-Tubulin-Tubulin
        for tubulin_flag1, tubulin_flag2, tubulin_flag3 in tubulin_triples:
            self._system.topologies.configure_harmonic_angle(
                f"tubulin{tubulin_flag1}",
                f"tubulin{tubulin_flag2}",
                f"tubulin{tubulin_flag3}",
                force_constant=self.parameters["tubulin"]["kc_angle"],
                equilibrium_angle=self.parameters["tubulin"]["eq_angle"]
            )

        # Breakable Topology Configurations
        ptype_map = self._system._context.particle_types.type_mapping
        motor_ptype_ids = [ptype_map[mtype] for mtype in self._motor_types]
        mito_at_ptype_id = ptype_map["mitochondria#terminal#AT"]
        for motor_ptype_id in motor_ptype_ids:
            self._break_config.add_breakable_pair(
                type1=mito_at_ptype_id,
                type2=motor_ptype_id,
                threshold_energy=self.parameters["transport_parameters"]["bond_energy_threshold"],
                rate=self.parameters["transport_parameters"]["rc_cleave_bond"])

    def _setup_simulation(self):
        self._system = readdy.ReactionDiffusionSystem(
            box_size=1.1 * self.simbox_dims,
            temperature=self._temperature,
            periodic_boundary_conditions=[False, False, False],
            unit_system={
                "length_unit": "micrometer",
                "time_unit": "second"
            }
        )

        self._register_species()
        self._engine = ActiveTransportEngine(timestep=self._timestep, stride=self._stride)
        if self._enable_active_transport:
            self._engine.single_motor = self._single_motor
            self._engine.count_steps = self._count_steps
            self._engine.count_events = self._count_events

            for motor in self._motor_types:
                self._engine.register_motor_type(motor_type=motor, parameters=self.parameters[motor])

            at_reactions = ActiveTransportReactions(parameters=self.parameters, engine=self._engine)
            at_reactions.register(self._system)
            self._reaction_config.register_reaction("Step AT")

        if self._enable_mitochondrial_dynamics:
            if self._enable_active_transport:
                mito_dynamics = MitochondrialDynamics(parameters=self.parameters['mitochondria'],
                                                      engine=self._engine)
                print("Mitochondrial Dynamics with Active Transport")
            else:
                mito_dynamics = MitochondrialDynamics(parameters=self.parameters['mitochondria'])
            mito_dynamics.register(self._system)

        self._simulation = self._system.simulation(kernel=self._kernel)

        if os.path.exists(self._outdir + self._outfile + ".h5"):
            if self._replace_trials:
                os.remove(self._outdir + self._outfile + ".h5")
            else:
                n_files = len([f for f in os.listdir(self._outdir) if f.endswith(".h5")])
                self._outfile = self._outfile + f"_{n_files + 1}"

        if not self._outfile.endswith(".h5"):
            self._simulation.output_file = self._outdir + self._outfile + ".h5"
        else:
            self._simulation.output_file = self._outdir + self._outfile

        # Adding microtubule network to simulation
        if self._enable_active_transport:
            self._engine.register_with_simulation(self._microtubules.data[0],
                                                  self._simulation)

        # Adding topologies and particles to simulation
        self.add_topology_to_simulation(self._mitochondria)
        self.add_particles_to_simulation(self._nucleus)
        self.add_particles_to_simulation(self._membrane)

        self._simulation.reaction_handler = self._reaction_handler
        if self._skin:
            self._simulation._skin = self._skin

        self._simulation.record_trajectory(stride=self._stride)
        self._simulation.observe.particle_positions(stride=self._stride)
        self._simulation.observe.topologies(stride=self._stride)
        self._simulation.observe.particles(stride=self._stride)
        self._simulation.observe.reaction_counts(stride=1)
        self._simulation.observe.reactions(stride=1)

    def save(self):
        """
        Save the results to a pickle file.
        """
        mito_flags = ["", "#AT-IM1", "#AT-IM2", "#terminal#AT", "#terminal", "#internal", "#Fusion-IM1",
                      "#Fusion-IM2", "#Fission-IM1", "#Fission-IM2"]
        motor_flags = ["#IM", "#decay", "#kinesin", "#dynein"]
        tubulin_flags = ["", "#AT-IM1", "#AT-IM2"]

        # Save the parameters
        self._save_parameters()

        if self._save_trajectory:
            self._trajectory = readdy.Trajectory(self._simulation.output_file)
            self._trajectory.convert_to_xyz(
                particle_radii={
                    **{f"mitochondria{flag}": 0.15 for flag in mito_flags},
                    **{f"tubulin{flag}": 0.05 for flag in tubulin_flags},
                    **{f"motor{flag}": 0.05 for flag in motor_flags},
                    "membrane": 0.25,
                    "nucleus": 0.25,
                },
                color_ids={
                    **{f"mitochondria{flag}": 7 for flag in mito_flags},
                    **{f"tubulin{flag}": 11 for flag in tubulin_flags},
                    "motor#motor": 16,
                    "motor#IM": 2,
                    "motor#decay": 8,
                    "membrane": 1,
                    "nucleus": 0,
                },
                draw_box=True,
            )
            print("Trajectory saved to: ", self._outfile + ".h5.xyz")

        sc = SimulariumConverter(readdy_h5_path=self._simulation.output_file,
                                 with_fibers=self._enable_active_transport)
        sc.save(outfile=(self._outdir + self._outfile))

    def run(self, **kwargs):
        """ Main routine to run model construction. """
        visualize_models = kwargs.get("visualize_models", True)
        self.save_models()

        if visualize_models:
            self._setup_simulation()
            custom_loop = self._engine.get_custom_loop(self._system,
                                                       self._simulation,
                                                       self._timestep,
                                                       self._n_steps,
                                                       self._break_config,
                                                       self._reaction_config,
                                                       self._enable_active_transport)
            # self._save_parameters()
            self._simulation._run_custom_loop(custom_loop)


if __name__ == "__main__":

    cell_idx = 1
    condition = "control"
    # cell_idx = 1
    # condition = "nocodazole_30min"
    # cell_idx = 3
    # condition = "nocodazole_60min"

    # mf = ModelFactory(cell_idx=cell_idx,
    #                   condition=condition,
    #                   n_steps=10,
    #                   enable_active_transport=True,
    #                   enable_mitochondrial_dynamics=False,
    #                   membrane_fraction=0.55,
    #                   outdir="/home/earkfeld/PycharmProjects/mitosim/data/mt_network_plotting/")
    #                   # outdir="/home/earkfeld/PycharmProjects/mitosim/data/processed_data/models/noco60_with_microtubules/")

    mf = ModelFactory(cell_idx=cell_idx,
                      condition=condition,
                      n_steps=10,
                      enable_active_transport=True,
                      enable_mitochondrial_dynamics=False,
                      outdir="/home/earkfeld/PycharmProjects/mitosim/data/processed_data/models/model_files_v6/")

    mf.run()
    mf.save()

    # for cid in [0, 1, 2, 3]:
    #     mf = ModelFactory(cell_idx=cid,
    #                       condition="nocodazole_60min",
    #                       n_steps=10,
    #                       enable_active_transport=True,
    #                       enable_mitochondrial_dynamics=False,
    #                       # outdir="/home/earkfeld/PycharmProjects/mitosim/data/outlier_exploration/")
    #                       outdir="/home/earkfeld/PycharmProjects/mitosim/data/processed_data/models/noco60_with_microtubules/")
    #     mf.run()
    #     mf.save()

    # for condition in ["control", "nocodazole_30min"]:
    #     for cell_idx in [0, 1, 2, 3]:
    #         mf = ModelFactory(cell_idx=cell_idx,
    #                           condition=condition,
    #                           n_steps=10,
    #                           enable_active_transport=True,
    #                           enable_mitochondrial_dynamics=False)
    #         mf.run()
    #         mf.save()

    # mf = ModelFactory(cell_idx=0,
    #                   condition="nocodazole_30min",
    #                   n_steps=10,
    #                   enable_active_transport=True,
    #                   enable_mitochondrial_dynamics=False)
    # mf.run()
    # mf.save()
