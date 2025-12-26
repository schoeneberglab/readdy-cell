from itertools import product
import igraph as ig
import json
import os
import numpy as np
import pickle
import readdy
from readdy.api.experimental.action_factory import BreakConfig, ReactionConfig
from tqdm import trange, tqdm
import yaml

from src.analysis import Velocity, TopologyGraphs
from src.core import DataLoader
from src.reactions import ActiveTransportEngine, ActiveTransportReactions, MitochondrialDynamics
from src.utils.utils import *
from src.visualization import SimulariumConverter

ut = readdy.units
np.random.seed(42)

class CellSimulation:
    def __init__(self, cfg, *args, **kwargs):
        self._cfg = cfg
        self.models = kwargs.get("models", None)

        self._nucleus = None
        self._membrane = None
        self._mitochondria = None
        self._microtubules = None

    def _setup_system(self, **kwargs):
        """ Setup the simulation system with parameters from the configuration."""
        self._cfg["run_parameters"]["flags"]["enable_active_transport"] = kwargs.get("enable_active_transport", True)
        self._cfg["run_parameters"]["flags"]["enable_mitochondrial_dynamics"] = kwargs.get("enable_mitochondrial_dynamics", True)

        rc_fusion = kwargs.get("mitochondria_fusion_rate", None)
        if rc_fusion is not None:
            self._cfg["mitochondria"]["mitochondria_fusion_rate"] = rc_fusion

        # Run parameters
        self._save_simularium = self._cfg["run_parameters"]["flags"]["save_simularium"]
        self._save_trajectory = self._cfg["run_parameters"]["flags"]["save_trajectory"]
        self._enable_active_transport = self._cfg["run_parameters"]["flags"]["enable_active_transport"]
        self._single_motor = self._cfg["run_parameters"]["flags"]["single_motor"]
        self._count_steps = self._cfg["run_parameters"]["flags"]["count_steps"]
        self._count_events = self._cfg["run_parameters"]["flags"]["count_events"]
        self._enable_mitochondrial_dynamics = self._cfg["run_parameters"]["flags"]["enable_mitochondrial_dynamics"]

        self.kc_boxpot = 0.01 * ut.kilojoule / ut.mol / ut.nm ** 2
        self._skin = 1.0

        self._kernel = kwargs.get("kernel", "CPU")
        self._reaction_handler = "Gillespie"

        self._n_steps = int(float(self._cfg["simulation"]["n_steps"]))
        self._stride = int(float(self._cfg["simulation"]["stride"]))
        self._timestep = float(self._cfg["simulation"]["dt"]) * ut.s
        self._temperature = float(self._cfg["simulation"]["temperature"]) * ut.kelvin

        model_file = self._cfg["run_parameters"]["io"]["model_file"]
        condition = kwargs.get("condition", model_file.split("/")[-1].split("_")[0])
        cell_idx = kwargs.get("cell_idx", model_file.split("/")[-1].split("_")[1][-1])

        try:
            self.load_models()
        except Exception as e:
            raise BrokenPipeError(f"Error loading models: {e}")

        self._run_index = kwargs.get("run_index", None)
        self._outdir = kwargs.get("workdir", self._cfg["run_parameters"]["io"]["workdir"])
        self._outfile = kwargs.get("outfile", self._cfg["run_parameters"]["io"]["outfile"])

        if self._run_index is not None:
            self._outfile += f"_{self._run_index}"

        p_active_activate = kwargs.get("p_active_activate", None)
        if p_active_activate is not None:
            self._cfg["motor#kinesin"]["p_active"] = p_active_activate
            self._cfg["motor#dynein"]["p_active"] = p_active_activate
            self._cfg["motor#kinesin"]["p_activate"] = p_active_activate
            self._cfg["motor#dynein"]["p_activate"] = p_active_activate

        prev_trajfile = f"data/trajectories/cell_model_validation/production_runs/t2/{condition}_c{cell_idx}/" + f"{condition}_c{cell_idx}.h5"
        self._from_prev_traj = kwargs.get("from_prev_traj", False)
        self._replace_trials = kwargs.get("replace_trials", True)

        print(f"Output directory: {self._outdir}")
        print(f"Output file: {self._outfile}")

        # Check if the output directory exists
        if not os.path.exists(self._outdir):
            os.makedirs(self._outdir)
        self.simbox_dims = None

        # Private Vars
        self._system = None
        self._simulation = None
        self._engine = None
        self._trajectory = None
        self._converged = False
        self._break_config = None
        self._reaction_config = None
        self._motor_types = ["motor#kinesin", "motor#dynein"]

        max_coord = np.max(self._membrane.data, axis=0)
        min_coord = np.min(self._membrane.data, axis=0)
        self.simbox_dims = max_coord - min_coord

        if self._from_prev_traj:
            print(f"Getting model from: \n {prev_trajfile}")
            self._mitochondria.data = self.get_model_from_trajectory(prev_trajfile)

    @property
    def cfg(self):
        return self._cfg
    
    @cfg.setter
    def cfg(self, cfg):
        self._cfg = cfg

    def load_models(self):
        if self.models is None:
            model_file = self._cfg["run_parameters"]["io"]["model_file"]
            
            with open(model_file, "rb") as f:
                self.models = pickle.load(f)
        
        self._nucleus = self.models["nucleus"]
        self._membrane = self.models["membrane"]
        self._mitochondria = self.models["mitochondria"]
        
        if self._enable_active_transport:
            self._microtubules = self.models["microtubules"]

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
        print(f"Added {len(graphs)} topologies of type {topology_type} with particle type {particle_type} to the simulation.")

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
        converted_parameters = convert_parameters(self._cfg)

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
        motor_flags = ["", "#IM", "#decay", "#kinesin", "#dynein"]
        tubulin_flags = ["", "#AT-IM1", "#AT-IM2"]

        for ttype in topology_types:
            for flag in topology_flags:
                top_type = f"{ttype}{flag}"
                self._system.topologies.add_type(top_type)

        # Particles
        for flag in mito_flags:
            self._system.add_topology_species(
                f"mitochondria{flag}",
                self._cfg["mitochondria"]["diffusion_constant"] * ut.um ** 2 / ut.s)

        for flag in motor_flags:
            if "decay" in flag:
                self._system.add_species(f"motor{flag}", 0.0 * ut.um ** 2 / ut.s)
            else:
                self._system.add_topology_species(f"motor{flag}", 0.0 * ut.um ** 2 / ut.s)

        for flag in tubulin_flags:
            self._system.add_topology_species(f"tubulin{flag}", self._cfg["tubulin"]["diffusion_constant"] * ut.um ** 2 / ut.s)

        self._system.add_species("nucleus", 0.0 * ut.um ** 2 / ut.s)
        self._system.add_species("membrane", 0.0 * ut.um ** 2 / ut.s)

        mito_flag_singles = mito_flags
        mito_flag_doubles = list(product(mito_flags, repeat=2))
        mito_flag_triples = list(product(mito_flags, repeat=3))

        tubulin_singles = tubulin_flags
        tubulin_doubles = list(product(tubulin_flags, repeat=2))
        tubulin_triples = list(product(tubulin_flags, repeat=3))

        for mito_flag in mito_flag_singles:
            # Mitochondria-Nucleus/Membrane
            self._system.potentials.add_harmonic_repulsion(
                f"mitochondria{mito_flag}",
                "nucleus",
                force_constant=self._cfg["mitochondria"]["kc_repulsion"] * ut.kilojoule / ut.mol / ut.nm ** 2,
                interaction_distance=(self._cfg["mitochondria"]["radius"] + self._cfg["nucleus"]["radius"])
            )

            self._system.potentials.add_harmonic_repulsion(
                f"mitochondria{mito_flag}",
                "membrane",
                force_constant=self._cfg["mitochondria"]["kc_repulsion"] * ut.kilojoule / ut.mol / ut.nm ** 2,
                interaction_distance=(self._cfg["mitochondria"]["radius"] + self._cfg["membrane"]["radius"])
            )

            # Mito-Motor
            for motor_flag in motor_flags:
                self._system.topologies.configure_harmonic_bond(
                    f"mitochondria{mito_flag}",
                    f"motor{motor_flag}",
                    force_constant=self._cfg["mitochondria"]["kc_bond"] * ut.kilojoule / ut.mol / ut.nm ** 2,
                    length=(self._cfg["mitochondria"]["radius"] + self._cfg["motor"]["kinesin"]["radius"])
                )

            # Mito-Tubulin
            for tubulin_flag in tubulin_flags:
                self._system.topologies.configure_harmonic_bond(
                    f"mitochondria{mito_flag}",
                    f"tubulin{tubulin_flag}",
                    force_constant=self._cfg["mitochondria"]["kc_bond"] * ut.kilojoule / ut.mol / ut.nm ** 2,
                    length=(self._cfg["mitochondria"]["radius"] + self._cfg["tubulin"]["radius"])
                )

        # Mito-Mito
        for mito_flag1, mito_flag2 in mito_flag_doubles:
            self._system.potentials.add_harmonic_repulsion(
                f"mitochondria{mito_flag1}",
                f"mitochondria{mito_flag2}",
                force_constant=self._cfg["mitochondria"]["kc_repulsion"] * ut.kilojoule / ut.mol / ut.nm ** 2,
                interaction_distance=self._cfg["mitochondria"]["radius"] * 2)

            self._system.topologies.configure_harmonic_bond(
                f"mitochondria{mito_flag1}",
                f"mitochondria{mito_flag2}",
                force_constant=self._cfg["mitochondria"]["kc_bond"] * ut.kilojoule / ut.mol / ut.nm ** 2,
                length=self._cfg["mitochondria"]["eq_bond"]
            )

        # Mito-Mito-Mito
        for mito_flag1, mito_flag2, mito_flag3 in mito_flag_triples:
            self._system.topologies.configure_harmonic_angle(
                f"mitochondria{mito_flag1}",
                f"mitochondria{mito_flag2}",
                f"mitochondria{mito_flag3}",
                force_constant=self._cfg["mitochondria"]["kc_angle"] * ut.kilojoule / ut.mol / ut.rad ** 2,
                equilibrium_angle=np.pi
            )

        for tubulin_flag in tubulin_singles:
            # Tubulin-Nucleus/Membrane
            self._system.potentials.add_harmonic_repulsion(
                f"tubulin{tubulin_flag}",
                "nucleus",
                force_constant=self._cfg["mitochondria"]["kc_repulsion"] * ut.kilojoule / ut.mol / ut.nm ** 2,
                interaction_distance=(self._cfg["nucleus"]["radius"] + self._cfg["tubulin"]["radius"])
            )

            self._system.potentials.add_harmonic_repulsion(
                f"tubulin{tubulin_flag}",
                "membrane",
                force_constant=self._cfg["mitochondria"]["kc_repulsion"] * ut.kilojoule / ut.mol / ut.nm ** 2,
                interaction_distance=(self._cfg["membrane"]["radius"] + self._cfg["tubulin"]["radius"])
            )

            # Tubulin-Motor
            for motor_flag in motor_flags:
                self._system.topologies.configure_harmonic_bond(
                    f"tubulin{tubulin_flag}",
                    f"motor{motor_flag}",
                    force_constant=self._cfg["tubulin"]["kc_bond"] * ut.kilojoule / ut.mol / ut.nm ** 2,
                    length=(self._cfg["tubulin"]["radius"] + self._cfg["motor"]["kinesin"]["radius"])
                )

        # Tubulin-Tubulin
        for tubulin_flag1, tubulin_flag2 in tubulin_doubles:
            self._system.topologies.configure_harmonic_bond(
                f"tubulin{tubulin_flag1}",
                f"tubulin{tubulin_flag2}",
                force_constant=self._cfg["tubulin"]["kc_bond"] * ut.kilojoule / ut.mol / ut.nm ** 2,
                length=self._cfg["tubulin"]["eq_bond"]
            )

        # Tubulin-Tubulin-Tubulin
        for tubulin_flag1, tubulin_flag2, tubulin_flag3 in tubulin_triples:
            self._system.topologies.configure_harmonic_angle(
                f"tubulin{tubulin_flag1}",
                f"tubulin{tubulin_flag2}",
                f"tubulin{tubulin_flag3}",
                force_constant=self._cfg["tubulin"]["kc_angle"] * ut.kilojoule / ut.mol / ut.rad ** 2,
                equilibrium_angle=self._cfg["tubulin"]["eq_angle"]
            )

        # Breakable Topology Configurations
        ptype_map = self._system._context.particle_types.type_mapping
        motor_ptype_ids = [ptype_map[mtype] for mtype in self._motor_types]
        mito_at_ptype_id = ptype_map["mitochondria#terminal#AT"]
        for motor_ptype_id in motor_ptype_ids:
            self._break_config.add_breakable_pair(
                type1=mito_at_ptype_id,
                type2=motor_ptype_id,
                threshold_energy=self._cfg["transport_parameters"]["bond_energy_threshold"],
                rate=float(self._cfg["transport_parameters"]["rc_cleave_bond"]))

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

        # Register all species and topology types
        self._register_species()

        self._engine = ActiveTransportEngine(
            config=self._cfg,
            timestep=self._timestep, 
            stride=self._stride
            )

        if self._enable_active_transport:
            self._engine.single_motor = self._single_motor
            self._engine.count_steps = self._count_steps
            self._engine.count_events = self._count_events

            for motor in self._motor_types:
                self._engine.register_motor_type(motor_type=motor, parameters=self._cfg["motor"][motor.split("#")[-1]])

            at_reactions = ActiveTransportReactions(parameters=self._cfg, engine=self._engine)
            at_reactions.register(self._system)
            self._reaction_config.register_reaction("Step AT")

        if self._enable_mitochondrial_dynamics:
            if self._enable_active_transport:
                mito_dynamics = MitochondrialDynamics(parameters=self._cfg['mitochondria'],
                                                      engine=self._engine)
                print("Mitochondrial Dynamics with Active Transport")
            else:
                mito_dynamics = MitochondrialDynamics(parameters=self._cfg['mitochondria'])
            mito_dynamics.register(self._system)

        self._simulation = self._system.simulation(kernel=self._kernel)

        if os.path.exists(self._outdir + self._outfile + ".h5"):
            if self._replace_trials:
                os.remove(self._outdir + self._outfile + ".h5")
            else:
                n_files = len([f for f in os.listdir(self._outdir) if f.endswith(".h5")])
                self._outfile = self._outfile + f"_{n_files+1}"

        if not self._outfile.endswith(".h5"):
            self._simulation.output_file = self._outdir + self._outfile + ".h5"
        else:
            self._simulation.output_file = self._outdir + self._outfile

        # Adding microtubule network to simulation
        if self._enable_active_transport:
            self._engine.register_with_simulation(self._microtubules.data,
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

        if self._save_simularium:
            sc = SimulariumConverter(readdy_h5_path=self._simulation.output_file,
                                    with_fibers=self._enable_active_transport)
            sc.save(outfile=(self._outdir + self._outfile))

    def run(self, show_summary=True):
        """ Main routine to run the _simulation optimization loop. """
        self._setup_system()
        self._setup_simulation()
        custom_loop = self._engine.get_custom_loop(self._system,
                                                   self._simulation,
                                                   self._timestep,
                                                   self._n_steps,
                                                   self._break_config,
                                                   self._reaction_config,
                                                   self._enable_active_transport,
                                                   self._stride)
        self._save_parameters()
        self._simulation._run_custom_loop(custom_loop, show_summary=show_summary)
        self.save()

if __name__ == "__main__":
    config = "config.yaml"
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)

    csim = CellSimulation(cfg)
    csim.run()