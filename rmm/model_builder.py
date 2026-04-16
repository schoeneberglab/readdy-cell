import readdy
import numpy as np
import tifffile
from tqdm import trange
import os
from readdy.api.utils import vec3_of
from itertools import product

def from_vec3(vec):
    if not isinstance(vec, np.ndarray):
        vec = np.array([vec[0], vec[1], vec[2]])
    return vec

class ModelBuilder:
    def __init__(self):        
        self.n_steps = int(1e4)
        self.voxel_size = np.array([0.111, 0.111, 0.111])

        self.particle_radius = 0.1
        self.diffusivity = 0.001
        self.temperature = 1.0
        self.dt = 0.01
        self.stride = 10

        self.box_size = None
        self.system = None
        self.simulation = None

        self.fc_box = 0.1
        self.fc_repulsion = 0.001
        self.fc_bond = 0.002
        self.eq_bond = 0.2
        self.fc_angle = 3.0
        self.eq_angle = np.pi
        self.fc_gradient = 0.005
        self.min_topology_size = 5
        
        self.seed_n_particles = 100
        self.n_add_per_step = 10
        self.max_n_particles = 20000
        self.decay_rate = 10.0
        self.bond_rate = 100.0
        
        self._img_arr = np.random.rand(10, 10, 10)
        self._intensity_threshold = 0.7
        self._high_intensity_coords = None
        self._ugrad_arr = None

        self.p_types = ["P0", "P1", "P2", "C"]
        self.centrosome_coordinate = None

        self._debug = False

    def load(self, path):
        img = np.array(tifffile.imread(path))
        
        # Swap the axes from (z,y,x) to (x,-y,z) 
        self._img_arr = np.transpose(img, (2, 1, 0))
        self._img_arr = np.flip(self._img_arr, axis=1)
        self._img_arr_coords = np.copy(self._img_arr)

        # Check if the image is binary
        unique_values = np.unique(self._img_arr)
        if len(unique_values) == 2 and set(unique_values).issubset({0, 1}):
            self._img_arr = self._img_arr.astype(np.float32)

            # Light gaussian blur to create gradients
            from scipy.ndimage import gaussian_filter
            self._img_arr = gaussian_filter(self._img_arr.astype(np.float32), sigma=1.0, radius=2)

        # Normalize to [0, 1]
        self._img_arr = (self._img_arr - self._img_arr.min()) / (self._img_arr.max() - self._img_arr.min())

        self.box_size = np.array([self._img_arr.shape[0], self._img_arr.shape[1], self._img_arr.shape[2]], dtype=float) * self.voxel_size
        self.box_origin = -self.box_size / 2.0 + self.voxel_size / 2.0 # Offset to voxel center to avoid boundary issues

        # Set up coordinates
        indices = np.argwhere(self._img_arr >= self._intensity_threshold)
        physical_coords = indices * self.voxel_size
        physical_coords += self.box_origin
        self._high_intensity_coords = physical_coords.reshape(-1, 3)

    def _calculate_intensity_gradient(self):
        """ Computes the (unit) intensity gradient of the image array in +∇I direction. """
        # np.gradient returns gradients along axis order (x, y, z) for an array shaped (x, y, z).
        # Use physical voxel spacing so the gradient is correct in world units.
        gx, gy, gz = np.gradient(
            self._img_arr,
            self.voxel_size[0],
            self.voxel_size[1],
            self.voxel_size[2]
        )

        magnitude = np.sqrt(gx**2 + gy**2 + gz**2) + 1e-10  # Avoid division by zero
        self._ugrad_arr = np.stack((gx / magnitude, gy / magnitude, gz / magnitude), axis=-1) # Unit grad towards higher intensity

    def _setup(self):

        self.system = readdy.ReactionDiffusionSystem(box_size=1.2 *self.box_size, temperature=self.temperature)

        self.system.topologies.add_type("T")
        self.system.topologies.add_type("D")
        self.system.add_species("decay", diffusion_constant=0.0)

        for pt in self.p_types:
            self.system.add_topology_species(name=pt, diffusion_constant=self.diffusivity)
            self.system.potentials.add_box(pt, origin=self.box_origin, extent=self.box_size, force_constant=self.fc_box)

        for pt1, pt2 in product(self.p_types, repeat=2):
            # Add repulsion between all particle type pairs (including same-type)
            self.system.potentials.add_harmonic_repulsion(pt1, pt2, force_constant=self.fc_repulsion, interaction_distance=2*self.particle_radius)
            self.system.topologies.configure_harmonic_bond(pt1, pt2, force_constant=self.fc_bond, length=self.eq_bond)

        for pt1, pt2, pt3 in product(self.p_types, repeat=3):
            self.system.topologies.configure_harmonic_angle(pt1, pt2, pt3, force_constant=self.fc_angle, equilibrium_angle=self.eq_angle)

        #-- tubule fusion reactions
        self.system.topologies.add_spatial_reaction("fusion_00: T(P0) + T(P0) -> T(P1--P1)", rate=self.bond_rate, radius=2.0*self.particle_radius)
        self.system.topologies.add_spatial_reaction("fusion_10: T(P1) + T(P0) -> T(P2--P1)", rate=self.bond_rate, radius=3.0*self.particle_radius)
        self.system.topologies.add_spatial_reaction("fusion_11: T(P1) + T(P1) -> T(P2--P2)", rate=self.bond_rate, radius=4.0*self.particle_radius)

        #-- decay reactions
        self.system.topologies.add_spatial_reaction("decay_r1: T(P2) + T(P0) -> T(P2) + D(P0)", rate=self.decay_rate, radius=2.0*self.particle_radius)
        self.system.topologies.add_structural_reaction(name="decay_r2", topology_type="D", reaction_function=self.to_decay_reaction_fn, rate_function=lambda x: 1.e32, expect_connected=False)
        self.system.reactions.add_decay("decay_r3", "decay", rate=1.e32)
        
        #-- centrosome reactions
        if self.centrosome_coordinate is not None:
            self.system.topologies.add_spatial_reaction("fusion_C0: T(C) + T(P0) -> T(C--P1)", rate=self.bond_rate, radius=4.0*self.particle_radius)
            self.system.topologies.add_spatial_reaction("fusion_C1: T(C) + T(P1) -> T(C--P2)", rate=self.bond_rate, radius=4.0*self.particle_radius)

        self.simulation = self.system.simulation(kernel="CPU")

        # Add initial particles at high-intensity coordinates
        sampled_indices = np.random.choice(self._high_intensity_coords.shape[0], self.seed_n_particles, replace=False)
        initial_coords = self._high_intensity_coords[sampled_indices]
        for i in range(initial_coords.shape[0]):
            coord = initial_coords[i]
            self.simulation.add_topology(
                topology_type="T",
                particle_types=["P0"],
                positions=np.array([coord])
            )
        
        if self.centrosome_coordinate is not None:
            self.simulation.add_topology(
                topology_type="T",
                particle_types=["C"],
                positions=np.array([self.centrosome_coordinate])
            )

    def to_decay_reaction_fn(self, topology):
        """ Structural reaction which removes single-particle topologies. """
        recipe = readdy.StructuralReactionRecipe(topology)
        graph = topology.get_graph()
        v = graph.get_vertices(vertex_type="P0", exact_match=True)[0]

        # Separate and change to decay type
        recipe.change_particle_type(v, "decay")
        return recipe
    
    def to_decay_rate_fn(self):
        return self.decay_rate

    def simulation_loop(self):
        A = self.simulation._actions
        init = A.initialize_kernel()
        integrator = A.integrator_euler_active_brownian_dynamics(self.dt)
        create_nl = A.create_neighbor_list(self.system.calculate_max_cutoff().magnitude)
        update_nl = A.update_neighbor_list()
        calc_forces = A.calculate_forces()
        react_part = A.reaction_handler_gillespie(self.dt)
        react_topo = A.topology_reaction_handler(self.dt)
        observe = A.evaluate_observables()

        init()
        create_nl()
        observe(0)
        forces = None
        ids = None
        n_added = self.seed_n_particles
        add_particles = True
        for t in trange(1, self.n_steps + 1):
            update_nl()
            react_part()
            react_topo()
            update_nl()
            calc_forces()
            integrator.perform(ids=ids, forces=forces) if forces is not None else integrator.perform()
            observe(t)

            ps = self.simulation._simulation.current_particles
            pos = []
            ids = []
            for p in ps:
                pos.append(from_vec3(p.pos))
                ids.append(p.id)
            pos = np.array(pos)
            if pos.shape[0] > 0:
                forces = self._get_forces(pos)
            else:
                ids = None
                forces = None

            if add_particles:
                
                sampled_indices = np.random.choice(self._high_intensity_coords.shape[0], self.n_add_per_step, replace=False)
                sample_coords = self._high_intensity_coords[sampled_indices]
                
                # Remove the indices from the pool to avoid re-sampling
                for coord in sample_coords:
                    self.simulation.add_topology(
                        topology_type="T",
                        particle_types=["P0"],
                        positions=np.array([coord])
                    )

                n_added += self.n_add_per_step
                if n_added >= self.max_n_particles:
                    add_particles = False
                    print(f"Reached max number of particles: {self.max_n_particles}, stopping addition.")
        
        if add_particles:
            print(f"Finished simulation loop after {self.n_steps} steps with {n_added} particles added.")

    def _get_forces(self, positions):
        forces = []
        nx, ny, nz = self._ugrad_arr.shape[:3]
        for pos in positions:
            # get integer voxel indices
            rel_pos = (pos - self.box_origin) / self.voxel_size
            ix, iy, iz = np.floor(rel_pos).astype(int)
            
            # inside?
            inside = (0 <= ix < nx) and (0 <= iy < ny) and (0 <= iz < nz)
            if inside:
                # Apply gradient force
                grad = self._ugrad_arr[ix, iy, iz]
                force = self.fc_gradient * grad
            else:
                # no gradient force outside the image volume
                force = np.array([0.0, 0.0, 0.0])
            forces.append(vec3_of(force))
        return forces

    def run(self, filename="out", show_summary=True, visualize=True, remove_existing=True):
        self._calculate_intensity_gradient()
        self._setup()
       
        self.simulation.output_file = filename + ".h5"
        
        print(self.system.time_unit)
        print(self.system.length_unit)
        print(self.system.energy_unit)

        if remove_existing and os.path.exists(self.simulation.output_file):
            os.remove(self.simulation.output_file)

        self.simulation.observe.topologies(stride=self.stride)
        self.simulation.record_trajectory(stride=self.stride)

        self.simulation._run_custom_loop(self.simulation_loop, show_summary=show_summary)

        if visualize:
            trajectory = readdy.Trajectory(filename + ".h5")
            trajectory.convert_to_xyz(particle_radii={"P0": self.particle_radius,
                                                      "P1": self.particle_radius,
                                                      "P2": self.particle_radius,
                                                      "decay": self.particle_radius,
                                                      "C": 3*self.particle_radius,
                                                      }, draw_box=True)