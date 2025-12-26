import os
import numpy as np
import readdy

print(readdy.__version__)


name = "msd"
n_particles = 1000
origin = np.array([-9.,-9.,-9.])
extent = np.array([18.,18.,18.])

data_dir = "."
out_file = os.path.join(data_dir, f"{name}.h5")
n_steps = 2000
dt = 1e-2


system = readdy.ReactionDiffusionSystem(
    [20.,20.,20.], 
    periodic_boundary_conditions=[False, False, False],
    unit_system=None)

system.add_species("A", 0.1)

system.potentials.add_box("A", 100., origin=origin, extent=extent)
system.potentials.add_harmonic_repulsion("A", "A", force_constant=100., interaction_distance=2.)

simulation = system.simulation("CPU")
simulation.output_file = out_file

simulation.record_trajectory(stride=1)
simulation.observe.particle_positions(stride=1)

init_pos = np.random.uniform(size=(n_particles, 3)) * extent + origin
simulation.add_particles("A", init_pos)

if os.path.exists(simulation.output_file):
    os.remove(simulation.output_file)
    
simulation.run(n_steps, dt)
traj = readdy.Trajectory(out_file)
traj.convert_to_xyz(generate_tcl=True)
times, positions = traj.read_observable_particle_positions()

T = len(positions)
N = len(positions[0])
pos = np.zeros(shape=(T, N, 3))
for t in range(T):
    for n in range(N):
        pos[t, n, 0] = positions[t][n][0]
        pos[t, n, 1] = positions[t][n][1]
        pos[t, n, 2] = positions[t][n][2]

difference = pos - pos[0]
squared_displacements = np.sum(difference * difference, axis=2)  # sum over coordinates, per particle per timestep
squared_displacements = squared_displacements.transpose()  # T x N -> N x T
mean = np.mean(squared_displacements, axis=0)
std_dev = np.std(squared_displacements, axis=0)
std_err = np.std(squared_displacements, axis=0) / np.sqrt(len(squared_displacements))

print(f'Mean squared displacement: {mean[-1]}')
print(f'Standard deviation: {std_dev[-1]}')
print(f'Standard error: {std_err[-1]}')

# os.remove(out_file)

