import numpy as np
from scipy.linalg import lstsq

from src.analysis import TopologyGraphs
from src.analysis.base import AnalysisBase
import readdy

ut = readdy.units

class Diffusivity(AnalysisBase):
    def __init__(self, trajectory_file, timestep, **kwargs):
        super().__init__()
        self._trajectory_file = trajectory_file
        self._timestep = timestep
        self._particle_type = kwargs.get("particle_type", None)
        self._frame_interval = kwargs.get("frame_interval", 1)
        self._equilibration_fraction = kwargs.get("equilibration_fraction", 0)
        self._min_n_frames = kwargs.get("min_n_frames", 2)
        self._min_n_particles = kwargs.get("min_n_particles", 1)

    def _get_node_msd_matrix(self):
        """ Calculate the diffusion coefficient based on the MSD of all particles. """
        traj = readdy.Trajectory(self._trajectory_file)
        time, types, ids, positions = traj.read_observable_particles()
        times = time * self._timestep

        # Get the unique particle ids for the particle type
        puids = set()
        for t, frame_pos in enumerate(positions):
            for i, pos in enumerate(frame_pos):
                ptype = traj._inverse_types_map[types[t][i]]
                puid = ids[t][i]
                if self._particle_type in ptype:
                    puids.add(puid)

        # Get the positions of the particles
        ppos = {uid: np.zeros(shape=(len(times), 3)) for uid in puids}
        for t, frame_pos in enumerate(positions):
            for i, pos in enumerate(frame_pos):
                if ids[t][i] in puids:
                    ppos[ids[t][i]][t, :] = pos

        msd_matrix = np.full(shape=(len(times), len(puids)), fill_value=np.nan)
        for i, uid in enumerate(puids):
            for t in range(len(times)):
                msd_matrix[t, i] = np.mean(np.linalg.norm(ppos[uid][t] - ppos[uid][0], axis=0) ** 2)
        return msd_matrix

    def _get_segment_msd_matrix(self):
        raise NotImplementedError

    def _get_fragment_msd_matrix(self):
        # Get the center of mass trajectories for each unique topology
        tg = TopologyGraphs(self._trajectory_file, self._timestep)
        tg.run(particle_types=self._particle_type)
        ttraj = tg.get_unique_topology_trajectories(self._particle_type, self._equilibration_fraction)


        # Flatten the dictionary to a list per-trajectory dictionaries
        ttrajs = []
        max_n_frames = 0
        for tuid, tuid_trajs in ttraj.items():
            for traj in tuid_trajs:

                # Set up frame indices
                indices = np.arange(0, len(traj["coordinate"]), self._frame_interval)
                n_particles = traj["graphs"][0].vcount()
                # if len(indices) < self._min_n_frames:
                if len(indices) < self._min_n_frames or n_particles < self._min_n_particles:
                    continue
                else:
                    max_n_frames = max(len(indices), max_n_frames)

                # Get the coordinates for the frames
                traj["coordinate"] = traj["coordinate"][indices]
                ttrajs.append(traj)

        # Add the fragment trajectories to the results
        self._results.fragment.trajectories = ttrajs

        # Set up and populate the msd matrix
        msd_matrix = np.full(shape=(max_n_frames, len(ttrajs)), fill_value=np.nan)
        disp_matrix = np.full_like(msd_matrix, fill_value=np.nan)
        for i, traj in enumerate(ttrajs):
            coordinates = traj["coordinate"]
            for t in range(coordinates.shape[0]):
                msd_matrix[t, i] = np.mean(np.linalg.norm(coordinates[t] - coordinates[0], axis=0) ** 2)
                disp_matrix[t, i] = np.linalg.norm(coordinates[t] - coordinates[0], axis=0)

        self._results.fragment.displacement_matrix = disp_matrix
        self._results.fragment.msd_matrix = msd_matrix

    def _calculate_diffusivity(self, msd_matrix: np.ndarray):
        """Computes the diffusivity for each MSD in the MSD array based on tau values."""
        diffusivity = []
        for i in range(msd_matrix.shape[1]):
            # Slice the matrix to get the valid MSD values (non-NaN)
            valid_msd = msd_matrix[:, i][~np.isnan(msd_matrix[:, i])]

            # Check if there are enough valid MSD values to calculate a diffusivity
            if len(valid_msd) <= 1:
                diffusivity.append({'diffusivity': np.nan, 'msd': np.nan, 'r_squared': np.nan, 'num_points': 1})
                continue

            n_points = min(len(valid_msd), msd_matrix.shape[1])
            all_tau = np.arange(0, n_points * self._frame_interval, self._frame_interval)[:, np.newaxis]

            slope, res, _, _ = lstsq(all_tau[:n_points], valid_msd[:n_points])
            d = slope[0] / 6
            msd_per_frame = 6 * d * self._frame_interval

            # R² calculation
            msd_mean = np.mean(valid_msd[:n_points])
            total_sum = np.sum((valid_msd[:n_points] - msd_mean) ** 2)
            r_squared = np.nan if total_sum == 0 else 1 - res / total_sum

            diffusivity.append({
                'diffusivity': d,
                'msd': msd_per_frame,
                'r_squared': r_squared,
                'num_points': n_points
            })
        return diffusivity

    def run(self):
        self._results.node.msd_matrix = self._get_node_msd_matrix()
        self._results.node.diffusivity = self._calculate_diffusivity(self._results.node.msd_matrix)

        # TODO: Implement me
        # self._results.segment.msd_matrix = self._get_segment_msd_matrix()
        # self._results.segment.diffusivity = self._calculate_diffusivity(self._results.segment.msd_matrix)

        self._get_fragment_msd_matrix()
        self._results.fragment.diffusivity = self._calculate_diffusivity(self._results.fragment.msd_matrix)

    def save(self, filename: str):
        pass

    @property
    def particle_type(self):
        return self._particle_type

    @particle_type.setter
    def particle_type(self, value):
        self._particle_type = value

    @property
    def frame_interval(self):
        return self._frame_interval

    @frame_interval.setter
    def frame_interval(self, value):
        self._frame_interval = value

    @property
    def equilibration_fraction(self):
        return self._equilibration_fraction

    @equilibration_fraction.setter
    def equilibration_fraction(self, value):
        self._equilibration_fraction = value

    @property
    def min_n_frames(self):
        return self._min_n_frames

    @min_n_frames.setter
    def min_n_frames(self, value):
        self._min_n_frames = value

    def __repr__(self):
        return f"Diffusivity(trajectory_file={self._trajectory_file}, timestep={self._timestep}, particle_type={self._particle_type}, frame_interval={self._frame_interval}, equilibration_fraction={self._equilibration_fraction}, min_n_frames={self._min_n_frames})"