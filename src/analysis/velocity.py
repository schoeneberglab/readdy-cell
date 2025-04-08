import readdy
import numpy as np
from typing import Optional
ut = readdy.units

# TODO: Update to use base classes
class Velocity:
    def __init__(self, timestep, simulation: Optional[readdy.Simulation] = None, **kwargs):
        """ Registers required observables and calculates the velocity of specified particles in the simulation. """
        self._timestep = timestep
        if isinstance(self._timestep, ut.Unit):
            self._timestep = self._timestep.to("s").magnitude

        self._readdy_file = None
        self._stride = kwargs.get("stride", 1)
        self._time_unit = kwargs.get("time_unit", "s")
        self._length_unit = kwargs.get("length_unit", "um")
        self.velocity_unit = kwargs.get("velocity_unit", "um/s")

        self._trajectory = None
        self.times = None
        self.types = None
        self.ids = None
        self.positions = None
        self.timestep = None

        if simulation is not None:
            self._register(simulation)

    @property
    def readdy_file(self):
        return self._readdy_file

    @readdy_file.setter
    def readdy_file(self, value):
        self._readdy_file = value

    def _register(self, simulation: readdy.Simulation):
        """ Registers the required observables for calculating velocities. """
        self._readdy_file = simulation.output_file
        try:
            simulation.observe.particles(self._stride)
            # simulation.observe.particles(self._stride, save={"name": "velocity_analysis", "chunk_size": 100})
        except RuntimeError:
            "Observable already registered."

    def _load(self):
        self._trajectory = readdy.Trajectory(self._readdy_file)
        self.times, self.types, self.ids, self.positions = self._trajectory.read_observable_particles()
        # self.times, self.types, self.ids, self.positions = self._trajectory.read_observable_particles("velocity_analysis")

        if self._stride is None:
            self._stride = self.times[1] - self.times[0]

    def _restride_trajectory(self, _new_stride):
        times = self.times[::_new_stride]
        types = self.types[::_new_stride]
        ids = self.ids[::_new_stride]
        positions = self.positions[::_new_stride]
        return times, types, ids, positions

    def run(self, particle_type: str, **kwargs):
        """ Calculates the velocity for the specified particle type. """
        _min_n_frames = kwargs.get("min_n_frames", 5)
        _restride = kwargs.get("restride", None)

        if self.times is None:
            try:
                self._load()
            except Exception as e:
                print(f"Error loading trajectory: {e}")
                return None

        if isinstance(_restride, int) and _restride > self._stride and _restride is not None:
            times, types, ids, positions = self._restride_trajectory(_restride)
        else:
            times = self.times
            types = self.types
            ids = self.ids
            positions = self.positions

        p_uids = set()
        for frame_ids in range(len(ids)):
            for i in range(len(ids[frame_ids])):
                p_uid = ids[frame_ids][i]
                p_type = types[frame_ids][i]
                if self._trajectory.species_name(p_type) == particle_type:
                    p_uids.add(p_uid)
        p_uids = list(p_uids)

        ptype_data = {key: {"times": [], "positions": [], "uids": [], "types": []} for key in p_uids}

        for frame_ids in range(len(ids)):
            for i in range(len(ids[frame_ids])):
                p_uid = ids[frame_ids][i]
                p_type = types[frame_ids][i]
                p_pos = positions[frame_ids][i]
                p_time = times[frame_ids]
                if p_uid in p_uids and self._trajectory.species_name(p_type) == particle_type:
                    ptype_data[p_uid]["positions"].append(p_pos)
                    ptype_data[p_uid]["times"].append(p_time)
                    ptype_data[p_uid]["uids"].append(p_uid)
                    ptype_data[p_uid]["types"].append(p_type)

        uid_list = list(ptype_data.keys())
        for uid in uid_list:
            if len(ptype_data[uid]["positions"]) < _min_n_frames:
                del ptype_data[uid]

        # Calculating based on accumulated distances
        for p_uid, p_data in ptype_data.items():
            d_time = (p_data["times"][-1] - p_data["times"][0])
            if d_time == 0:
                p_data["mean_velocity"] = 0.
                continue
            else:
                d_pos = 0.
                for i in range(1, len(p_data["positions"])):
                    d_pos += np.linalg.norm(np.array(p_data["positions"][i]) - np.array(p_data["positions"][i - 1]))
                v = d_pos / (d_time * self._timestep) if d_time > 0 else 0.
                v *= (ut.Unit(self._length_unit) / ut.Unit(self._time_unit))
                p_data["mean_velocity"] = v.magnitude
        return ptype_data