import readdy
import numpy as np
import igraph as ig
from dataclasses import dataclass, field
from typing import Union, List, Optional

ut = readdy.units

@dataclass
class TopologyGraphs:
    trajectory_file: Optional[str] = field(default=None)
    timestep: Optional[Union[float, ut.Quantity]] = field(default=5.e-3 * ut.s)
    _all_topology_graphs: Optional[list] = field(default_factory=list, repr=False)
    _trajectory: readdy.Trajectory = None
    _particles: list = None
    _topologies: list = None
    _times: np.ndarray = None
    _types: np.ndarray = None
    _ids: np.ndarray = None
    _pos: np.ndarray = None
    _stride: int = None
    _stride_time: Union[float, ut.Quantity] = None
    _restride_time: Union[int, float, ut.Quantity] = None
    _restride_interval: int = None
    _results: dict = field(default_factory=dict)

    def __post_init__(self):
        self._load()
        if self._restride_time is not None:
            self._calculate_restride_interval()
        self._setup()

    def _load(self):
        self._trajectory = readdy.Trajectory(self.trajectory_file)
        self._times, self._topologies = self._trajectory.read_observable_topologies()
        _, self._types, self._ids, self._pos = self._trajectory.read_observable_particles()
        self._particles = self._trajectory.read()
        self._stride = int(self._times[1] - self._times[0])
        self._stride_time = self._stride * self.timestep
        self._times = np.array(self._times) * self.timestep
        self._types = np.array(self._types)
        self._ids = np.array(self._ids)
        self._pos = np.array(self._pos)

    def _setup(self):
        """ Set up the topology graphs for all frames. """
        if self._all_topology_graphs:
            self._results["all_topology_graphs"] = self._all_topology_graphs
        else:
            self._all_topology_graphs = []
            if self._restride_interval is not None:
                # frame_indices = np.arange(0, len(self._topologies), self._restride_interval)
                frame_indices = np.arange(0, len(self._trajectory), self._restride_interval)
            else:
                # frame_indices = np.arange(0, len(self._topologies))
                frame_indices = np.arange(0, len(self._trajectory))

            for i in frame_indices:
                frame_particles = self._particles[i]
                frame_topologies = self._topologies[i]
                frame_graphs = []
                for top in frame_topologies:
                    vs_top = top.particles
                    es_top = top.edges

                    ptypes = [frame_particles[v_id].type for v_id in vs_top]
                    ppos = np.vstack([frame_particles[v_id].position for v_id in vs_top])
                    puids = [frame_particles[v_id].id for v_id in vs_top]

                    g_top = ig.Graph(n=len(vs_top), edges=es_top)
                    g_top.vs["uid"] = puids
                    g_top.vs["type"] = ptypes
                    g_top.vs["coordinate"] = ppos.tolist()  # Convert to list for compatibility

                    frame_graphs.append(g_top)
                self._all_topology_graphs.append(frame_graphs)
            self._results["all_topology_graphs"] = self._all_topology_graphs

    # TODO: Update this to filter based on multiple particle type strings
    # ORIGINAL
    def run(self,
            particle_types: Union[str, List[str]] = None,
            resort: bool = False,
            exact_match=False,
            exclude_types=[]):
        """ Extract the topology graphs for a given particle type. """
        if isinstance(particle_types, str):
            particle_types = [particle_types]

        for particle_type in particle_types:
            all_frame_graphs = self._results["all_topology_graphs"]
            particle_graphs = np.empty(shape=len(all_frame_graphs), dtype=object)

            for t, frame_graphs in enumerate(all_frame_graphs):
                gs_frame = []
                for g in frame_graphs:
                    n_vs = len(g.vs)
                    vs_remove = []

                    for v in g.vs:
                        vtype = v["type"]
                        if vtype in exclude_types:
                            vs_remove.append(v.index)
                            continue
                        if exact_match:
                            if particle_type != vtype:
                                vs_remove.append(v.index)
                        else:
                            if particle_type not in vtype:
                                vs_remove.append(v.index)
                    if len(vs_remove) >= n_vs:
                        continue
                    else:
                        g.delete_vertices(vs_remove)
                        g.simplify()
                        gs_frame.append(g)
                particle_graphs[t] = np.array(gs_frame)

            if resort:
                self._results[particle_type] = self._order_topologies_by_uid(particle_graphs)
            else:
                self._results[particle_type] = particle_graphs


    # TESTING; STATUS: TBD
    # def run(self,
    #         particle_types: Union[str, List[str]] = None,
    #         resort: bool = False,
    #         exact_match=False,
    #         exclude_types=[]):
    #     """ Extract the topology graphs for a given particle type. """
    #     if isinstance(particle_types, str):
    #         particle_types = [particle_types]
    #
    #     # valid_dict_template = {ptype: False for ptype in particle_types}
    #
    #
    #     all_frame_graphs = self._results["all_topology_graphs"]
    #     particle_graphs = np.empty(shape=len(all_frame_graphs), dtype=object)
    #
    #     for t, frame_graphs in enumerate(all_frame_graphs):
    #         gs_frame = []
    #         for g in frame_graphs:
    #             n_vs = len(g.vs)
    #             vs_remove = []
    #
    #             for v in g.vs:
    #                 vtype = v["type"]
    #                 if vtype in exclude_types:
    #                     vs_remove.append(v.index)
    #                     continue
    #                 if exact_match:
    #                     # Check if the vtype is in the particle type list
    #                     if vtype != any(particle_types):
    #                         vs_remove.append(v.index)
    #                 else:
    #                     is_valid = False
    #                     for ptype in particle_types:
    #                         if ptype in vtype:
    #                             is_valid = True
    #                             break
    #
    #                     if not is_valid:
    #                         vs_remove.append(v.index)
    #
    #             if len(vs_remove) >= n_vs:
    #                 continue
    #             else:
    #                 g.delete_vertices(vs_remove)
    #                 g.simplify()
    #                 gs_frame.append(g)
    #         particle_graphs[t] = np.array(gs_frame)
    #
    #     multiparticle = False if len(particle_types) == 1 else True
    #     res_key = particle_types[0] if not multiparticle else "topologies"
    #     if resort:
    #         self._results[res_key] = self._order_topologies_by_uid(particle_graphs)
    #     else:
    #         self._results[res_key] = particle_graphs

    @staticmethod
    def _order_topologies_by_uid(graphs):
        """ Reorders topology graphs according to their order in the first frame for topologies reacting only in heterogeneous fashion."""
        anchor_gs_uids = [graphs[0][i].vs["uid"] for i in range(len(graphs[0]))]
        ordered_topology_graphs = np.empty(shape=(len(graphs), len(anchor_gs_uids)), dtype=object)
        for t, gs_frame in enumerate(graphs):
            for i, anchor_uids in enumerate(anchor_gs_uids):
                for g in gs_frame:
                    if g.vs["uid"] == anchor_uids:
                        ordered_topology_graphs[t][i] = g
                        break
                    else:
                        continue
        return ordered_topology_graphs

    def _get_topology_uids(self):
        topology_uids = set()
        for frame_graphs in self.results["all_topology_graphs"]:
            for g in frame_graphs:
                uid = tuple(sorted(g.vs["uid"]))
                topology_uids.add(uid)
        return topology_uids

    # def get_unique_topology_trajectories(self,
    #                                      particle_type: str):
    #     """ Extracts all unique topologies for a given particle type. """
    #     assert self.results[particle_type] is not None, \
    #         "Please run the analysis for the given particle type first."
    #
    #     tuids = set()
    #     template_dict = {"start_frame": np.nan,
    #                      "graph_trajectory": []}
    #
    #     # Set up the trajectory dictionary
    #     ttraj = {uid: template_dict for uid in tuids}
    #     for i, gs_frame in enumerate(self.results[particle_type]):
    #         for g in gs_frame:
    #             uid = tuple(sorted(g.vs["uid"]))
    #             if ttraj[uid]["start_frame"] == np.nan:
    #                 ttraj[uid]["start_frame"] = i
    #             ttraj[uid]["graph_trajectory"].append(g)
    #     return ttraj

    # Testing 1
    # def get_unique_topology_trajectories(self,
    #                                      particle_type: str):
    #     """ Extracts all unique topologies for a given particle type. """
    #     assert self.results[particle_type] is not None, \
    #         "Please run the analysis for the given particle type first."
    #
    #     # Set up the trajectory dictionary
    #     ttraj = {}
    #     open_tuids = set()
    #     closed_tuids = set()
    #     for t, gs_frame in enumerate(self.results[particle_type]):
    #         visited_tuids = set()
    #         for g in gs_frame:
    #             uid = tuple(sorted(g.vs["uid"]))
    #             # Case 1: new uid, new instance --> create new uid entry + instance
    #             if uid not in ttraj.keys():
    #                 ttraj[uid] = [{"start_frame": t,
    #                               "end_frame": np.nan,
    #                               "graph_trajectory": []}]
    #                 open_tuids.add(uid)
    #
    #             # Case 2: old uid, new instance --> create new instance under existing uid entry
    #             if uid in ttraj.keys() and uid in closed_tuids:
    #                 ttraj[uid].append({"start_frame": t,
    #                                     "end_frame": np.nan,
    #                                     "graph_trajectory": []})
    #                 open_tuids.add(uid)
    #                 closed_tuids.remove(uid)
    #
    #             # Case 3: current uid, current instance --> update current instance
    #             if uid in ttraj.keys and uid in open_tuids:
    #                 ttraj[uid][-1]["graph_trajectory"].append(g)
    #                 visited_tuids.add(uid)
    #
    #         # Close all open tuids that were not visited in the current frame
    #         tuids_to_close = open_tuids - visited_tuids
    #         for uid in tuids_to_close:
    #             # Set the index of the last frame it was seen
    #             ttraj[uid][-1]["end_frame"] = t
    #             closed_tuids.add(uid)
    #             open_tuids.remove(uid)
    #     return ttraj


    # # FUNCTIONAL VERSION
    # def get_unique_topology_trajectories(self, particle_type: str, equilibration_fraction: float):
    #     """ Extracts all unique topologies for a given particle type. """
    #     assert self.results[particle_type] is not None, \
    #         f"Results for particle type '{particle_type}' are missing. Please run the analysis first."
    #
    #     global_start_idx = 0
    #     if equilibration_fraction > 0.0:
    #         global_start_idx = int(equilibration_fraction * len(self._times))
    #
    #     # Set up the trajectory dictionary
    #     ttraj = {}
    #     open_tuids, closed_tuids = set(), set()
    #     for t, gs_frame in enumerate(self.results[particle_type]):
    #         if t < global_start_idx:
    #             continue
    #
    #         visited_tuids = set()
    #         for g in gs_frame:
    #             if g is None:
    #                 continue
    #             uid = tuple(sorted(g.vs["uid"]))
    #
    #             # Case 1 & 2: Handle new uid or closed uid
    #             if uid not in ttraj:
    #                 ttraj[uid] = [{"start_frame": t, "graphs": [], "coordinate": []}]
    #                 open_tuids.add(uid)
    #             elif uid in closed_tuids:
    #                 ttraj[uid].append({"start_frame": t, "graphs": [], "coordinate": []})
    #                 open_tuids.add(uid)
    #                 closed_tuids.remove(uid)
    #
    #             # Case 3: Update current instance
    #             if uid in open_tuids:
    #                 ttraj[uid][-1]["graphs"].append(g)
    #                 coordinates = np.vstack(g.vs["coordinate"])
    #                 mean_coordinate = np.mean(coordinates, axis=0)
    #                 ttraj[uid][-1]["coordinate"].append(mean_coordinate)
    #                 visited_tuids.add(uid)
    #
    #         # Close all open tuids that were not visited in the current frame
    #         tuids_to_close = open_tuids - visited_tuids
    #         for uid in tuids_to_close:
    #             closed_tuids.add(uid)
    #             open_tuids.remove(uid)
    #
    #     # Reformat the coordinate trajectory to a numpy array
    #     for uid, trajs in ttraj.items():
    #         for i, traj in enumerate(trajs):
    #             ttraj[uid][i]["coordinate"] = np.vstack(traj["coordinate"])
    #
    #     return ttraj

    # TESTING 2
    def get_unique_topology_trajectories(self, particle_type: str, equilibration_fraction: float):
        """ Extracts all unique topologies for a given particle type. """
        if self.results[particle_type] is None:
            self.run(particle_types=particle_type)

        # print("Getting unique topology trajectories...")
        global_start_idx = 0
        if equilibration_fraction > 0.0:
            global_start_idx = int(equilibration_fraction * len(self._times))
            times = self._times[global_start_idx:] - self._times[global_start_idx]
            self._times = times

        # Set up the trajectory dictionary
        ttraj = {}
        open_tuids, closed_tuids = set(), set()
        for t, gs_frame in enumerate(self.results[particle_type]):
            if t < global_start_idx:
                continue

            visited_tuids = set()
            for g in gs_frame:
                if g is None:
                    continue
                uid = tuple(sorted(g.vs["uid"]))

                # Case 1 & 2: Handle new uid or closed uid
                if uid not in ttraj:
                    ttraj[uid] = [{"start_frame": t - global_start_idx, "graphs": [], "coordinate": []}]
                    open_tuids.add(uid)
                elif uid in closed_tuids:
                    ttraj[uid].append({"start_frame": t - global_start_idx, "graphs": [], "coordinate": []})
                    open_tuids.add(uid)
                    closed_tuids.remove(uid)

                # Case 3: Update current instance
                if uid in open_tuids:
                    ttraj[uid][-1]["graphs"].append(g)
                    coordinates = np.vstack(g.vs["coordinate"])
                    mean_coordinate = np.mean(coordinates, axis=0)
                    ttraj[uid][-1]["coordinate"].append(mean_coordinate)
                    visited_tuids.add(uid)

            # Close all open tuids that were not visited in the current frame
            tuids_to_close = open_tuids - visited_tuids
            for uid in tuids_to_close:
                closed_tuids.add(uid)
                open_tuids.remove(uid)

        # Reformat the coordinate trajectory to a numpy array
        for uid, trajs in ttraj.items():
            for i, traj in enumerate(trajs):
                ttraj[uid][i]["coordinate"] = np.vstack(traj["coordinate"])
        return ttraj


    # def run(self,
    #         particle_types: Union[str, List[str]] = None,
    #         sorted: bool = True,
    #         exact_match=False,
    #         exclude_types=[]):
    #     """ Extract the topology graphs for a given particle type. """
    #
    #     if isinstance(particle_types, str):
    #         particle_types = [particle_types]
    #
    #     all_frame_graphs = self._results["all_topology_graphs"]
    #
    #     for particle_type in particle_types:
    #         particle_graphs = []
    #
    #         for frame_graphs in all_frame_graphs:
    #             gs_frame = []
    #             for g in frame_graphs:
    #                 vs_remove = [
    #                     v.index for v in g.vs if (
    #                             v["type"] in exclude_types or
    #                             (exact_match and particle_type != v["type"]) or
    #                             (not exact_match and particle_type not in v["type"])
    #                     )
    #                 ]
    #
    #                 if len(vs_remove) < len(g.vs):  # Skip entirely removed graphs
    #                     g.delete_vertices(vs_remove)
    #                     g.simplify()
    #                     gs_frame.append(g)
    #
    #             particle_graphs.append(gs_frame)
    #
    #         particle_graphs = np.array(particle_graphs, dtype=object)
    #
    #         if sorted:
    #             anchor_gs_uids = [sorted(g.vs["uid"]) for g in particle_graphs[0]]
    #             ordered_topology_graphs = np.empty(
    #                 (len(particle_graphs), len(anchor_gs_uids)), dtype=object
    #             )
    #
    #             for t, gs_frame in enumerate(particle_graphs):
    #                 uid_to_graph = {tuple(sorted(g.vs["uid"])): g for g in gs_frame}
    #                 for i, anchor_uids in enumerate(anchor_gs_uids):
    #                     ordered_topology_graphs[t, i] = uid_to_graph.get(tuple(anchor_uids))
    #
    #             self._results[particle_type] = ordered_topology_graphs
    #         else:
    #             self._results[particle_type] = particle_graphs

    def _calculate_restride_interval(self):
        """ Calculate the interval to restride the results to achieve a desired time between frames. """
        if not isinstance(self._restride_time, ut.Quantity):
            print("No units provided; Assuming time unit of seconds.")
            self._restride_time = float(self._restride_time) * ut.s

        assert self._restride_time > 0 and self._restride_time > self._stride_time, \
            "Please provide a positive stride time that is greater than the original stride."

        self._restride_interval = int(self._restride_time / self._stride_time)

    def get_results_with_stride(self, restride_time: Union[int, float, ut.Quantity] = None):
        original_stride_time = self._stride * self.timestep

        assert restride_time is not None, \
            "Please provide a time to stride the results."

        if not isinstance(restride_time, ut.Quantity):
            print("No units provided; Assuming units seconds.")
            restride_time = float(restride_time) * ut.s

        assert restride_time > 0 and restride_time > (self._stride * self.timestep), \
            "Please provide a positive stride time that is greater than the original stride."

        restride_interval = int(restride_time / original_stride_time)
        restrided_results = {}
        for key, value in self._results.items():
            if key == "all_topology_graphs":
                continue
            restrided_results[key] = value[::restride_interval]
        return restrided_results

    @property
    def results(self):
        return self._results


# Example Usage
if __name__ == "__main__":
    traj_file = "/home/earkfeld/PycharmProjects/mitosim/data/trajectories/production_runs/t1/control/cell_0/control_c0_0.h5"
    tg = TopologyGraphs(trajectory_file=traj_file, timestep=5.e-3 * ut.s)
    tg.run(particle_type="mitochondria")
    tg.get_unique_topology_trajectories(particle_type="mitochondria", equilibration_fraction=0.5)

