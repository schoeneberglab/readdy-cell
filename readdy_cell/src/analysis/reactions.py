import numpy as np
import igraph as ig
from scipy.linalg import lstsq
from typing import Union, List, Dict
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

from src.analysis import TopologyGraphs
from src.analysis.base import AnalysisBase
import readdy

ut = readdy.units
# TODO: Set up so restriding is performed after setup to allow for reuse with different restriding time(s)

class Reactions(AnalysisBase):
    def __init__(self, trajectory_file, timestep, **kwargs):
        super().__init__()
        self._trajectory_file = trajectory_file
        self._trajectory = readdy.Trajectory(trajectory_file)
        self._timestep = timestep
        self._particle_type = kwargs.get("particle_type", None)
        self._frame_interval = kwargs.get("frame_interval", 1)
        self._restride_time = kwargs.get("restride_time", None)
        self._equilibration_fraction = kwargs.get("equilibration_fraction", 0)

        meta_data = kwargs.get("meta_data", None)
        if meta_data and isinstance(meta_data, dict):
            for key, value in meta_data.items():
                setattr(self._results, key, value)

    def run(self):
        if self._restride_time:
            self._frame_interval = int(self._restride_time / self._timestep)

        self._calculate_mean_sizes()
        self._get_reaction_counts()
        self._get_degree_counts()

    def save(self, filename: str):
        pass

    def _get_reaction_counts(self):
        times, counts = self._trajectory.read_observable_reaction_counts()

        fusion_counts = []
        for key in counts['spatial_topology_reactions'].keys():
            if "Fusion Reaction 1" in key:
                fusion_counts.append(counts['spatial_topology_reactions'][key])

        fission_counts = []
        for key in counts['structural_topology_reactions'].keys():
            if "Fission Reaction 1" in key:
                fission_counts.append(counts['structural_topology_reactions'][key])

        fusion_counts = np.array(fusion_counts)
        fission_counts = np.array(fission_counts)

        fusion_counts = np.sum(fusion_counts, axis=0)
        fission_counts = np.sum(fission_counts, axis=0)

        self._results.raw_data.fusion_counts = fusion_counts
        self._results.raw_data.fission_counts = fission_counts
        self._results.raw_data.times = times

        fusion_counts_resampled = []
        fission_counts_resampled = []
        for i in range(0, len(fusion_counts), int(self._frame_interval)):
            resampled_fusion_count = np.sum(fusion_counts[i:i + int(self._frame_interval)], axis=0)
            resampled_fission_count = np.sum(fission_counts[i:i + int(self._frame_interval)], axis=0)
            fusion_counts_resampled.append(resampled_fusion_count)
            fission_counts_resampled.append(resampled_fission_count)

        self._results.frames = np.arange(0, len(fusion_counts_resampled))
        self._results.times = np.arange(0, len(fusion_counts_resampled)) * int(self._frame_interval * self._timestep)

        self._results.fusion_counts = np.array(fusion_counts_resampled)
        self._results.fission_counts = np.array(fission_counts_resampled)

    def _calculate_mean_sizes(self):
        tg = TopologyGraphs(self._trajectory_file, timestep=self._timestep)
        tg.run("mitochondria")
        # gs = tg.results["all_topology_graphs"]
        gs = tg.results["mitochondria"]

        self._results.raw_data.topology_graphs = gs

        gs = gs[int(self._equilibration_fraction * len(gs)):]
        indices = np.arange(0, len(gs), self._frame_interval)
        gs = [gs[i] for i in indices]

        sizes = []
        mean_sizes = []
        top_counts = []
        for i, g_frame in enumerate(gs):
            mean_sizes.append(0)
            top_counts.append(0)
            frame_sizes = []
            for j, g in enumerate(g_frame):
                top_size = 0
                for v in g.vs:
                    if "mitochondria" in v["type"]:
                        top_size += 1
                mean_sizes[-1] += top_size
                frame_sizes.append(top_size)
                if top_size > 0:
                    top_counts[-1] += 1
            if top_counts[-1] > 0:
                mean_sizes[-1] /= top_counts[-1]
            sizes.append(frame_sizes)
        self._results.mean_sizes = mean_sizes
        self._results.sizes = sizes

    def _get_degree_counts(self):
        gs = self._results.raw_data.topology_graphs
        degree_counts = []
        for g_frame in gs:
            frame_degree_counts = []
            for g in g_frame:
                vertex_types = np.array(g.vs["type"])
                adjacency_matrix = np.array(g.get_adjacency().data)
                mitochondria_indices = np.where(vertex_types == "mitochondria")[0]
                mitochondria_adj = adjacency_matrix[mitochondria_indices][:, mitochondria_indices]
                degrees = np.sum(mitochondria_adj, axis=1)
                frame_degree_counts.extend(degrees)
            degree_counts.append(frame_degree_counts)
        self._results.raw_data.degree_counts = degree_counts

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
    def restride_time(self):
        return self._restride_time

    @restride_time.setter
    def restride_time(self, value):
        self._restride_time = value

    @property
    def equilibration_fraction(self):
        return self._equilibration_fraction

    @equilibration_fraction.setter
    def equilibration_fraction(self, value):
        self._equilibration_fraction = value