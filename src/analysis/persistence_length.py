import numpy as np
import igraph as ig
import scipy.optimize
from src.analysis import TopologyGraphs
from dataclasses import dataclass

ut = readdy.units

class PersistenceLength:
    """
    Calculate the persistence length for polymer chains in a ReaDDy simulation.

    Persistence length is calculated by measuring the autocorrelation of bond vectors
    along polymer chains and fitting an exponential decay to determine the characteristic
    decorrelation length.

    Parameters
    ----------
    trajectory_file : str
        Path to the ReaDDy trajectory file.
    particle_type : str
        Type of particle species to analyze.
    **kwargs
        Additional keyword arguments for the TopologyGraphs class.
    """
    @dataclass
    class Results:
        """Class to store the results of persistence length calculations."""
        bond_autocorrelation: np.ndarray = None
        lb: float = None
        lp: float = None
        x: np.ndarray = None
        fit: np.ndarray = None

    def __init__(self, trajectory_file, particle_type, **kwargs):
        self.results = self.Results()  # Initialize Results instance

        tg_util = TopologyGraphs(trajectory_file, **kwargs)
        tg_util.run(particle_type, as_list=True)

        self.frame_graphs = tg_util.results[particle_type]
        self.n_frames = len(self.frame_graphs)
        self.n_chains = len(self.frame_graphs[0])
        self.chain_length = self.frame_graphs[0][0].vcount()

        for frame in self.frame_graphs:
            for g in frame:
                if g.vcount() != self.chain_length:
                    raise ValueError("All chains must have the same length.")
        self._results = np.zeros(self.chain_length - 1, dtype=np.float32)

    def run(self):
        """Run the persistence length analysis across all frames."""
        for gs_frame in self.frame_graphs:
            for g in gs_frame:
                coordinates = np.array(g.vs["coordinate"])
                self._single_frame(coordinates)
        self._conclude()

    def _single_frame(self, coordinates):
        """Process a single frame to calculate bond vectors and their correlations."""
        vecs = coordinates[1:] - coordinates[:-1]
        vecs /= np.linalg.norm(vecs, axis=1)[:, None]
        inner_pr = np.inner(vecs, vecs)
        for i in range(self.chain_length - 1):
            self._results[:(self.chain_length - 1) - i] += inner_pr[i, i:]

    def _conclude(self):
        """Finalize the persistence length calculation by normalizing and fitting decay."""
        norm = np.linspace(self.chain_length - 1, 1, self.chain_length - 1)
        norm *= self.n_chains * self.n_frames
        self.results.bond_autocorrelation = self._results / norm
        self._calc_bond_length()
        self._perform_fit()

    def _calc_bond_length(self):
        """Calculate the average bond length over all frames and chains."""
        bond_lengths = []
        for gs_frame in self.frame_graphs:
            for g in gs_frame:
                edges = np.array(g.get_edgelist())
                for e in edges:
                    coord1 = np.array(g.vs[e[0]]["coordinate"])
                    coord2 = np.array(g.vs[e[1]]["coordinate"])
                    bond_lengths.append(np.linalg.norm(coord1 - coord2))
        self.results.lb = np.mean(bond_lengths)

    def _perform_fit(self):
        """Fit the bond autocorrelation data to an exponential decay."""
        x = self.results.lb * np.arange(len(self.results.bond_autocorrelation))
        self.results.x = x
        self.results.lp = self.fit_exponential_decay(x, self.results.bond_autocorrelation)
        self.results.fit = np.exp(-x / self.results.lp)

    def plot(self, ax=None):
        """Plot the bond autocorrelation and the exponential fit."""
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.results.x, self.results.bond_autocorrelation, 'ro', label='Result')
        ax.plot(self.results.x, self.results.fit, label='Fit')
        ax.set_xlabel(r'x')
        ax.set_ylabel(r'$C(x)$')
        ax.legend(loc='best')
        return ax

    def fit_exponential_decay(self, x, y):
        """
        Fit an exponential decay to the data.

        Parameters
        ----------
        x, y : array_like
            Arrays of data.

        Returns
        -------
        float
            Coefficient for the decay function.
        """
        def expfunc(x, a):
            return np.exp(-x / a)

        a, _ = scipy.optimize.curve_fit(expfunc, x, y)
        return a[0]

