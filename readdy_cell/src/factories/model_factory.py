import os
import os.path as osp
import readdy
from readdy.api.experimental.action_factory import BreakConfig, ReactionConfig

import numpy as np
import yaml
import json
import pickle
import igraph as ig
from itertools import product

from src.core import DataLoader
from src.factories import *
from src.reactions import ActiveTransportEngine, ActiveTransportReactions, MitochondrialDynamics
from src.analysis import Velocity, TopologyGraphs
from src.visualization import SimulariumConverter
from main import CellSimulation

from rmm.utils import *

ut = readdy.units
np.random.seed(42)

class ModelFactory:
    def __init__(self, data_root, outdir, model_name, *args, **kwargs):
        print(f"Loading data from {data_root}")
        self._dl = DataLoader(data_root)
        self.models = {}
        self._models_unchanged = {}
        self.outdir = outdir
        self.model_name = model_name

        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

    def hide_membrane_fraction(self, fraction):
        """ Hides a fraction of the membrane model. """
        max_coord = np.max(self._membrane.data, axis=0)
        min_coord = np.min(self._membrane.data, axis=0)

        if fraction < 0:
            fraction = abs(fraction)
            threshold = min_coord[0] + fraction * (max_coord[0] - min_coord[0])
            self._membrane.data = self._membrane.data[self._membrane.data[:, 0] > threshold]
        else:
            threshold = min_coord[0] + fraction * (max_coord[0] - min_coord[0])
            self._membrane.data = self._membrane.data[self._membrane.data[:, 0] < threshold]

    def run(self):
        """ Generates all models and stores them in the models dictionary. """
        mt_filename = osp.join(self.outdir, f"{self.model_name}_microtubules")
        self._membrane = MembraneFactory(self._dl).run()
        self._nucleus = NucleusFactory(self._dl).run()
        self._mitochondria = MitochondriaFactory(self._dl).run()
        self._microtubules = MicrotubulesFactory(self._dl, nucleus=self._nucleus).run(mt_filename)

        centering_vector = -1 * (np.max(self._membrane.data, axis=0) - np.min(self._membrane.data, axis=0)) / 2
        self._membrane.center_model(centering_vector)
        self._nucleus.center_model(centering_vector)
        self._mitochondria.center_model(centering_vector)

        self.models = {
            "nucleus": self._nucleus,
            "membrane": self._membrane,
            "mitochondria": self._mitochondria,
            "microtubules": self._microtubules,
        }
        self._models_unchanged = self.models.copy()

    def load(self, filepath):
        """ Loads models from a pickle file. """
        with open(filepath, "rb") as f:
            self.models = pickle.load(f)
        self._models_unchanged = self.models.copy()
        print(f"Models loaded from: {filepath}")

    def reset_models(self):
        """ Resets the models to their original state. """
        self.models = self._models_unchanged.copy()

    def translate_model(self, model_name, translation_vector):
        """ Translates a specified model by a given vector. """
        if model_name in ["nucleus", "membrane"]:
            self.models[model_name].data += translation_vector
        elif model_name in ["mitochondria", "microtubules"]:
            g = self.models[model_name].data
            coords = np.array(g.vs["coordinates"])
            coords += translation_vector
            g.vs["coordinates"] = coords
        else:
            raise ValueError(f"Model '{model_name}' not recognized for translation.")
    
    def scale_model(self, model_name, scale_factor):
        """ Scales the specified model according to the provided factor. """
        if model_name in ["nucleus", "membrane"]:
            coords = self.models[model_name].data
            center = np.mean(coords, axis=0)
            coords -= center
            coords *= scale_factor
            coords += center
            self.models[model_name].data = coords
        elif model_name in ["mitochondria", "microtubules"]:
            g = self.models[model_name].data
            coords = np.array(g.vs["coordinates"])
            center = np.mean(coords, axis=0)
            coords -= center
            coords *= scale_factor
            coords += center
            g.vs["coordinates"] = coords
        else:
            raise ValueError(f"Model '{model_name}' not recognized for scaling.")

    def save(self):
        """ Saves the models to a pickle file. """
        outfile = osp.join(self.outdir, self.model_name + ".pkl" if not self.model_name.endswith(".pkl") else "")
        with open(outfile, "wb") as f:
            pickle.dump(self.models, f)
        print(f"Models saved to: {outfile}")

    def equilibrate(self, config="config_equilibrate.yaml", **kwargs):
        """ Runs a short simulation to visualize the model. """
        with open(config, "r") as f:
            cfg = yaml.safe_load(f)

        cfg[""]["n_steps"] = 1e3
        cfg[""]["stride"] = 10
        cfg["run_parameters"]["flags"]["enable_mitochondrial_dynamics"] = False
        cfg["run_parameters"]["io"]["outdir"] = kwargs.get("outdir", self.outdir)
        cfg["run_parameters"]["io"]["outfile"] = kwargs.get("outfile", self.model_name + "_equilbrate")

        csim = CellSimulation(cfg, models=self.models)
        csim.run(show_summary=False)

    def visualize(self, config="config_v3.yaml", **kwargs):
        """ Runs a short simulation to visualize the model. """
        with open(config, "r") as f:
            cfg = yaml.safe_load(f)

        cfg[""]["n_steps"] = 10
        cfg[""]["stride"] = 1
        cfg["run_parameters"]["io"]["outdir"] = kwargs.get("outdir", self.outdir)
        cfg["run_parameters"]["io"]["outfile"] = kwargs.get("outfile", self.model_name + "_vis")
        csim = CellSimulation(cfg, models=self.models)
        csim.run(show_summary=False)
        