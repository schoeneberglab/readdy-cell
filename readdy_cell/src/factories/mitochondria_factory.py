from typing import Optional

import numpy as np

from src.core.model import Model
from src.core.model_utils import ModelUtils
from src.core.data_loader import DataLoader
from src.factories.base_factory import Factory

def write_bild_spheres(coords, filename="output.bild", radius=0.15, color=(0, 1, 0)):
    """
    Write a .bild file to display spheres at given 3D coordinates.

    Parameters:
    - coords: iterable of (x, y, z) coordinate tuples
    - filename: output .bild file path
    - radius: radius of each sphere
    - color: RGB tuple, values between 0 and 1
    """
    with open(filename, 'w') as f:
        f.write(".color {:.3f} {:.3f} {:.3f}\n".format(*color))
        for coord in coords:
            x, y, z = coord
            f.write(".sphere {:.3f} {:.3f} {:.3f} {:.3f}\n".format(x, y, z, radius))

class MitochondriaFactory(Factory, ModelUtils):
    """Factory class for creating mitochondria model objects."""
    DEFAULT_TOPOLOGY_TYPE = "Mitochondria"
    DEFAULT_PARTICLE_TYPE = "mitochondria"
    DEFAULT_PARAMETERS = {
        "tags": ["boundary"],
        "flags": [],
        "radius": [0.15],
        "diffusion_constant": [0.0056]
    }
    DEFAULT_VISUALIZATION_PROPERTIES = {
        "radii": {"all": DEFAULT_PARAMETERS["radius"]},
        "display_types" : {"all": "SPHERE"},
        "viz_types": {"all": 1000.0},
        "colors": {"all": "green"},
        "url": ""
    }

    def __init__(self, data_loader: DataLoader, model: Optional[Model] = None, *args, **kwargs):
        assert data_loader is not None, "Please provide a MitoTNTDataLoader object"
        super().__init__(data_loader, *args, **kwargs)
        self.data_loader = data_loader
        self.model = model if model else self._get_default_model()

        self._graph_type = "full_graphs"
        self._frame_index = 0

    @property
    def graph_type(self):
        return self._graph_type

    @graph_type.setter
    def graph_type(self, value):
        assert isinstance(value, str), "Graph type must be a string."
        self._graph_type = value

    @property
    def frame_index(self):
        return self._frame_index

    def run(self, *args, **kwargs) -> Model:
        """ Construct and return a model."""
        raw_graphs = self.data_loader['tracking_inputs'][self._graph_type]
        g = raw_graphs[self._frame_index]
        self.model.data = g.decompose()
        return self.model

if __name__ == "__main__":
    data_dir = "/Users/earkfeld/Projects/mitosim/data/mitosim_dataset_v6/nocodazole_60min/cell_3"
    dl = DataLoader(data_dir)
    factory = MitochondriaFactory(data_loader=dl)
    model = factory.run()
    print(model)

    gs = model.data
    coords = np.vstack([g.vs["coordinate"] for g in gs])
    write_bild_spheres(coords, filename="noco60_c3_f0.bild", radius=0.15, color=(0, 1, 0))
