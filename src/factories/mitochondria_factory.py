from typing import Optional

from src.core.model import Model
from src.core.model_utils import ModelUtils
from src.core.data_loader import DataLoader
from src.factories.base_factory import Factory

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
    data_dir = "/Users/earkfeld/PycharmProjects/mitosim/data/mitosim_dataset_v4/control/cell_1"
    dl = DataLoader(data_dir)
    factory = MitochondriaFactory(data_loader=dl)
    model = factory.run()
    print(model)

