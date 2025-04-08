from abc import ABC, abstractmethod
from typing import Any, Optional

from src.core.data_loader import DataLoader
from src.core.model import Model

# TODO: Add consolidate construction vars
class Factory(ABC):
    """Abstract Factory class for creating model objects."""
    DEFAULT_TOPOLOGY_TYPE = None
    DEFAULT_PARTICLE_TYPE = None
    DEFAULT_PARAMETERS = None
    DEFAULT_VISUALIZATION_PROPERTIES = None

    def __init__(self, data_loader: DataLoader, model: Optional[Model] = None, *args, **kwargs):
        assert data_loader is not None, "Please provide a DataLoader object"
        super().__init__(*args, **kwargs)
        self.data_loader = data_loader
        self.model = model if model else self._get_default_model()

    def _get_default_model(self) -> Model:
        """Returns a default model object."""
        return Model(
            topology_type=self.DEFAULT_TOPOLOGY_TYPE,
            particle_type=self.DEFAULT_PARTICLE_TYPE,
            parameters=self.DEFAULT_PARAMETERS,
            display_properties=self.DEFAULT_VISUALIZATION_PROPERTIES
        )

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """Constructs a model object."""
        pass