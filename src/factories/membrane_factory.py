import numpy as np
from typing import Optional

from src.core.data_loader import DataLoader
from src.core.model import Model
from src.core.model_utils import ModelUtils
from src.factories.base_factory import Factory


class MembraneFactory(Factory, ModelUtils):
    """Factory class for creating membrane model objects."""

    DEFAULT_TOPOLOGY_TYPE = None
    DEFAULT_PARTICLE_TYPE = "membrane"
    DEFAULT_PARAMETERS = {
        "tags": ["boundary"],
        "flags": [],
        "radius": [0.2],
        "diffusion_constant": [0.0]
    }
    DEFAULT_VISUALIZATION_PROPERTIES = {
        "display_types" : {"all": "SPHERE"},
        "viz_types": {"all": 1000.0},
        "colors": {"all": "red"},
        "url": ""
    }

    def __init__(self, data_loader: DataLoader, model: Optional[Model] = None, *args, **kwargs):
        assert data_loader is not None, "Please provide a MitoTNTDataLoader object"
        super().__init__(data_loader, *args, **kwargs)
        self.data_loader = data_loader
        self.model = model if model else self._get_default_model()

        # Factory attributes
        self._downsample_factor = None
        self._buffer_factor = 1.2
        self._dilation_array = None
        self._padding_array = None
        self._radius = 0.25

    @property
    def downsample_factor(self):
        return self._downsample_factor

    @downsample_factor.setter
    def downsample_factor(self, value):
        assert isinstance(value, int), "Downsample factor must be a float."
        self._downsample_factor = value

    @property
    def buffer_factor(self):
        return self._buffer_factor

    @buffer_factor.setter
    def buffer_factor(self, value):
        assert isinstance(value, float), "Buffer factor must be a float."
        self._buffer_factor = value

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        assert isinstance(value, float), "Radius must be a float."
        self.model.parameters["radius"] = [value]
        self._radius = value

    @property
    def dilation_array(self):
        return self._dilation_array

    @dilation_array.setter
    def dilation_array(self, value):
        assert isinstance(value, np.ndarray), "Dilation array must be a numpy array."
        self._dilation_array = value

    @property
    def padding_array(self):
        return self._padding_array

    @padding_array.setter
    def padding_array(self, value):
        self._padding_array = value

    # def _get_default_model(self):
    #     model = Model(topology_type=self.DEFAULT_TOPOLOGY_TYPE,
    #                   particle_type=self.DEFAULT_PARTICLE_TYPE,
    #                   parameters=self.DEFAULT_PARAMETERS)
    #     return model

    @staticmethod
    def _get_coordinates(edge_data, padding_array, voxel_scale):
        return (np.argwhere(edge_data > 0) - padding_array) * voxel_scale

    def run(self, *args, **kwargs) -> Model:
        """ Construct and return a model."""
        img = self.data_loader['membrane_mask'].copy()
        mask_data = self.format_mask(img)
        voxel_scale, voxel_units = self.data_loader.get_voxel_scale()

        if self._dilation_array is None:
            self._dilation_array = self._calculate_dilation_array(self._radius, voxel_scale)

        if self._padding_array is None:
            self._padding_array = 2 * self._dilation_array

        padded_mask = self._pad_mask_3d(mask_data, self._padding_array)
        dilated_mask = self._dilate_mask_3d(padded_mask, self._dilation_array)
        edge_data = self._canny_3d(dilated_mask)
        # Save the data as a black and white image using
        self.save_mask_as_binary_tif(edge_data)
        # self.save_image(edge_data, "edge_data.tif")
        # coordinates = (np.argwhere(edge_data > 0) - self._padding_array) * voxel_scale
        coordinates = MembraneFactory._get_coordinates(edge_data, self._padding_array, voxel_scale)

        if self._downsample_factor is None:
            self._downsample_factor = self.get_downsample_factor(coordinates, self._radius, self._buffer_factor)

        self.model.data = MembraneFactory._downsample_coordinates(coordinates, self._downsample_factor)
        return self.model

if __name__ == "__main__":
    data_dir = "/Users/earkfeld/PycharmProjects/mitosim/data/mitosim_dataset_v5/control/cell_1"
    dl = DataLoader(data_dir)
    factory = MembraneFactory(data_loader=dl)
    model = factory.run()
    print(type(factory.model.data))

