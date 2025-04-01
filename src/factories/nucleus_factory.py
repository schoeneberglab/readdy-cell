import tqdm
import logging
import numpy as np
from typing import Optional
from src.core.data_loader import DataLoader
from src.core.model import Model
from src.core.model_utils import ModelUtils
from src.factories.base_factory import Factory

class NucleusFactory(Factory, ModelUtils):
    """ Factory class for constructing a Nucleus model. """

    DEFAULT_TOPOLOGY_TYPE = None
    DEFAULT_PARTICLE_TYPE = "nucleus"
    DEFAULT_PARAMETERS = {
        "tags": ["boundary"],
        "flags": [],
        "radius": [0.4],
        "diffusion_constant": [0.0]
    }
    DEFAULT_VISUALIZATION_PROPERTIES = {
        "radii": {"all": DEFAULT_PARAMETERS["radius"]},
        "display_types" : {"all": "SPHERE"},
        "viz_types": {"all": 1000.0},
        "colors": {"all": "blue"},
        "url": "",
    }

    def __init__(self, data_loader: DataLoader, model: Optional[Model] = None, *args, **kwargs):
        assert data_loader is not None, "Please provide a MitoTNTDataLoader object"
        super().__init__(data_loader, *args, **kwargs)
        self.data_loader = data_loader
        self.model = model if model else self._get_default_model()

        # Factory attributes
        self._downsample_factor = None
        self._buffer_factor = 1.0
        self._dilation_array = None
        # self._padding_array = None
        self._padding_array = np.array([1, 1, 1])
        self._radius = 0.2
        # self._scale_factors = np.array([0.95, 0.95, 0.95])

        self._scale_factors = np.array([0.9, 0.9, 0.8]) # Control C1 v6
        # self._scale_factors = np.array([0.8, 0.8, 0.7]) # Noco60 C1 params

        # self._scale_factors = np.array([0.95]*3)
        # self._scale_factors = kwargs.get("scale_factors", np.array([0.9, 0.9, 0.9]))

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

    @property
    def scale_factors(self):
        return self._scale_factors

    @scale_factors.setter
    def scale_factors(self, value):
        assert isinstance(value, np.ndarray), "Scale factors must be a numpy array."
        self._scale_factors = value

    def run(self) -> Model:
        """ Construct and return a model."""
        img = self.data_loader['nucleus_mask'].copy()
        mask_data = self.format_mask(img)
        voxel_scale, voxel_units = self.data_loader.get_voxel_scale()

        # if self._dilation_array is None:
        #     self._dilation_array = self._calculate_dilation_array(self._radius, voxel_scale)

        if self._padding_array is not None:
            # Pad the mask data
            mask_data = self._pad_mask_3d(mask_data, self._padding_array)
            edge_data = self._canny_3d(mask_data)
            self.save_mask_as_binary_tif(edge_data)
            coordinates = (np.argwhere(edge_data > 0) - self._padding_array) * voxel_scale
            coordinates = self.rescale(coordinates, self._scale_factors)
        else:
            # mask_data = self._pad_mask_3d(mask_data, self._padding_array)
            # mask_data = self._dilate_mask_3d(mask_data, self._dilation_array)
            edge_data = self._canny_3d(mask_data)
            # coordinates = (np.argwhere(edge_data > 0) - self._padding_array) * voxel_scale
            coordinates = (np.argwhere(edge_data > 0) * voxel_scale)

            # Rescale the coordinates while
            coordinates = self.rescale(coordinates, self._scale_factors)

        if self._downsample_factor is None:
            self._downsample_factor = self.get_downsample_factor(coordinates, self._radius, self._buffer_factor)

        self.model.data = self._downsample_coordinates(coordinates, self._downsample_factor)
        return self.model

if __name__ == "__main__":
    data_dir = "/home/earkfeld/PycharmProjects/mitosim/data/mitosim_dataset_v5/control/cell_1"
    dl = DataLoader(data_dir)
    factory = NucleusFactory(dl)
    factory.run()
    print(factory.model.data)
