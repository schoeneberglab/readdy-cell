import os
import re
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import skimage.io as ski_io
from src.utils.io import get_mitotnt_configfile, get_project_root
import pprint

pp = pprint.PrettyPrinter(indent=4)


class DataLoader:
    """ Class for parsing and loading MitoTNT data. """

    DEFAULT_PATHS = {
        "nucleus_mask": "nucleus.tif",
        "membrane_mask": "membrane.tif",
        "microtubules": "microtubule/frame_{frame}/frame_{frame}.tif",
        "microtubules_segmented": "microtubule/frame_{frame}/frame_{frame}_segmented.tif",
        "mitograph_config": "mitograph/frame_{frame}/mitograph.config",
        "mitograph_report": "mitograph/frame_{frame}/frame_{frame}.mitograph",
        "fragment_diffusivity": "mitotnt/post_analysis/motility/fragment_diffusivity.csv",
        "segment_diffusivity": "mitotnt/post_analysis/motility/segment_diffusivity.csv",
        "node_diffusivity": "mitotnt/post_analysis/motility/node_diffusivity.csv",
        "remodeling_events": "mitotnt/post_analysis/remodeling/remodeling_events.csv",
        "tracking_inputs": "mitotnt/tracking_inputs/tracking_inputs.npz",
        "final_node_tracks": "mitotnt/tracking_outputs/final_node_tracks.csv",
        "frametoframe_tracking_outputs": "mitotnt/tracking_outputs/frametoframe_tracking_outputs.npz",
        "mt_fragment_diffusivity": "mitotnt_microtubule/post_analysis/motility/fragment_diffusivity.csv",
        "mt_segment_diffusivity": "mitotnt_microtubule/post_analysis/motility/segment_diffusivity.csv",
        "mt_node_diffusivity": "mitotnt_microtubule/post_analysis/motility/node_diffusivity.csv",
        "mt_remodeling_events": "mitotnt_microtubule/post_analysis/remodeling/remodeling_events.csv",
        "mt_tracking_inputs": "mitotnt_microtubule/tracking_inputs/tracking_inputs.npz",
        "mt_final_node_tracks": "mitotnt_microtubule/tracking_outputs/final_node_tracks.csv",
        "mt_frametoframe_tracking_outputs": "mitotnt_microtubule/tracking_outputs/frametoframe_tracking_outputs.npz",
    }

    def __init__(self, datadir, *args, **kwargs):
        """
        Initialize the DataLoader with a data directory and specific frame number.

        Parameters:
        - datadir (str): Path to the data directory.
        - frame (int): The specific frame number to load data for (default is 0).
        """
        assert datadir is not None, "Please provide a valid path to the directory containing the MitoTNT data"
        self.datadir = datadir
        self.root_dir = get_project_root()
        self.configfile = kwargs.get('configfile', get_mitotnt_configfile())
        self.from_root = kwargs.get('from_root', True)

        if self.from_root:
            self.datadir = self.root_dir / Path(datadir)

        self.pathdict = {}

        self._frame = kwargs.get("frame", 0)
        self._setup_path_dict()

    # @property
    # def frame(self):
    #     return self._frame
    #
    # @frame.setter
    # def frame(self, frame):
    #     self._frame = frame
    #     self._setup_path_dict()

    def _setup_path_dict(self):
        """Sets up path dictionary using the specified configuration file and frame number."""
        self.pathdict = {k: v for k, v in self.DEFAULT_PATHS.items()}
        for key, value in self.pathdict.items():
            # Replace frame placeholders in paths with the specified frame number
            if "{frame}" in value:
                self.pathdict[key] = os.path.join(str(self.datadir), value.replace("{frame}", str(self._frame)))
            else:
                self.pathdict[key] = os.path.join(str(self.datadir), value)

    def __getitem__(self, key: str):
        """Return the file corresponding to the key."""
        filepath = str(self.pathdict[key])
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        if filepath.endswith(".csv"):
            return pd.read_csv(filepath)
        elif filepath.endswith(".mitograph"):
            return pd.read_csv(filepath, sep='\t')
        elif filepath.endswith('.npz'):
            return np.load(filepath, allow_pickle=True)
        elif filepath.endswith('.pkl'):
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        elif filepath.endswith('.tif'):
            return ski_io.imread(filepath)
        else:
            try:
                with open(filepath, 'r') as f:
                    return f.read()
            except Exception as e:
                raise Exception(f"Error loading file '{filepath}': {e}")

    def get_voxel_scale(self):
        """Read the voxel scale from the configuration file in default units of micrometers"""

        data = self['mitograph_config']
        pattern = r"Pixel size: -xy (\d+\.\d+)(\w+), -z (\d+\.\d+)(\w+)"
        match = re.search(pattern, data)

        if match:
            xy_scale = float(match.group(1))
            xy_unit = match.group(2)
            z_scale = float(match.group(3))
            z_unit = match.group(4)

            voxel_scale = np.array([xy_scale, xy_scale, z_scale])
            voxel_units = [xy_unit, xy_unit, z_unit]

            return voxel_scale, voxel_units

if __name__ == "__main__":
    data_dir = "data/mitosim_dataset_v5/control/cell_0"
    dl = DataLoader(data_dir)
    print("Initial Path Dictionary:")
    pp.pprint(dl.pathdict)
    dl.frame = 1
    print("")
    print("Updated Path Dictionary:")
    pp.pprint(dl.pathdict)
