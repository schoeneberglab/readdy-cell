import yaml
from pathlib import Path
import readdy

ut = readdy.units

# TODO: Definitely a better way to do this but it'll do for now
def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


def get_default_configs_dir() -> Path:
    return get_project_root() / "configs" / "defaults"


def get_mitotnt_configfile() -> Path:
    return get_project_root() / "configs" / "defaults" / "mitotnt_defaults.yaml"


def get_workdir_configfile() -> Path:
    return get_project_root() / "configs" / "workdir_config.yaml"


def read_config(filepath):
    """Read a YAML configuration file and return the contents as a dictionary."""
    with open(filepath, 'r') as f:
        return yaml.full_load(f)


def write_config(filepath, data):
    """Write a dictionary to a YAML configuration file."""
    with open(filepath, 'w') as f:
        # Write the dictionary to the file so that it is human-readable
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


