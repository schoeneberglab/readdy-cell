import yaml
from pathlib import Path
import readdy
from types import SimpleNamespace

ut = readdy.units

# TODO: Use .project-root file to infer project root
def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


def get_default_configs_dir() -> Path:
    return get_project_root() / "configs" / "defaults"


def get_mitotnt_configfile() -> Path:
    return get_project_root() / "configs" / "defaults" / "mitotnt_defaults.yaml"


def get_outdir_configfile() -> Path:
    return get_project_root() / "configs" / "outdir_config.yaml"


def read_config(filepath):
    """Read a YAML configuration file and return the contents as a dictionary."""
    with open(filepath, 'r') as f:
        return yaml.full_load(f)


def write_config(filepath, data):
    """Write a dictionary to a YAML configuration file."""
    with open(filepath, 'w') as f:
        # Write the dictionary to the file so that it is human-readable
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

def load_config(path):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    def _to_ns(d):
        return SimpleNamespace(**{k: _to_ns(v) if isinstance(v, dict) else v
                                  for k, v in d.items()})
    return _to_ns(cfg)
