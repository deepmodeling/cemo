import yaml


def read(path: str) -> dict:
    """
    Read a YAML file.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)
