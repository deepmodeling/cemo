import os


def dir_root() -> str:
    """
    Return the root dir of the project repo.
    """
    return os.path.join("..", "..", "..", "..")


def dir_data() -> str:
    """
    Return the path to the data dir
    """
    return os.path.join(dir_root(), "data", "cryosparc", "cs")
