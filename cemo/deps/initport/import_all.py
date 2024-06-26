from pathlib import Path
from typing import List
from .is_internal import is_internal
from .rm_file_ext import rm_file_ext
from .import_into_parent import import_into_parent


def import_all(this_file: str, parent_mod: str, mod_dict: dict):
    """
    Import all functions/modules from the python files 
    in the current directory into the parent module.

    Args:
        this_file: the full path of this __init__.py file.
        parent_mod: the name of the parent module.
        mod_dict: the dictionary representing the current __init__ module.
    """
    paths = sorted(Path(this_file).parent.glob("*.py"))
    fnames = list(map(lambda x: x.name, paths))
    visible_fnames = filter(lambda x: not is_internal(x), fnames)
    submod_names = list(map(rm_file_ext, visible_fnames))
    _ = list(map(lambda x: import_into_parent(x, parent_mod, mod_dict), submod_names))
