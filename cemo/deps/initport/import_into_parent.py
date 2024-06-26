import importlib
from .is_internal import is_internal


def import_into_parent(mod_name: str, parent_mod: str, parent_mod_dict: dict):
    """
    Import a module's functions/classes into the parent module.

    Args:
        mod_name: The name of the module.
        parent_mod: The name of the parent module.
        parent_mod_dict: The dictionary of the parent module.
    """
    new_mod= f".{mod_name}"
    m = importlib.import_module(new_mod, package=parent_mod)
    m_d = m.__dict__
    def aux(k: str):
        parent_mod_dict[k] = m_d[k]
    _ = list(map(aux, filter(lambda x: not is_internal(x), m_d)))
