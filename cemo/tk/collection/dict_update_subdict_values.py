from copy import deepcopy
from typing import List


def update_subdict_values(d: dict, keys: List[str], values: dict) -> dict:
    output = deepcopy(d)

    def update(k: str):
        output[k].update(values)

    _ = list(map(update, d.keys()))
    return output
