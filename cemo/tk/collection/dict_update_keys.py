from copy import deepcopy


def update_dict_keys(d: dict, name_mapping: dict) -> dict:
    output = deepcopy(d)

    def update(k: str):
        if k in name_mapping:
            new_key = name_mapping[k]
            output[new_key] = output.pop(k)

    _ = list(map(update, d.keys()))
    return output
