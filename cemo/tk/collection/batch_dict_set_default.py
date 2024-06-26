def batch_dict_set_default(d: dict, defaults: dict) -> dict:
    """
    Update a dictionary with a default value if the key is not present.

    Args:
        d: dictionary to update
        defaults: dictionary of default values

    Returns:
        d: updated dictionary
    """

    def update(k: str):
        d.setdefault(k, defaults[k])

    _ = list(map(update, defaults.keys()))
    return d
