def dict_update_default(c: dict, k: str, v: str) -> dict:
    """Update a dictionary with a default value if the key is not present."""

    if k not in c:
        c[k] = v
    return c
