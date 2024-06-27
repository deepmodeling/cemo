from typing import Optional


def atom_radius(element: str) -> Optional[float]:
    """
    Return the atomic radius for an element.
    Ref: https://doi.org/10.1039/C3DT50599E
    If element is not found, return None
    """
    db = {
        "C":  1.77,
        "H":  1.20,
        "O":  1.50,
        "N":  1.66,
        "P":  1.90,
        "S":  1.89,
        "F":  1.46,
        "Cl": 1.82,
        "Br": 1.86,
        "I":  2.04,
        "Ca": 2.62,
        "Mg": 2.51,
        "Zn": 2.39
    } 
    if element in db:
        return db[element]
    else:
        return None
