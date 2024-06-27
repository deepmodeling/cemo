import pickle


def read(f: str) -> dict:
    """
    Read a pickle file.
    """
    with open(f, "rb") as IN:
        return pickle.load(IN)
