import pickle


def write(fname: str, obj: object):
    """
    Write a pkl file.

    Args:
        fname: output file name
        obj: output python object
    """
    with open(fname, "wb") as OUT:
        pickle.dump(obj, OUT)
