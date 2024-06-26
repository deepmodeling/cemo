from cemo.io import pkl
import numpy
import os
import pickle


def test():
    x = numpy.ones((3,3))
    f = os.path.join("tmp", "t1.pkl")
    pkl.write(f, x)
    with open(f, "rb") as IN:
        y = pickle.load(IN)
    print("y=\n", y)
    assert x.all() == y.all()
