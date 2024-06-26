from cemo.noise.angle import add_noise
from dirs import dir_data
import cemo
import os
import numpy
from pytest import approx


def test_1():
    rnd_seed = 1234
    rnd_std = 1.0
    f_in = os.path.join(dir_data(), "test_cs_1_item.cs")
    cs = cemo.io.cs.read(f_in)
    new_cs = add_noise(cs, rnd_std, rnd_seed)
    result = new_cs.data[0]["alignments3D/pose"]
    expect = numpy.array([0.0688143, -0.97855496, -1.1646752])

    assert result == approx(expect)
