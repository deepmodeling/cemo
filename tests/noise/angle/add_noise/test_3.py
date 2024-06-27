from cemo.noise.angle import add_noise
from dirs import dir_data
import cemo
import os
import numpy
from pytest import approx


def test_3():
    rnd_seed = 1234
    rnd_std = 1.0
    f_in = os.path.join(dir_data(), "test_cs_10_item.cs")
    cs = cemo.io.cs.read(f_in)
    new_cs = add_noise(cs, rnd_std, rnd_seed)
    result = new_cs.data[:]["alignments3D/pose"]
    expect = numpy.loadtxt("expect/expect_10.dat")

    def comp(i: int) -> bool:
        print(i, result[i], expect[i])
        assert result[i] == approx(expect[i])

    assert list(map(comp, range(3)))
