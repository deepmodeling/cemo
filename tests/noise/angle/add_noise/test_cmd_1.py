from cemo.cs.angle import add_noise
from cemo.cs import CryoSparcCS
from dirs import dir_data
import cemo
import os
import numpy
from pytest import approx
import subprocess
import numpy


def test_cmd_1():
    f_in = os.path.join(dir_data(), "test_cs_1_item.cs")
    f_out = os.path.join("tmp", "out_cs_1_item.cs")
    f_expect = os.path.join("expect", "expect_cs_1_item.cs")
    random_seed = 123
    gau_std = 1.0
    cmd = [
        "cemo",
        "add-angle-noise",
        "-i", f_in,
        "-o", f_out,
        "--gau-std", str(gau_std),
        "--random-seed", str(random_seed),
    ]
    subprocess.run(cmd)
    result = numpy.load(f_out)
    expect = numpy.load(f_expect)
    assert result == expect
