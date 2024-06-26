import subprocess
import os
import numpy
from .reprocess_fsc import reprocess_fsc
from .calc_fsc_res import calc_fsc_res
from .plot_fsc import plot_fsc


def calc_fsc(c: dict, local_env: dict):
    env = os.environ.copy()
    env["PATH"] = "{}:{}".format(local_env["eman2"], env["PATH"])
    vol_ref = c['input']['ref']
    vol_target = c['input']['target']
    f_fsc_data_freq = c["output"]["data"]["freq"]
    f_fsc_data_res = c["output"]["data"]["res"]
    fsc_fig = c["output"]["fig"]
    fsc_std = c["input"]["fsc_std"]
    cmd = [
        "python",
        f"{local_env['e2proc3d']}",
        f"--calcfsc={vol_ref}",
        vol_target,
        f_fsc_data_freq,
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, env=env)
    raw_fsc_data = numpy.loadtxt(f_fsc_data_freq)
    fsc = reprocess_fsc(raw_fsc_data)
    numpy.savetxt(f_fsc_data_res, fsc, fmt='%.4f')
    resolution = calc_fsc_res(fsc, fsc_std)
    plot_fsc(fsc_fig, fsc, resolution, fsc_std)
    print(f_fsc_data_freq)
    print(f_fsc_data_res)
