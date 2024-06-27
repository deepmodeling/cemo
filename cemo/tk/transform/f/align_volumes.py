import os
import subprocess


def align_volumes(c: dict, local_env: dict):
    env = os.environ.copy()
    env["PATH"] = "{}:{}".format(local_env["eman2"], env["PATH"])
    if "aligner" in c:
        aligner_config = c["aligner"]
    else:
        aligner_config = "rotate_translate_3d_tree:verbose=1"
    print(f"[[ aligner = {aligner_config}]]")
    verbosity = "9"
    cmd = [
        "python",
        f"{local_env['e2proc3d']}",
        f"--align={aligner_config}",
        f"--output-align-rotmat={c['output']['align']['tmat']}",
        f"--alignref={c['input']['ref']}",
        f"--verbose={verbosity}",
        f"{c['input']['target']}",
        f"{c['output']['align']['volume']}",
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, env=env)
