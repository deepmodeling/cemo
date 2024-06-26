from argparse import Namespace
from cemo.io import yml
from cemo.tk.transform import align_volumes
from .calc_rotmat_error import calc_rotmat_error
from .calc_shift_error import calc_shift_error
import numpy


def cli_tmat_error(args: Namespace):
    """
    Input pkl data file must contain a dictionary with the following keys:
    If doing rotation matrix error calculation:
        "rot": ground-truth rotation matrices (N, 3, 3)
        "real_rot_pred": best predicted rotation matrices (N, 3, 3)
    If doing shift error calculation:
        "trans_raw": raw shift vectors (N, 2)
        "real_trans_raw_pred": best predicted shift vectors (N, 2)
    """
    config = yml.read(args.config)
    if not args.skip_volume_align:
        _ = [align_volumes(c, config["env"]) for c in config["files"]]

    if args.rotation_error:
        print("Rotation matrix error:")
        rot_err_sqr = args.rotation_error_squared
        rot_err_stat = args.rotation_error_stat
        rotmat_error_list = [
            calc_rotmat_error(c, squared=rot_err_sqr, stat=rot_err_stat)
            for c in config["files"]]
        all_rotmat_errors = numpy.stack(rotmat_error_list)
        print(all_rotmat_errors.shape)
        rotmat_err_mean = all_rotmat_errors.mean()
        rotmat_err_median = numpy.median(all_rotmat_errors)
        rotmat_err_std = all_rotmat_errors.std()
        print(f"Total rotmat error (mean): {rotmat_err_mean:.3e}")
        print(f"Total rotmat error (median): {rotmat_err_median:.3e}")
        print(f"Total rotmat error (std): {rotmat_err_std:.3e}")

    print("----------------")

    if args.shift_error:
        print("Shift error:")
        shift_err_sqr = args.shift_error_squared
        shift_err_stat = args.shift_error_stat
        shift_error_list = [
            calc_shift_error(
                c,
                squared=shift_err_sqr,
                stat=shift_err_stat,
                apply_rotation=args.shift_use_rotation)
            for c in config["files"]]
        all_shift_errors = numpy.stack(shift_error_list)
        shift_err_mean = all_shift_errors.mean()
        shift_err_median = numpy.median(all_shift_errors)
        shift_err_std = all_shift_errors.std()
        print(f"Total shift error (mean): {shift_err_mean:.3e}")
        print(f"Total shift error (median): {shift_err_median:.3e}")
        print(f"Total shift error (std): {shift_err_std:.3e}")
