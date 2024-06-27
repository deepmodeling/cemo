from cemo.parsers import noise, img, diff, measure, file, transform, preprocess
from collections.abc import Callable
from typing import TYPE_CHECKING
import sys
if sys.version_info >= (3, 9) or TYPE_CHECKING:
    from argparse import _SubParsersAction
    FnSubparser = Callable[[_SubParsersAction], _SubParsersAction]
else:
    FnSubparser = Callable


def all_parsers() -> FnSubparser:
    """
    Return all parsers.
    """
    return [
        noise.p_add_angle_noise,
        img.p_random_shift_2d,
        img.p_undo_shift_2d,
        diff.p_diff_rotmat,
        measure.p_tmat_error,
        measure.p_fsc,
        file.p_mrc_set_location,
        transform.p_transform_volume,
        transform.p_align_volumes,
        preprocess.p_concat_labels,
        preprocess.p_concatcs,
        preprocess.p_concatmrcs,
        preprocess.p_cs2pkls,
        preprocess.p_save_mrcs_to_lmdb,
        preprocess.p_split_labels,
        preprocess.p_submrcs,
        preprocess.p_substars,
    ]
