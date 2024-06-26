from argparse import Namespace
from cemo.io import pkl
import numpy
import torch
from cemo.tk.rotation import rotmat_diff
from cemo.tk import plot


def diff_rotmat(
        f_data: str,
        f_align_rotmat: str,
        squared: bool,
        stat: str,
        f_fig: str,
        num_bins: int,
        fig_title: str = "",
        xlabel: str = "log10(diff)",
        ylabel: str = "count",
        f_data_out: str = "",
        f_mirror_rotmat: str = "",
        ):
    poses = pkl.read(f_data)
    R_ref = poses["rot"]
    R_pred = poses["real_rot_pred"]
    dtype = R_ref.dtype
    # transformation matrix
    T = torch.tensor(numpy.loadtxt(f_align_rotmat), dtype=dtype)
    R = T[:, :3]
    if f_mirror_rotmat != "":
        print(">>>> Calculate the correct the rotmat for the mirrored volume")
        N = R_ref.shape[0]
        # F is the mirror-flipping matrix for making the mirrored volume
        raw_F = torch.tensor(
            numpy.loadtxt(f_mirror_rotmat),
            dtype=dtype)
        if raw_F.shape == (3, 4):
            # if the input is a 3x4 transformation matrix
            # only keep the 3x3 rotation matrix part
            F = raw_F[:, 0:3]
        elif raw_F.shape == (3, 3):
            F = raw_F
        else:
            print(">>> Error hint: input mirror rotmat matrix must have a shape of (3, 3) or (3, 4)")
            print(f"    but got {raw_F.shape} instead")
            F = numpy.eye(3)
        print(f"mirror rotation matrix F =\n{F.numpy()}")
        F_inverse = torch.transpose(F, 0, 1)
        # note:
        # Since there is a relationship FR = R'F
        # where R' is the rotmat corresponding a mirrored-version of the 
        # original volume that can generate the same projection image.
        # Then R' = FRF^(-1)
        # after multiplying both sides by F^(-1)
        # (see cryoAI paper appendix D.1 https://arxiv.org/abs/2203.08138)
        R_pred_correct = F.expand(N, 3, 3) @ R_pred @ F_inverse.expand(N, 3, 3)
    else:
        R_pred_correct = R_pred

    diff = rotmat_diff(
        rotmat_ref=R_ref,
        rotmat_target=R_pred_correct,
        squared=squared,
        align_matrix=R,
        ord="fro",
        ).numpy()

    log_diff = numpy.log10(diff)

    if stat == "mean":
        error_stat = numpy.mean(diff)
    elif stat == "median":
        error_stat = numpy.median(diff)
    else:
        raise ValueError(
            f">>> found unsupported shift error stat type: {stat}")

    msg_stat = "{} +/- std: {:.2e} +/- {:.2e}".format(
        stat,
        error_stat,
        numpy.std(diff))

    # plot statistics
    fig_title = f"{fig_title}\n{msg_stat}"
    plot.hist(
        f_fig,
        log_diff,
        num_bins=num_bins,
        fig_title=fig_title,
        xlabel=xlabel,
        ylabel=ylabel,
    )

    # save rotmat difference to an output file.
    if f_data_out != "":
        numpy.savetxt(f_data_out, diff)

    print(msg_stat)

    return diff
