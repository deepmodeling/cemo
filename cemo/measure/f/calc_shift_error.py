from cemo.io import pkl, mrc
import numpy
import torch
from cemo.tk import plot
from cemo.tk.transform import transform_translation_3d
from torch import Tensor


def calc_shift_error(
        c: dict,
        squared: bool,
        stat: str,
        ord: str = 2,
        apply_rotation: bool = False,
        debug: bool = True,
        ):
    """
    Input pkl data file must contain a dictionary with the following keys:
        "trans_raw": raw shift vectors (N, 2)
        "real_trans_raw_pred": best predicted shift vectors (N, 2)
        "real_rot_pred": best predicted rotation matrices
    
    Args:
        c: input parametes (dict)
        squared: if True, use the squared norm,
            otherwise use the standard form.
        stat: either "mean" or "median"
        side_len: number of pixels for one of the image sides
        ord: order of the norm (default: 2)
           --------------------------------------
            ord     vector norm
           --------------------------------------
            2                2-norm
            inf              max(abs(x))
           -inf              min(abs(x))
            0                sum(x != 0)
           other int/float   sum(abs(x)^{ord})^{(1 / ord)}
           --------------------------------------
           https://pytorch.org/docs/stable/generated/torch.linalg.vector_norm.html
        apply_rotation: whether to apply 3D rotation to the 2D shift vectors

    Return:
        shift error (N,)
        unit: pixels
    """
    data = pkl.read(c["input"]["data"])
    dtype = data["real_rot_pred"].dtype

    if "mirror-rotmat" in c["input"]:
        f_mirror_rotmat = c["input"]["mirror-rotmat"]
    else:
        f_mirror_rotmat = ""

    # get number of pixels along one image side
    m = mrc.read(c["input"]["target"])
    side_len = torch.tensor(m.header.nx, dtype=dtype)
    print(f"Image side length D = {side_len}")

    # print(f"side length: {side_len}")
    # print("t_gt(percent)", data["trans_raw"][:5])
    # print("t_pred(percent)", data["real_trans_raw_pred"][:5])
    
    def percent2pixels(x: Tensor) -> Tensor:
        return x * side_len
    
    shift_gt_percent = data["trans_raw"]
    shift_pred_percent = data["real_trans_raw_pred"]
    shift_gt_2d_pixels = percent2pixels(shift_gt_percent)  # (N, 2)
    shift_pred_2d_pixels = percent2pixels(shift_pred_percent)  # (N, 2)
    rotmat_gt = data["rot"]
    rotmat_pred = data["real_rot_pred"]  # (N, 3, 3)
    N = shift_pred_2d_pixels.size(dim=0)

    fig_p = c["output"]["shift"]["fig"]
    out_p = c["output"]["shift"]
    align_p = c["output"]["align"]
    fig_title = fig_p["title"]
    num_bins = fig_p["bins"]
    fig_file = fig_p["file"]
    fig_xlabel = fig_p["xlabel"] if "xlabel" in fig_p else "log10(error)"
    fig_ylabel = fig_p["ylabel"] if "ylabel" in fig_p else "count"
    f_data_out = out_p["data"] if "data" in out_p else ""
    f_align_mat = align_p["tmat"] if "tmat" in align_p else ""

    if f_mirror_rotmat != "":
        print(">>>> Calculate the correct the rotmat for the mirrored volume")
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
        # by multiplying both sides by F^(-1)
        # (see cryoAI paper appendix D.1 https://arxiv.org/abs/2203.08138)
        R_pred_correct = F.expand(N, 3, 3) @ rotmat_pred @ F_inverse.expand(N, 3, 3)
    else:
        R_pred_correct = rotmat_pred

    if f_align_mat != "":
        vol_align_tmat_3x4 = torch.tensor(
            numpy.loadtxt(f_align_mat), dtype=dtype)
        # rotmat for volume alignment (N, 3, 3)
        R_vol_align = vol_align_tmat_3x4[:, 0:3].expand(N, 3, 3)
        # shift vector for volume alignment (N, 3, 1)
        t_vol_align = vol_align_tmat_3x4[:, 3].reshape(3, 1).expand(N, 3, 1)

        # calculate the final translation and rotmat after volume alignment
        R_pred_aligned = R_vol_align.expand(N, 3, 3) @ R_pred_correct
    else:
        R_pred_aligned = R_pred_correct
        t_vol_align = torch.zeros(N, 3, 1)

    if apply_rotation:
        # 3D shift vector for the ground-truth volume before projection
        # (N, 3, 1)
        t_gt_3d_pixels = transform_translation_3d(
                rotmat_gt, shift_gt_2d_pixels).squeeze(dim=-1)

        # (N, 3, 1)
        t_pred_3d_pixels = (
            transform_translation_3d(
                R_pred_aligned, shift_pred_2d_pixels) +
            t_vol_align).squeeze(dim=-1)

        raw_error = torch.linalg.vector_norm(
            t_pred_3d_pixels - t_gt_3d_pixels,
            ord=ord,
            dim=1)

        if debug:
            print(f"pred 3d (pixels): {t_pred_3d_pixels[:10]}")
            print(f"GT 3d (pixels): {t_gt_3d_pixels[:10]}")
    else:
        raw_error = torch.linalg.vector_norm(
            shift_gt_2d_pixels - shift_pred_2d_pixels,
            ord=ord,
            dim=1)

        if debug:
            print(f"pred 2d (pixels): {shift_pred_2d_pixels[:2]}")
            print(f"GT 2d (pixels): {shift_gt_2d_pixels[:2]}")

    if squared:
        print("[output: squared shift error]")
        error_tensor = torch.pow(raw_error, 2)
    else:
        error_tensor = raw_error

    error = error_tensor.numpy()
    log_error = numpy.log10(error)

    if stat == "mean":
        error_stat = numpy.mean(error)
    elif stat == "median":
        error_stat = numpy.median(error)
    else:
        raise ValueError(
            f">>> found unsupported shift error stat type: {stat}")
    msg_stat = "{} +/- std: {:.2e} +/- {:.2e}".format(
        stat,
        error_stat,
        numpy.std(error))
    fig_title = f"{fig_title}\n{msg_stat}"

    plot.hist(
        fig_file,
        log_error,
        num_bins=num_bins,
        fig_title=fig_title,
        xlabel=fig_xlabel,
        ylabel=fig_ylabel,
    )

    # save shift to an output file.
    if f_data_out != "":
        numpy.savetxt(f_data_out, error)

    print(msg_stat)

    return error

    
