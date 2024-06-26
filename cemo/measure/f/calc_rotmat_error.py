from cemo.diff.rotmat import diff_rotmat


def calc_rotmat_error(
        c: dict,
        squared: bool,
        stat: str):
    """
    Input pkl data file must contain a dictionary with the following keys:
        "rot": ground-truth rotation matrices
        "real_rot_pred": best predicted rotation matrices
    """
    if "mirror-rotmat" in c["input"]:
        f_mirror_rotmat = c["input"]["mirror-rotmat"]
    else:
        f_mirror_rotmat = ""

    if squared:
        print("[output: squared rotmat error]")

    return diff_rotmat(
        f_data=c["input"]["data"],
        f_align_rotmat=c["output"]["align"]["tmat"],
        squared=squared,
        stat=stat,
        f_fig=c["output"]["rotation"]["fig"]["file"],
        num_bins=c["output"]["rotation"]["fig"]["bins"],
        fig_title=c["output"]["rotation"]["fig"]["title"],
        xlabel=c["output"]["rotation"]["fig"]["x-label"],
        ylabel=c["output"]["rotation"]["fig"]["y-label"],
        f_mirror_rotmat=f_mirror_rotmat,
        f_data_out=c["output"]["rotation"]["data"],
    )
