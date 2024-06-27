import os


def data_dir() -> str:
    return "../data"


def cs_file_name(which_file: int) -> str:
    if which_file == 1:
        fname = "tg2_n5_with_ctf_and_shift_snr0.1.cs"
    elif which_file == 2:
        fname = "tg2_n10000_with_ctf_and_shift_snr0.1.cs"
    else:
        raise ValueError(f"Unsupported file ID: {which_file}")

    return os.path.join(data_dir(), fname)


