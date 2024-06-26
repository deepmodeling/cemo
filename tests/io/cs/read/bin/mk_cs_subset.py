import numpy as np


def write_cs(fname: str, data: np.ndarray):
    with open(fname, "wb") as OUT:
        np.save(OUT, data)


subset_size = 5
f_input = "data/tg2_n10000_with_ctf_and_shift_snr0.1.cs"
f_output = f"data/tg2_n{subset_size}_with_ctf_and_shift_snr0.1.cs"
x = np.load(f_input)
print(x.shape)
print(f"new size: {subset_size}")
write_cs(f_output, x[:subset_size])

