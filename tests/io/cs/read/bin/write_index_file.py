import pickle
import numpy as np


def write_pkl(f_output: str, data: list):
    with open(f_output, 'wb') as f:
        pickle.dump(data, f)


n = 3
# index = [0, 1, 2]
# f_output = f"data/index_{n}.pkl"
index = np.arange(3, dtype=np.int32)
f_output = f"data/index_{n}.npy"
write_pkl(f_output, index)
