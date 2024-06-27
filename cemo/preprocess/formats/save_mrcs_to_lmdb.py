import os
import lmdb
import mrcfile
import math
import argparse
import torch
from argparse import Namespace

def save_mrcs_to_lmdb(f_output: str, f_mrcs: str, chunk: int = 1000):
    '''
    convert mrcs file to lmdb
    '''
    if os.path.exists(f_output):
        os.remove(f_output)

    env_new = lmdb.open(
        f_output,
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=int(100e9),
    )

     # Load mrc particle images
    with mrcfile.mmap(f_mrcs) as mrc:
        print(f"### MRCS file `{f_mrcs}` header ###")
        mrc.print_header()

        num_images = mrc.header.nz
        h, w = mrc.header.ny, mrc.header.nx
        assert h == w, f"Images must be square! but got {h}x{w}"
        mu, sigma = 0, 0
        for ind in range(math.ceil(num_images / chunk)):
            start = ind * chunk
            end = min((ind + 1) * chunk, num_images)

            particles = torch.tensor(mrc.data[start:end])

            # Calculate normalization statistics
            # For spatial models, normalization should be done before window function
            # For spectral models, normalization should be done after window function and frequency transform
            split = ind + 1
            mu = (1 - 1 / split) * mu + 1 / split * torch.mean(particles)
            sigma = (1 - 1 / split) * sigma + 1 / split * torch.std(particles)


            # Commit on each chunk
            txn_writer = env_new.begin(write=True)
            for image_ind, key_ind in enumerate(range(start, end)):
                txn_writer.put(
                    str(key_ind).encode("ascii"), particles[image_ind].numpy().dumps()
                )
            txn_writer.commit()
            print(f"Processed {start}-{end} projection images.")

    # commit mu and sigma
    txn_writer = env_new.begin(write=True)
    txn_writer.put("num_images".encode("ascii"), num_images.dumps())
    txn_writer.put("mu".encode("ascii"), mu.numpy().dumps())
    txn_writer.put("sigma".encode("ascii"), sigma.numpy().dumps())
    txn_writer.commit()
    print(f"Commit number of {num_images} images, mean and standard variance.")

    env_new.close()