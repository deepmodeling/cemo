import mrcfile
import numpy as np
from typing import List

'''
use to concat mrcs from command "cryodrgn downsample"
'''

def match_header(mrc1, mrc2):
    mrc1.header.cella.x = mrc2.header.cella.x
    mrc1.header.cella.y = mrc2.header.cella.y
    mrc1.header.cella.z = mrc2.header.cella.z
    mrc1.header.nxstart = mrc2.header.nxstart
    mrc1.header.nystart = mrc2.header.nystart
    mrc1.header.nzstart = mrc2.header.nzstart
    mrc1.header.origin.x = mrc2.header.origin.x
    mrc1.header.origin.y = mrc2.header.origin.y
    mrc1.header.origin.z = mrc2.header.origin.z
    return mrc1


def concat_mrcs(input_names:List[str], output_file:str, apix: float):
    mrc_objs = []
    nz_total = 0
    for i, f in enumerate(input_names):
        temp_mrcs = mrcfile.mmap(f, 'r', permissive=True)
        mrc_objs.append(temp_mrcs)
        nz, _, _ = temp_mrcs.data.shape
        nz_total += nz
    _, ny, nx = mrc_objs[0].data.shape
    mrc_mode = mrc_objs[0].header.mode
    out_shape = (nz_total, ny, nx)
    OUT = mrcfile.new_mmap(
        output_file,
        shape=out_shape,
        mrc_mode=mrc_mode,
        overwrite=True)
    
    OUT = match_header(OUT, mrc_objs[0])
    OUT.header.cella.z = float(nz_total)
    
    # update volume size in angstroms
    if apix > 0:
        OUT.header.cella.x = apix * nx
        OUT.header.cella.y = apix * ny

    def add_one_mrc(output_mrc, mrc_obj, offset):
        nz, _, _ = mrc_obj.data.shape
        for i in range(nz):
            global_idx = i + offset
            if (global_idx + 1) % 1000 == 0:
                print(f"frame {global_idx + 1}")
            output_mrc.data[global_idx] = mrc_obj.data[i]

    def add_frames(output_mrc, mrc_objs, offset):
        if len(mrc_objs) == 0:
            return output_mrc
        else:
            add_one_mrc(output_mrc, mrc_objs[0], offset)
            nz, _, _ = mrc_objs[0].data.shape
            return add_frames(output_mrc, mrc_objs[1:], offset+nz)
    
    add_frames(OUT, mrc_objs, 0)
    _ = [m.close() for m in mrc_objs]
    OUT.close()
    
    # outf.set_data(outdata)
    # outf.header.cella.z = outdata.shape[0]
    # outf.close()
    # os.chmod(output_file, 493)

    _ = [m.close() for m in mrc_objs]
    return
