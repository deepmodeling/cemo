import mrcfile
import numpy as np
NumpyArray=np.ndarray

def save_mrc(f: str, vol: NumpyArray, apix: float, L_min: NumpyArray):
    with mrcfile.new(f, overwrite=True) as out_mrc:
        out_mrc.set_data(vol.astype(np.float32))
        out_mrc.voxel_size = apix
        out_mrc.header.origin.x = L_min[0]
        out_mrc.header.origin.y = L_min[1]
        out_mrc.header.origin.z = L_min[2]