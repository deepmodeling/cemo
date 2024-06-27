from ..t.MRCX import MRCX
import mrcfile


def write(f_out: str, obj: MRCX):
    """
    Write an MRCS object to a file.
    """
    
    nz, ny, nx = obj.data.shape
    if obj.header is not None:
        mode = obj.header.mode
    else:
        mode = 2  # float32

    OUT = mrcfile.new_mmap(
        f_out,
        shape=(nz, ny, nx),
        mrc_mode=mode,
        overwrite=True)
    
    OUT.voxel_size = obj.voxel_size

    def set_header(k: str):
        OUT.header[k] = obj.header[k]

    def aux(i: int):
        if (i+1) % 1000 == 0:
            print(i+1)
        OUT.data[i] = obj.data[i]

    _ = list(map(aux, range(nz)))

    if obj.header is not None:
        header_fields = obj.header.dtype.names
        _ = [set_header(k) for k in header_fields]
    OUT.close()
