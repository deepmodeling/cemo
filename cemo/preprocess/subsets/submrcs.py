import mrcfile
import pickle


def load_pkl(f):
    with open(f, "rb") as IN:
        return pickle.load(IN)


def read_mrcs(f: str) -> object:
    with mrcfile.mmap(f, mode="r") as IN:
        return (IN.data, IN.header)


def write_mrcs(f: str, data, header, ind):
    num_frames = len(ind)
    _, ny, nx = data.shape
    new_shape = (num_frames, ny, nx)
    print(f"new mrcs shape: {new_shape}")
    with mrcfile.new_mmap(f, shape=new_shape, mrc_mode=2, overwrite=True) as OUT:
        def add_frame(i: int):
            original_id = ind[i]
            OUT.data[i] = data[original_id]
            if (i+1) % 5000 == 0 or i == 0:
                print(f"frame {i+1} (original id: {original_id})")
        OUT.header.cella = header.cella
        OUT.header.origin = header.origin
        OUT.header.nxstart = header.nxstart
        OUT.header.nystart = header.nystart
        OUT.header.nzstart = header.nzstart
        _ = list(map(add_frame, range(num_frames)))


def submrcs(input_mrcs_file_name:str, output_mrcs_file_name:str,index_file_name:str):
    '''
    Split mrcs by index files.
    '''
    ind=load_pkl(index_file_name)
    old_data, old_header = read_mrcs(input_mrcs_file_name)
    write_mrcs(output_mrcs_file_name, old_data, old_header, ind)