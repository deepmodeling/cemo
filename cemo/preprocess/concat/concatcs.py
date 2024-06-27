import mrcfile
import numpy as np
from typing import List

def concat_cs(input_file_list:List[str], output_file:str):
    '''
    use to concat cs from command "cryodrgn downsample"
    '''
    data = np.concatenate(list(map(np.load, input_file_list)))
    frame_id_column = 2 # index of the column that stores frame index
    len_data = data.shape[0]
    # redo indexing of frames
    for i in range(len_data):
        data[i][frame_id_column] = np.uint32(i)
    with open(output_file, "wb") as OUT:
        np.save(OUT, data)
