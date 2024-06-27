import numpy as np
import os
import pickle
import mrcfile
import numpy
import starfile
from typing import List
from pandas import DataFrame

def cs2star(input_file_name:str,output_file_name:str):
    data=np.load(input_file_name)
    # TODO
    #data_output = {
    #    "optics": df_in["optics"],
    #    "particles": data,
    #}
    #starfile.write(data_output, output_file_name, overwrite=True)