import os
from cemo.preprocess.labels.split_labels import split_labels

rootdir='/data/users/xiazeqing/cemo/tests/preprocess/labels/data'

split_labels(input_file_name=os.path.join(rootdir,'labels.pkl'),
            output_file_names=os.path.join(rootdir,'index%d.pkl'),
            n_types=3,
            start_num=1
            )