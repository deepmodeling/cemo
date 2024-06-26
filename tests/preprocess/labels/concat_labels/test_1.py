import os
from cemo.preprocess.labels.concat_labels import concat_labels

rootdir='/data/users/xiazeqing/cemo/tests/preprocess/labels/data'

concat_labels(input_file_names=os.path.join(rootdir,'index%d.pkl'),
            output_file_name=os.path.join(rootdir,'indexs12.pkl'),
            nums=[1,2]
            )