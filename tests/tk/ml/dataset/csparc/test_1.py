import numpy as np
from cemo.tk.ml.dataset import csparc
from cemo.tk.physics.microscope import CTF_Params


def test():
    use_shift_ratio = True
    f_input = "data/7bcq_c1_n10_ctf_noise-free_maxshift8px.cs"
    image_size = 64

    data_ref = np.load(f_input)

    # Create an instance of CSDataset
    dataset = csparc.CS_Dataset(
        file_path=f_input,
        image_size=image_size,
        use_shift_ratio=use_shift_ratio,
        use_ctf=True,
        verbose=True,
    )

    rotmat, shift, ctf_params = dataset[0]
    assert len(dataset) == len(data_ref)
    assert rotmat.shape == (3, 3)
    assert shift.shape == (2,)
    assert isinstance(ctf_params, CTF_Params)
    assert abs(shift[0]) <= 1.
    assert abs(shift[1]) <= 1.
