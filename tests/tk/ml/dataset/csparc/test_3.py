import pytest
from cemo.tk.ml.dataset.csparc import CS_Dataset


def test_invalid_cs_file_raises_error():
    file_path = "invalid_cs_file.cs"
    image_size = 64
    use_shift_ratio = True

    with pytest.raises(FileNotFoundError):
        CS_Dataset(file_path, image_size, use_shift_ratio)
