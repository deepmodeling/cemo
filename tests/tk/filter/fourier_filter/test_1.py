import torch
import pytest
import itertools
from cemo.tk.filter import fourier_filter
from cemo.tk.math import dft
from cemo.tk.mask import make_mask


@pytest.mark.parametrize(
    (
        "L",
        "domain",
        "use_rfft",
        "symm_freq",
    ),
    itertools.product(
        [5],
        ["real", "fourier"],
        [False, True],
        [False, True],
    ),
)
def test_fourier_filter(
        L: int,
        domain: str,
        use_rfft: bool,
        symm_freq: bool,
        ):
    print("========================")
    print(f"{use_rfft=}")
    print(f"{symm_freq=}")
    print("========================")
    if symm_freq and use_rfft:
        return

    # Define the parameters for the function
    full_shape = (L, L)
    dims = (-2, -1)
    images_real = torch.rand(full_shape)  # 5x5 tensor with random values
    images_ft = dft.fftn(
            images_real,
            dims=dims,
            use_rfft=use_rfft,
            symm_freq=symm_freq,
            symm_real=True,
        )
    if domain == "real":
        x = images_real
    elif domain == "fourier":
        x = images_ft
    else:
        raise ValueError(f"Invalid domain: {domain}")

    mask = make_mask(
        size=(L, L),
        r=L,
        shape="square",
        is_rfft=use_rfft,
        symm=symm_freq,
        inclusive=True,
    )
    

    # Call the function with the parameters
    if symm_freq and use_rfft:
        with pytest.raises(ValueError):
            fourier_filter(
                x,
                domain=domain,
                mask=mask,
                dims=dims,
                use_rfft=use_rfft,
                s=full_shape,
                symm_freq=symm_freq,
                symm_real=True,
            )
        return
    else:
        new_images, new_images_ft = fourier_filter(
            x,
            dims=dims,
            domain=domain,
            mask=mask,
            use_rfft=use_rfft,
            s=full_shape,
            symm_freq=symm_freq,
            symm_real=True,
        )

    # Define the expected output
    # As mask is all ones, images should remain the same

    expected_images_ft = mask * images_ft
    expected_images = dft.ifftn(
        expected_images_ft,
        dims=dims,
        use_rfft=use_rfft,
        symm_freq=symm_freq,
        symm_real=True,
        s=full_shape,
    ).real

    print(f"shape of x: {x.shape}")
    print(f"shape of new_images: {new_images.shape}")
    print(f"shape of expected_images: {expected_images.shape}")
    print(f"shape of new_images_ft: {new_images_ft.shape}")
    print(f"shape of expected_images_ft: {expected_images_ft.shape}")

    # Assert that the function output is as expected
    assert new_images.shape == expected_images.shape
    assert torch.allclose(new_images, expected_images)
    assert torch.allclose(new_images_ft, expected_images_ft)
