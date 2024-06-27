import torch
from typing import Tuple
from cemo.tk.mask.freq_march import update_mask, FreqMarchMask
from cemo.tk.mask import make_mask, update_radius
from cemo.tk.scheduler import make_scheduler


def run(
        iter: int,
        size: Tuple[int, int],
        r_min: int,
        r_max: int,
        r_now: int,
        max_iter: int,
        scheduler_type: str,
        mask_shape: str,
        inclusive: bool,
        dtype: torch.dtype,
        device: torch.device,
        ):
    scheduler = make_scheduler(
        scheduler_type,
        max_iter,
        inclusive=inclusive)

    p = FreqMarchMask(
        r_min=r_min,
        r_max=r_max,
        r_now=r_now,
        scheduler=scheduler,
        mask_shape=mask_shape,
    )
    dtype = torch.float32
    device = torch.device('cpu')

    # Call the function with the parameters
    mask, r_now = update_mask(iter, size, p, dtype, device)

    new_radius = int(
        update_radius(
            iter=iter,
            fn_scheduler=p.scheduler,
            r_min=p.r_min,
            r_max=p.r_max,
        ))
    print(f">>> new radius: {new_radius}")
    print(f">>> current radius: {p.r_now}")

    expected_mask = make_mask(
                size=size,
                r=new_radius,
                shape=p.mask_shape,
                dtype=dtype,
                device=device,
            )
    expected_r_now = new_radius

    # Assert that the function output is as expected
    if expected_mask is None:
        assert mask is None
    else:
        assert torch.allclose(mask, expected_mask)
    assert r_now == expected_r_now
