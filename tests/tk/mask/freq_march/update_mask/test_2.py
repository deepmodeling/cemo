import torch
from run import run


def test_update_mask():
    # Define the parameters for the function
    iter = 5
    size = (100, 100)
    r_min = 1
    r_max = 10
    r_now = 5
    max_iter = 10
    scheduler_type = 'linear'
    mask_shape = 'round'
    inclusive = True
    dtype = torch.float32
    device = torch.device('cpu')

    run(
        iter=iter,
        size=size,
        r_min=r_min,
        r_max=r_max,
        r_now=r_now,
        max_iter=max_iter,
        scheduler_type=scheduler_type,
        mask_shape=mask_shape,
        inclusive=inclusive,
        dtype=dtype,
        device=device,
    )
