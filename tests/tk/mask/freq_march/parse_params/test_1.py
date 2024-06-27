from cemo.tk.mask.freq_march import parse_params, FreqMarchMask
from cemo.tk.scheduler import make_scheduler


def test_parse_params():
    # Define the parameters for the function
    r_min = 1
    r_max = 10
    mask_shape = 'circle'
    scheduler_type = 'linear'
    max_iter = 100
    inclusive = True
    verbose = True

    # Call the function with the parameters
    result = parse_params(
        r_min=r_min,
        r_max=r_max,
        mask_shape=mask_shape,
        scheduler_type=scheduler_type,
        max_iter=max_iter,
        inclusive=inclusive,
        verbose=verbose,
    )

    # Define the expected output
    scheduler = make_scheduler(
        name=scheduler_type,
        iter_max=max_iter,
        inclusive=inclusive,
    )

    expected_output = FreqMarchMask(
        r_min=r_min,
        r_max=r_max,
        r_now=r_min,  # r_now is initialized to r_min
        mask_shape=mask_shape,
        scheduler=scheduler,
    )

    # Assert that the function output is as expected
    attrs_for_comparison = [
        "r_min", "r_max", "r_now", "mask_shape",
    ]

    def aux(attr: str):
        if attr in attrs_for_comparison:
            assert getattr(result, attr) == getattr(expected_output, attr)

    _ = list(map(aux, attrs_for_comparison))
