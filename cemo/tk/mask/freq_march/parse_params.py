import logging
from cemo.tk.scheduler import make_scheduler
from cemo.tk.mask.freq_march.FreqMarchMask import FreqMarchMask
logger = logging.getLogger(__name__)


def parse_params(
        r_min: int,
        r_max: int,
        mask_shape: str,
        scheduler_type: str,
        max_iter: int,
        inclusive: bool = True,
        verbose: bool = False,
        ) -> FreqMarchMask:
    """
    Make parameters for creating freqency marching mask.

    Args:
        r_min: Minimum radius of the mask.
        r_max: Maximum radius of the mask.
        mask_shape: Shape of the mask.
        scheduler_type: Type of the scheduler.
        max_iter: Maximum iteration of the scheduler.
        inclusive: Whether to include the maximum radius.
        verbose: Whether to print the log.

    Returns:
        Parameters (type: FreqMarchMask)
    """
    r_now = r_min
    freq_march_scheduler = make_scheduler(
        scheduler_type,
        max_iter,
        inclusive=inclusive)

    if verbose:
        logger.info(f">>> Freq. march. mask r min: {r_min}")
        logger.info(f">>> Freq. march. mask r max: {r_max}")
        logger.info(f">>> Freq. march. mask shape: {mask_shape}")
        logger.info(f">>> Initial freq. mask. r = {r_now}")
        logger.info(f">>> Use freq. march. scheduler: {scheduler_type}")
        logger.info(f">>> Freq. march. max iter.: {max_iter}")

    return FreqMarchMask(
        r_min=r_min,
        r_max=r_max,
        r_now=r_now,
        mask_shape=mask_shape,
        scheduler=freq_march_scheduler,
        )
