import torch


def project_volumes(volumes: torch.Tensor) -> torch.Tensor:
    """
    Project the volumes `volumes` along the first axis.

    Args:
        volumes: The volume to be transformed. (batch_size, n_z, n_y, n_x)

    Returns:
        The projected images (batch_size, n_y, n_z)
    """
    # input dimensions: (batch_size, n_z, n_y, n_x)
    return volumes.mean(-3)  # (batch_size, n_x, n_y)
