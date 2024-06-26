import torch
Tensor = torch.Tensor


def shift2d(x: Tensor, delta: Tensor, indexing: str = "ij") -> Tensor:
    """
    Shift an image by a given amount of pixels by indexing.
    Args:
        x: Image to shift. Shape: (H, W) or (B, H, W)
        delta: Amount to shift by. Shape: (2,) or (B, 2)
        indexing: if "ij", dims 0 and 1 correspond to x and y.
            If "xy", dims 0 and 1 correspond to y and x.
    Returns:
        Shifted image.
    """
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if delta.ndim == 1:
        delta = delta.unsqueeze(0)
    batch_size = x.shape[0]
    assert delta.shape[0] == batch_size, \
        f"batch_size = {batch_size}, delta.shape[0] = {delta.shape[0]}"

    shift = delta.to(torch.int64)

    if indexing == "ij":
        ix, iy = (-2, -1)
    elif indexing == "xy":
        ix, iy = (-1, -2)
    else:
        raise NotImplementedError(
            f"This indexing method ({indexing}) hasn't been implemented yet.")

    dx = shift[..., ix].reshape(-1, 1)
    dy = shift[..., iy].reshape(-1, 1)
    len_x, len_y = x.shape[-2:]
    batch_shape = x.shape[:-2]
    x_range = torch.arange(len_x).expand(*batch_shape, -1)
    y_range = torch.arange(len_y).expand(*batch_shape, -1)
    x_range - dx
    x_idx = (x_range - dx) % len_x
    y_idx = (y_range - dy) % len_y
    output = torch.empty_like(x)

    def aux(i: int):
        output[i, :, :] = x[i, x_idx[i, :], :][:, y_idx[i, :]]

    _ = list(map(aux, range(batch_size)))

    return output.squeeze(0)
