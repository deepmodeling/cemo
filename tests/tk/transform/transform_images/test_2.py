import torch
from torchvision import transforms
from PIL import Image
from cemo.tk.transform import transform_images
import os


def read_img(f_img: str) -> torch.Tensor:
    return transforms.ToTensor()(Image.open(f_img))


def save_img(f_out: str, x: torch.Tensor):
    transforms.ToPILImage()(x[-1]).save(f_out)
    print(f"Saved {f_out}")


def test():
    N = 1
    f_img = os.path.join("..", "data", "img1.png")
    f_out = os.path.join("tmp", "out1.png")
    f_expect = os.path.join("expect", "expect1.png")
    x_in = read_img(f_img).expand(N, -1, -1, -1)
    shift_ratio = torch.tensor([-0.5, -0.5]).expand(N, -1)
    x_out = transform_images(
        x_in,
        shift_ratio=shift_ratio,
        align_corners=False)
    x_expect = read_img(f_expect).expand(N, -1, -1, -1)
    save_img(f_out, x_out)
    print(torch.max(torch.abs(x_out - x_expect)))
    assert torch.allclose(x_out, x_expect, atol=1e-2)
