import torch
from pytorch3d import transforms


def t1():
    x1 = torch.randn(2, 3)
    x2 = x1.transpose(1, 0)
    x3 = x1.transpose(0, 1)
    print(f"x1 = {x1}")
    print(f"x2 = {x2}")
    print(f"x3 = {x3}")


def deg2rad(x):
    return x*(torch.pi/180.)

def t2():
    angles = torch.tensor(
        [
            deg2rad(0.),
            deg2rad(90.),
            deg2rad(0.) 
        ]
    )
    convention = "ZYZ"
    # R = transforms.random_rotation()
    R = transforms.euler_angles_to_matrix(angles, convention=convention)
    print(f"R = \n{R.numpy().round()}")
    print((R @ R.transpose(0, 1)).numpy().round())
    v = torch.tensor([[1., 0., 0.]])
    v_col = v.transpose(0, 1)
    v_rotated = R @ v_col
    print(f"R * v = \n{v_rotated.numpy().round()}")
    print("pytorch3d rotate the object rather than the axes")

def t3():
    L_min = -1
    L_max = 1
    L = 3
    ax = torch.linspace(L_min, L_max, L)
    ay = torch.linspace(L_min, L_max, L)
    az = torch.linspace(L_min, L_max, L)
    # xs, ys = torch.meshgrid(ax, ay, indexing="ij")
    xs, ys, zs = torch.meshgrid(ax, ay, az, indexing="ij")
    print(f"xs =\n{xs.numpy()}")
    print(f"ys =\n{ys.numpy()}")
    print(f"zs =\n{zs.numpy()}")
    grid = torch.stack([xs, ys, zs], dim=-1)
    print(f"grid[..., 0] =\n{grid[..., 0].numpy()}")
    print(f"grid[..., 1] =\n{grid[..., 1].numpy()}")
    print(f"grid[..., 2] =\n{grid[..., 2].numpy()}")
    

def t4():
    x = torch.randn(2, 3, 4)
    x2 = x.numpy().T
    print(f"x =\n{x} {x.shape}")
    print(f"x.T =\n{x2} {x2.shape}")

def main():
    # t2()
    # print(torch.zeros(1, 3))

   t4()


if __name__ == "__main__":
    main()
