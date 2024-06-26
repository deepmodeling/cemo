import torch
from pytorch3d import transforms
from scipy.spatial.transform import Rotation


def deg2rad(x):
    return x*(torch.pi/180.)

def gen_euler():
    angles = torch.tensor(
        [
            deg2rad(0.),
            deg2rad(0.),
            deg2rad(10.) 
        ]
    )
    return angles

def main():
    angles = gen_euler()
    convention = "ZYZ"
    R_p3d = transforms.euler_angles_to_matrix(angles, convention=convention)
    R_sci = Rotation.from_euler()