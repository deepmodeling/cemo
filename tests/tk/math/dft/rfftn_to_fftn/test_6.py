from cemo.tk.math import dft
from cemo.tk.math.dft import perturb_coeffs, half_index_edge
import torch


# batched & enforce=True
# use_conj2=True (separate real and imaginary parts)
def test():
    use_conj2 = True
    enforce = True
    symm = False
    inplace = False
    batch_size = 1
    N = 4
    shape = (batch_size, N, N)
    x = torch.rand(shape)
    dims = [-2, -1]
    x_rfft2_raw = torch.fft.rfft2(x, dim=dims)  # (B, N, N//2+1)

    # do some mutations to the input data
    # to effectively test the enforce=True option
    # indices for the columns where Fourier coefficient symmetry is enforced
    idx_edge = half_index_edge(x_rfft2_raw, dims=dims, symm=symm)
    x_rfft2 = perturb_coeffs(
        x_rfft2_raw, idx=idx_edge, inplace=False)

    x_rfft2_sep = torch.view_as_real(x_rfft2)

    dims_sep = [(x - 1) for x in dims]

    x_fft2_sep = dft.rfftn_to_fftn(
        x_rfft2_sep, dims=dims_sep, enforce=enforce,
        use_conj2=use_conj2,
        symm=symm, inplace=inplace)

    x_fft2 = torch.view_as_complex(x_fft2_sep)
    y = torch.fft.ifftn(x_fft2, dim=dims).real
    y_expect = x

    torch.testing.assert_close(y, y_expect, atol=1e-6, rtol=0.)
