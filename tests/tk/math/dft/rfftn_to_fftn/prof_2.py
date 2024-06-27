import torch
import os
from cemo.tk.math import dft
from cemo.tk import tensor


# batched & enforce=True
# add profiling
# use in-place
def run():
    inplace = True
    symm = False
    enforce = True
    batch_size = 500
    N = 1024
    shape = (batch_size, N, N)
    dtype = torch.float32
    device = "cuda"
    dims = [-2, -1]
    tb_dir = "./tmp/log/prof2"
    os.makedirs(tb_dir, exist_ok=True)
    print(tb_dir)

    num_repeats = 3
    wait = 1
    warmup = 1
    active = 3
    period = wait + warmup + active
    num_steps = num_repeats * period
    schedule = torch.profiler.schedule(
        wait=wait, warmup=warmup, active=active, repeat=num_repeats)

    prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
                ],
            schedule=schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(tb_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True)
    prof.start()
    for i in range(num_steps):
        prof.step()
        x = torch.rand(shape, dtype=dtype, device=device)
        x_rfft2 = torch.fft.rfft2(x, dim=dims)  # (B, N, N//2+1)
        x_rfft2_padded = tensor.pad(x_rfft2, dims=[-1], new_sizes=[N])
        x_fft2 = dft.rfftn_to_fftn(
            x_rfft2_padded,
            dims=dims, enforce=enforce, inplace=inplace, symm=symm)
        y = torch.fft.ifftn(x_fft2, dim=dims).real
    prof.stop()

    y_expect = x
    torch.testing.assert_close(y, y_expect, atol=5e-6, rtol=0.)


if __name__ == "__main__":
    run()
