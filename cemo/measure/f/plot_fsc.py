import matplotlib.pyplot as plt
import numpy


def plot_fsc(fig_out: str, fsc: numpy.ndarray, res: float, gold_std: float):
    """
    Plot FSC
    """
    fig, ax = plt.subplots()
    x, y = fsc[:, 0], fsc[:, 1]
    ax.plot(x, y, color="black")
    offset = 0.02
    print(f"res = {res}")
    ax.annotate(
        "{:.3f} (resolution: {:.2f})".format(gold_std, res), 
        xy=(res, gold_std),
        xytext=(30 + offset, gold_std + offset),
        xycoords="data",
        ha="left", va="bottom")
    
    plt.xlabel("Resolution")
    plt.ylabel("FSC")
    ax.set_xlim(1, 50)
    ax.hlines(gold_std, 2, 50, linestyles="dashed", colors="k")
    ax.invert_xaxis()
    fig.savefig(fig_out)
