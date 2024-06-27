import matplotlib.pyplot as plt
import numpy
from typing import Optional


def hist(
        fig_name: str,
        data: numpy.ndarray,
        num_bins: int = 10,
        fig_title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,):
    """
    Plot histogram of data.

    Args:
        fig_name: name of the figure.
        data: data to be plotted.
        num_bins: number of bins.
        fig_title: title of the figure.
        xlabel: label of the x-axis.
        ylabel: label of the y-axis.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(data, bins=num_bins)
    ax.set_title(fig_title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.canvas.draw()
    fig.savefig(fig_name)
    plt.close(fig)
