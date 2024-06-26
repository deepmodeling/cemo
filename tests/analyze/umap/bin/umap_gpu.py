from cuml.manifold.umap import UMAP
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import os
import time


def read_pkl(f: str) -> object:
    with open(f, "rb") as IN:
        return pickle.load(IN)


def write_pkl(f: str, data: object):
    with open(f, "wb") as OUT:
        pickle.dump(data, OUT)


def plot_umap(f_out: str, x, y):
    g = sns.jointplot({"x": x, "y": y}, x="x", y="y", kind="hex", height=4)
    g.ax_joint.set_xlabel("UMAP1")
    g.ax_joint.set_ylabel("UMAP2")
    plt.tight_layout()
    plt.savefig(f_out)


def main():
    f = "data/z.99.pkl"
    f_umap = "tmp/z_99_umap.pkl"
    f_fig = "tmp/z_99_umap.png"
    random_seed = 42
    n_epochs_umap = 25000
    z = read_pkl(f)
    sub_z = z
    if os.path.exists(f_umap):
        umap_embedding = read_pkl(f_umap)
    else:
        t0 = time.time()
        reducer = UMAP(random_state=random_seed, n_epochs=n_epochs_umap)
        umap_embedding = reducer.fit_transform(sub_z)
        t1 = time.time()
        dt = t1 - t0
        print(f"time cost = {dt:.2f} s")
        write_pkl(f_umap, umap_embedding)
    plot_umap(f_fig, umap_embedding[:, 0], umap_embedding[:, 1])


if __name__ == "__main__":
    main()
