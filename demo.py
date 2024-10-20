import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from bilateral import bilateral_filter


def example_usage():
    # 1-d space signal with 1 channel
    s = np.random.randn(100, 1) * 1e-2
    sf = bilateral_filter(s, sigma_space=10, sigma_channel=0.1).squeeze()

    # 1-d space signal with 3 channels
    s = np.random.randn(100, 3) * 1e-2
    sf = bilateral_filter(s, sigma_space=10, sigma_channel=0.1).squeeze()

    # 2-d space signal with 3 channels
    s = np.random.randn(100, 100, 3) * 1e-2
    sf = bilateral_filter(s, sigma_space=10, sigma_channel=0.1).squeeze()


def example1d():

    # 1-d
    n = 200
    s = np.concatenate((np.ones(n // 2), np.zeros(n // 2)))
    s += np.random.randn(n) * 1e-2

    bf = bilateral_filter(s[:, None], 10, 0.1).squeeze()
    gf = bilateral_filter(s[:, None], 10, 100).squeeze()
    t = np.linspace(0, 1, n)

    fig, ax = plt.subplots()
    ax.set_prop_cycle(mpl.cycler(color=["k", "c", "r"]))
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.plot(t, s, label="x")
    ax.plot(t, gf, label="Gaussian")
    ax.plot(t, bf, label="Bilateral")
    ax.legend()
    fig.savefig("1d.svg", dpi=300)


def example2d():
    # 2-d

    h = 0.1  # sampling distance
    x, y = np.meshgrid(np.arange(-5, 5, h), np.arange(-5, 5, h))
    z = x * np.exp(-(x**2) - y**2)
    zn = z + np.random.randn(*z.shape) * 1e-2

    space_sigma = int(0.5 / h)  # number of samples
    range_sigma = 0.07

    bf = bilateral_filter(zn[..., None], space_sigma, range_sigma).squeeze()
    gf = bilateral_filter(zn[..., None], space_sigma, 100).squeeze()

    fig = plt.figure(figsize=(12, 8), layout="constrained")
    gs = fig.add_gridspec(
        4, 3, height_ratios=[1.5, 0.05, 0.8, 0.05], wspace=0.01, hspace=0.01
    )

    ax = fig.add_subplot(gs[0, 0], projection="3d")
    s1 = ax.plot_surface(x, y, zn, cmap="gist_earth", label="x")
    ax.set_title("zn")
    ax.view_init(elev=30, azim=-130, roll=0)

    ax = fig.add_subplot(gs[0, 1], projection="3d")
    s2 = ax.plot_surface(x, y, gf, cmap="gist_earth", label="Gaussian")
    ax.set_title("Gaussian(zn)")
    ax.view_init(elev=30, azim=-130, roll=0)

    ax = fig.add_subplot(gs[0, 2], projection="3d")
    s3 = ax.plot_surface(x, y, bf, cmap="gist_earth", label="Bilateral")
    ax.view_init(elev=30, azim=-130, roll=0)
    ax.set_title("Bilateral(zn)")
    fig.savefig("2d.svg", dpi=300)

    ax = fig.add_subplot(gs[1, :])
    ax.set_title("Error plots")
    ax.axis("off")

    ax = fig.add_subplot(gs[2, 0])
    ax.imshow(abs(z - zn), vmin=0, vmax=0.2)

    ax = fig.add_subplot(gs[2, 1])
    img = ax.imshow(abs(z - gf), vmin=0, vmax=0.2)

    ax = fig.add_subplot(gs[2, 2])
    ax.imshow(abs(z - bf), vmin=0, vmax=0.2)

    ax = fig.add_subplot(gs[3, :])
    fig.colorbar(img, cax=ax, orientation="horizontal")
    fig.savefig("2d.svg", dpi=300)


if __name__ == "__main__":
    example_usage()
    example1d()
    example2d()
    plt.show()
