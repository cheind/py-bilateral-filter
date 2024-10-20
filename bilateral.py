from itertools import product

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def gaussian(x_squared, sigma):
    """Evaluate the unnormalized Gaussian"""
    return np.exp(-0.5 * x_squared / sigma**2)


def bilateral_filter(x, sigma_space, sigma_channel, k=None):
    """Apply bilateral filtering to a signal.

    Here we consider a signal $x$ having domain S in N-dimensional
    space and range R in C-dimensional space. The bilateral filtered
    signal $y$ at pixel location $p$ is given by

        y(p) = 1/w(p) \\sum_{q \\in S}w(q,p)I_q
        w(q,p) = G_s(|q-p|)G_c(|x_q-x_p|)
        w(p) = \\sum_{q \\in S}w(q,p)

    here $G_s$ and $G_c$ are unnormalized Gaussians with user defined
    variances.

    For the vectorized version we consider each contributing n-d shift $o$
    and compute

        y(p) = 1/w(p) \\sum_{o} w(p-o,p)I_{p-o}
        w(p-o,p) = G_s(|o|)G_c(|x_{p-o}-x_p|)
        w(p) = \\sum_{o} w(p-o,p)

    The contributing shifts are computed from the cut-off kernel size
    $k=2*ceil(3*sigma_space)+1$.

    This method is vectorized, but still slow for large sigma values,
    because it requires O(|S|*|k|^2) evaluations.

    While Gaussian filtering can be represented as a linear convolution
    with a fixed kernel, Bilateral filtern cannot as it a) requires
    a non-linear normalization factor and the range kernel depends
    on the actual signal values. See [1] for a linear formulation
    in a homogeneous space of higher dims.

    Params:
        x: (*,C) n-dimensional signal having C channels.
        sigma_space: standard deviation of Gaussian applied in
            space/domain dimensions
        sigma_channel: standard deviation of Gaussian applied in
            range dimension.
        k: optional kernel size. If not provided computed as
            2*ceil(3*sigma_space) + 1

    Returns:
        y: (*,C) filtered signal

    References:
        [1] Paris, Sylvain, and Frédo Durand.
        "A fast approximation of the bilateral filter using a signal processing approach.
    """

    # https://browncsci1290.github.io/webpage/labs/bilateral/

    if k is None:
        # odd kernel size with at least 3 sigma on each side.
        k = 2 * int(np.ceil(3 * sigma_space)) + 1
    half_k = int(k / 2)

    # Pad space dimensions
    space_dims = x.ndim - 1
    pad_widths = tuple([(half_k, half_k)] * space_dims) + ((0, 0),)
    xp = np.pad(x, pad_widths, "edge")

    # View strided i.e image rolled for various positions of the kernel.
    xv = sliding_window_view(xp, x.shape).squeeze(space_dims)

    # Then, for each possible offset
    summed = 0
    weights = 0
    for idx in product(*[range(k)] * space_dims):
        # idx is an n-dimensional integer giving us the current kernel
        # position with (0,...,0) being (-halfk,...,-halfk)
        off = np.array(idx).astype(float) - half_k
        # Slice the shifted image from the strided view
        shifted = xv[*idx]

        g_space = gaussian((off**2).sum(), sigma_space)
        g_channel = gaussian(((x - shifted) ** 2).sum(-1, keepdims=True), sigma_channel)

        w = g_space * g_channel
        summed += shifted * w
        weights += w

    return summed / weights


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import matplotlib as mpl

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

    example1d()
    example2d()
    plt.show()


# img = plt.imread("lena.png", format="L")[..., :1] / 255.0
# gf = bilateral_filter(img[None], 4, 20).squeeze(0)
# bf = bilateral_filter(img[None], 4, 0.001).squeeze(0)


# fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
# axs[0].imshow(img * 255)
# axs[1].imshow(bf * 255)
# axs[2].imshow(gf * 255)
# plt.show()
