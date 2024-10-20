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


# img = plt.imread("lena.png", format="L")[..., :1] / 255.0
# gf = bilateral_filter(img[None], 4, 20).squeeze(0)
# bf = bilateral_filter(img[None], 4, 0.001).squeeze(0)


# fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
# axs[0].imshow(img * 255)
# axs[1].imshow(bf * 255)
# axs[2].imshow(gf * 255)
# plt.show()
