import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from itertools import product


def gaussian(x_squared, sigma):
    return np.exp(-0.5 * x_squared / sigma**2)


def bilateral_filter(x, sigma_space, sigma_channel, k=None):
    # https://browncsci1290.github.io/webpage/labs/bilateral/

    space_dims = x.ndim - 2

    if k is None:
        k = int(2 * sigma_space + 1)
    half_k = int(k / 2)

    pad_widths = ((0, 0),) + tuple([(half_k, half_k)] * space_dims) + ((0, 0),)
    xp = np.pad(x, pad_widths, "edge")
    xv = sliding_window_view(xp, x.shape[1:], axis=range(1, x.ndim)).squeeze(
        1 + space_dims
    )
    summed = 0
    weights = 0

    # for each possible position of kernel
    for idx in product(*[range(k)] * space_dims):
        off = np.array(idx).astype(float) - half_k
        shifted = xv[:, *idx]
        g_space = gaussian((off**2).sum(), sigma_space)  # 1d radial for kernel offset
        g_channel = gaussian(((x - shifted) ** 2).sum(-1, keepdims=True), sigma_channel)

        w = g_space * g_channel
        summed += shifted * w
        weights += w

    return summed / weights


n = 1000
s = np.ones(n)
s[n // 2 :] = 0
s = s + np.random.randn(n) * 1e-2

bf = bilateral_filter(s.reshape(1, -1, 1), 50, 0.1).squeeze()
f = bilateral_filter(s.reshape(1, -1, 1), 50, 20).squeeze()

import matplotlib.pyplot as plt

plt.plot(np.linspace(0, 1, n), s, label="x")
plt.plot(np.linspace(0, 1, n), f, label="gaussian")
plt.plot(np.linspace(0, 1, n), bf, label="bilateral")
plt.legend()
plt.show()

img = plt.imread("lena.png", format="L")[..., :1] / 255.0
gf = bilateral_filter(img[None], 8, 20).squeeze(0)
bf = bilateral_filter(img[None], 8, 0.001).squeeze(0)


fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
axs[0].imshow(img * 255)
axs[1].imshow(bf * 255)
axs[2].imshow(gf * 255)
plt.show()
