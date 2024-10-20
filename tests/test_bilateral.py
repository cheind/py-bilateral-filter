import pytest
import cv2
import numpy as np

from bilateral import bilateral_filter


@pytest.mark.parametrize("shape", [(256, 1), (256, 256, 1)])
@pytest.mark.parametrize("sigma_space", [4, 8])
def test_bilateral(benchmark, shape, sigma_space):
    x = np.random.randn(*shape).astype(np.float32)

    _ = benchmark(bilateral_filter, x, sigma_space, 0.1)


@pytest.mark.parametrize("shape", [(256, 1), (256, 256, 1)])
@pytest.mark.parametrize("sigma_space", [4, 8, 10])
def test_cv(benchmark, shape, sigma_space):
    x = np.random.randn(*shape).astype(np.float32)

    _ = benchmark(cv2.bilateralFilter, x, -1, 0.1, sigma_space)


def test_with_cv2():
    x = np.random.randn(256, 1).astype(np.float32) * 1e-2
    x[:128] += 1

    bf = bilateral_filter(x, 8, 0.1)
    c = cv2.bilateralFilter(x, -1, 0.1, 8)

    # import matplotlib.pyplot as plt

    # plt.plot(np.arange(256), bf)
    # plt.plot(np.arange(256), c)
    # plt.show()

    print(abs(bf - c).max())
