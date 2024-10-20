import pytest
import cv2
import numpy as np

from bilateral import bilateral_filter


@pytest.mark.parametrize("shape", [(256, 1), (256, 256, 1)])
@pytest.mark.parametrize("sigma_space", [4, 8])
def test_this_bench(benchmark, shape, sigma_space):
    x = np.random.randn(*shape).astype(np.float32)

    _ = benchmark(bilateral_filter, x, sigma_space, 0.1)


@pytest.mark.parametrize("shape", [(256, 1), (256, 256, 1)])
@pytest.mark.parametrize("sigma_space", [4, 8, 10])
def test_cv_bench(benchmark, shape, sigma_space):
    x = np.random.randn(*shape).astype(np.float32)

    _ = benchmark(cv2.bilateralFilter, x, -1, 0.1, sigma_space)


def test_acc():
    gen = np.random.default_rng(123)
    x = gen.normal(0, 1e-2, (256, 1)).astype(np.float32)
    x[:128] += 1

    bf = bilateral_filter(x, 8, 0.1)
    c = cv2.bilateralFilter(x, -1, 0.1, 8)

    assert (abs(bf).mean() - 0.5) < 1e-2
    assert (abs(c).mean() - 0.5) < 1e-2

    # ignore beginning and end
    assert (abs(bf[30:-30] - c[30:-30]).max()) < 1e-3
