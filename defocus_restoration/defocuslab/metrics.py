"""Full-reference metrics. Callers, not metrics, choose a clipping policy."""

import numpy as np
from scipy.ndimage import gaussian_filter

from .optics import _finite_real, _real_image


def _pair(reference, image, data_range):
    reference = _real_image(reference, "reference")
    image = _real_image(image, "image")
    if reference.shape != image.shape:
        raise ValueError("reference and image must have the same shape")
    return reference, image, _finite_real(data_range, "data_range", positive=True)


def psnr(reference, image, data_range: float = 1.0) -> float:
    """Peak signal-to-noise ratio in dB; identical arrays return infinity."""
    reference, image, data_range = _pair(reference, image, data_range)
    mse = float(np.mean((reference - image) ** 2))
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(data_range**2 / mse))


def ssim(reference, image, data_range: float = 1.0) -> float:
    """Gaussian-window SSIM (11x11, sigma=1.5, population covariance).

    The five-pixel boundary is excluded after filtering. Unlike some libraries,
    no sample-covariance correction is used; record this when comparing scores.
    Images smaller than 11 pixels in either dimension are rejected.
    """
    reference, image, data_range = _pair(reference, image, data_range)
    if min(reference.shape) < 11:
        raise ValueError("SSIM needs images at least 11 x 11 pixels")

    def window(array):
        return gaussian_filter(array, sigma=1.5, truncate=5.0 / 1.5, mode="reflect")

    mu_x, mu_y = window(reference), window(image)
    var_x = np.maximum(window(reference * reference) - mu_x * mu_x, 0.0)
    var_y = np.maximum(window(image * image) - mu_y * mu_y, 0.0)
    covariance = window(reference * image) - mu_x * mu_y
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    score = ((2 * mu_x * mu_y + c1) * (2 * covariance + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
    )
    return float(np.mean(score[5:-5, 5:-5]))
