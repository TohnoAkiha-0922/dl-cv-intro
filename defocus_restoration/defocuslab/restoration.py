"""Known-PSF periodic Wiener/Tikhonov image restoration."""

import numpy as np

from .optics import _finite_real, _real_image, psf_to_otf


def wiener_restore(observation, psf, balance: float = 0.001) -> np.ndarray:
    """Return ``IFFT(conj(H)*FFT(y)/(|H|^2+balance))`` without clipping.

    A constant ``balance > 0`` is a simple noise-to-signal power ratio proxy;
    it is not estimated from the observation. The same PSF and periodic
    boundary convention as ``blur_image`` must be used. Frequencies with zero
    transfer cannot be recovered; regularisation trades detail for stability.
    """
    observation = _real_image(observation, "observation")
    balance = _finite_real(balance, "balance", positive=True)
    transfer = psf_to_otf(psf)
    if observation.shape != transfer.shape:
        raise ValueError("observation and psf must have the same shape")
    spectrum = np.fft.fft2(observation)
    inverse = np.conj(transfer) / (np.abs(transfer) ** 2 + balance)
    return np.fft.ifft2(inverse * spectrum).real
