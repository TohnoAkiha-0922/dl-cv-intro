"""Monochromatic, incoherent, shift-invariant imaging on a periodic grid.

All spatial lengths below refer to the image plane. ``defocus_waves`` is the
quadratic pupil optical path difference at its edge, measured in wavelengths;
it is neither a Gaussian width nor a sensor displacement in millimetres.
"""

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np


def _finite_real(value, name: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    number = float(value)
    if not np.isfinite(number) or (positive and number <= 0):
        qualifier = "positive and finite" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return number


@dataclass(frozen=True)
class OpticsConfig:
    """Image-plane sampling and paraxial circular-pupil optical parameters.

    Noncoherent Nyquist sampling requires ``pixel_pitch_um <= lambda * F#/2``.
    This describes an ideal point-sampled detector: finite pixel integration is
    omitted, and an FFT means that image and PSF repeat at the array boundaries.
    """

    size: int = 256
    pixel_pitch_um: float = 2.0
    wavelength_nm: float = 550.0
    f_number: float = 8.0

    def __post_init__(self):
        if isinstance(self.size, (bool, np.bool_)) or not isinstance(self.size, Integral):
            raise ValueError("size must be an integer >= 16")
        if self.size < 16:
            raise ValueError("size must be an integer >= 16")
        dx = _finite_real(self.pixel_pitch_um, "pixel_pitch_um", positive=True)
        wavelength = _finite_real(self.wavelength_nm, "wavelength_nm", positive=True)
        f_number = _finite_real(self.f_number, "f_number", positive=True)
        limit = wavelength * 1e-3 * f_number / 2.0
        if dx > limit * (1.0 + 1e-12):
            raise ValueError(
                "Incoherent Nyquist sampling violated: pixel_pitch_um must be "
                f"<= wavelength_um * f_number / 2 = {limit:.6g} um; got {dx:.6g} um"
            )


def _real_image(array, name: str = "image") -> np.ndarray:
    """Validate a finite grayscale array without rescaling or clipping."""
    value = np.asarray(array)
    if value.ndim != 2 or min(value.shape, default=0) < 1:
        raise ValueError(f"{name} must be a nonempty two-dimensional grayscale array")
    if not np.issubdtype(value.dtype, np.number) or np.iscomplexobj(value):
        raise ValueError(f"{name} must contain real numeric values")
    value = value.astype(np.float64, copy=False)
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values")
    return value


def _checked_psf(psf) -> np.ndarray:
    value = _real_image(psf, "psf")
    if np.any(value < 0):
        raise ValueError("psf must be nonnegative")
    if not np.isclose(value.sum(), 1.0, rtol=1e-6, atol=1e-9):
        raise ValueError("psf must have unit sum; explicitly normalize its energy first")
    return value


def pupil_function(config: OpticsConfig, defocus_waves: float) -> np.ndarray:
    """Return a centred complex pupil sampled on image spatial frequencies.

    ``fc=1/(2*lambda*F#)`` is the coherent amplitude cutoff; the incoherent
    intensity OTF cutoff is twice this. ``rho`` is frequency divided by ``fc``.
    """
    if not isinstance(config, OpticsConfig):
        raise ValueError("config must be an OpticsConfig")
    waves = _finite_real(defocus_waves, "defocus_waves")
    frequencies = np.fft.fftshift(
        np.fft.fftfreq(config.size, d=float(config.pixel_pitch_um))
    )
    fx, fy = np.meshgrid(frequencies, frequencies, indexing="xy")
    wavelength_um = float(config.wavelength_nm) * 1e-3
    cutoff = 1.0 / (2.0 * wavelength_um * float(config.f_number))
    rho_squared = (fx * fx + fy * fy) / cutoff**2
    pupil = np.zeros((config.size, config.size), dtype=np.complex128)
    inside = rho_squared <= 1.0
    pupil[inside] = np.exp(2j * np.pi * waves * rho_squared[inside])
    return pupil


def make_psf(config: OpticsConfig, defocus_waves: float = 0.5) -> np.ndarray:
    """Compute the centred, nonnegative, unit-energy intensity point response."""
    pupil = pupil_function(config, defocus_waves)
    amplitude = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(pupil)))
    intensity = np.abs(amplitude) ** 2
    total = intensity.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("pupil sampling produced no finite optical energy")
    return intensity / total


def psf_to_otf(psf) -> np.ndarray:
    """Convert a centred unit-energy PSF to the FFT-ordered complex OTF."""
    return np.fft.fft2(np.fft.ifftshift(_checked_psf(psf)))


def blur_image(image, psf) -> np.ndarray:
    """Apply full-grid circular convolution; return an unclipped float array."""
    image = _real_image(image)
    psf = _checked_psf(psf)
    if image.shape != psf.shape:
        raise ValueError("image and psf must have the same shape for periodic convolution")
    return np.fft.ifft2(np.fft.fft2(image) * psf_to_otf(psf)).real


def observe(image, psf, noise_std: float = 0.005, seed: int = 0) -> np.ndarray:
    """Blur then add independent Gaussian intensity noise, without clipping.

    ``noise_std`` is in linear intensity units (0.005 = 0.5% of full scale).
    This Gaussian model omits photon statistics, quantisation, and saturation.
    """
    noise_std = _finite_real(noise_std, "noise_std")
    if noise_std < 0:
        raise ValueError("noise_std must be nonnegative")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    blurred = blur_image(image, psf)
    rng = np.random.default_rng(int(seed))
    return blurred + rng.normal(0.0, noise_std, size=blurred.shape)
