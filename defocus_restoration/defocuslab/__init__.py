"""Small, explicit tools for learning incoherent defocus and image restoration.

The core package deliberately does not import PyTorch.  Simulation and classical
restoration work with NumPy, SciPy, and Pillow alone.
"""

from .data import load_grayscale, make_scene, scene_seed
from .metrics import psnr, ssim
from .optics import (
    OpticsConfig,
    blur_image,
    make_psf,
    observe,
    psf_to_otf,
    pupil_function,
)
from .restoration import wiener_restore

__all__ = [
    "OpticsConfig", "pupil_function", "make_psf", "psf_to_otf", "blur_image",
    "observe", "wiener_restore", "make_scene", "scene_seed", "load_grayscale",
    "psnr", "ssim",
]

__version__ = "0.1.0"
