"""Deterministic synthetic linear-intensity scenes, with source-level splits."""

from numbers import Integral
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


SPLIT_CAPACITY = 1_000_000
_SPLIT_IDS = {"train": 0, "val": 1, "validation": 1, "test": 2}


def _integer(value, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def scene_seed(split: str, index: int, base_seed: int = 20261019) -> int:
    """Reserve disjoint source seeds for train, val, and test at a fixed base.

    Every split has one million source slots. All degraded versions of one
    source belong to the same split; do not randomly split its blur variants.
    ``validation`` is an alias for ``val``, not an additional partition.
    """
    if not isinstance(split, str) or split not in _SPLIT_IDS:
        raise ValueError("split must be 'train', 'val' (or 'validation'), or 'test'")
    index = _integer(index, "index")
    base_seed = _integer(base_seed, "base_seed")
    if index >= SPLIT_CAPACITY:
        raise ValueError(f"index must be < {SPLIT_CAPACITY} to preserve disjoint split seeds")
    return (base_seed * 3 + _SPLIT_IDS[split]) * SPLIT_CAPACITY + index


def make_scene(size: int = 256, seed: int = 0) -> np.ndarray:
    """Create a reproducible, varied grayscale scene in linear intensity [0,1].

    Ellipses, rectangles, lines, and local sinusoidal textures form a deliberately
    small teaching distribution. These are not photographs or research data.
    """
    size = _integer(size, "size", minimum=16)
    seed = _integer(seed, "seed")
    rng = np.random.default_rng(seed)
    axis = np.arange(size, dtype=np.float64) / size
    x, y = np.meshgrid(axis, axis, indexing="xy")
    angle = rng.uniform(0, 2 * np.pi)
    ramp = np.cos(angle) * (x - 0.5) + np.sin(angle) * (y - 0.5)
    scene = rng.uniform(0.12, 0.3) + rng.uniform(-0.1, 0.1) * ramp
    phase = rng.uniform(0, 2 * np.pi)
    scene = scene + rng.uniform(0.015, 0.06) * np.sin(
        2 * np.pi * (rng.uniform(1, 5) * x + rng.uniform(1, 5) * y) + phase
    )

    for _ in range(int(rng.integers(9, 18))):
        cx, cy = rng.uniform(0.12, 0.88, size=2)
        width, height = rng.uniform(0.035, 0.16, size=2)
        rotation = rng.uniform(0, np.pi)
        u = (x - cx) * np.cos(rotation) + (y - cy) * np.sin(rotation)
        v = -(x - cx) * np.sin(rotation) + (y - cy) * np.cos(rotation)
        shape = int(rng.integers(0, 3))
        if shape == 0:
            mask = (u / width) ** 2 + (v / height) ** 2 <= 1.0
        elif shape == 1:
            mask = (np.abs(u) <= width) & (np.abs(v) <= height)
        else:
            mask = (np.abs(u) <= width) & (np.abs(v) <= max(1.0 / size, height / 7))
        tone = rng.uniform(0.04, 0.96)
        modulation = rng.uniform(0.0, 0.1) * np.sin(
            2 * np.pi * rng.uniform(5, min(28, size / 3)) * u + rng.uniform(0, 2 * np.pi)
        )
        scene[mask] = (tone + modulation)[mask]

    # One bounded stripe patch makes loss of fine detail easy to inspect.
    cx, cy = rng.uniform(0.2, 0.8, size=2)
    patch = (np.abs(x - cx) < 0.10) & (np.abs(y - cy) < 0.07)
    stripe_frequency = rng.uniform(4, min(25, size / 3))
    stripes = 0.5 + 0.3 * np.cos(2 * np.pi * stripe_frequency * (x - cx))
    scene[patch] = stripes[patch]
    return np.clip(scene, 0.0, 1.0).astype(np.float32)


def load_grayscale(path, size: int = 256) -> np.ndarray:
    """Centre-crop/resize an ordinary sRGB image and return linear luminance.

    sRGB decoding is applied before resizing the floating-point luminance.
    No camera calibration or ICC colour-management is attempted. Group related
    external images and crops by original source before defining data splits.
    """
    size = _integer(size, "size", minimum=16)
    path = Path(path)
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        srgb = np.asarray(image, dtype=np.float64) / 255.0
    linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    gray = linear @ np.array([0.2126, 0.7152, 0.0722])
    gray_image = Image.fromarray(gray.astype(np.float32))
    resized = ImageOps.fit(gray_image, (size, size), method=Image.Resampling.LANCZOS)
    return np.clip(np.asarray(resized, dtype=np.float32), 0.0, 1.0)
