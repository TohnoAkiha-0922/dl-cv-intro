"""Numerical physical invariants and source-level isolation, not snapshots."""

import numpy as np
import pytest
from PIL import Image

from defocuslab import (
    OpticsConfig, blur_image, load_grayscale, make_psf, make_scene, observe,
    psf_to_otf, psnr, pupil_function, scene_seed, ssim, wiener_restore,
)


@pytest.mark.parametrize("size", [64, 65])
@pytest.mark.parametrize("waves", [0.0, 0.5, 1.25])
def test_psf_energy_sign_symmetry_and_otf_dc(size, waves):
    config = OpticsConfig(size=size)
    psf = make_psf(config, waves)
    assert psf.shape == (size, size)
    assert np.min(psf) >= 0
    np.testing.assert_allclose(psf.sum(), 1.0, atol=1e-14)
    np.testing.assert_allclose(psf, make_psf(config, -waves), atol=1e-14)
    np.testing.assert_allclose(psf_to_otf(psf)[0, 0], 1.0, atol=1e-14)


def test_pupil_cutoff_and_phase_have_the_documented_units():
    config = OpticsConfig(size=128)
    waves = 0.75
    pupil = pupil_function(config, waves)
    frequencies = np.fft.fftshift(np.fft.fftfreq(config.size, config.pixel_pitch_um))
    fx, fy = np.meshgrid(frequencies, frequencies)
    cutoff = 1.0 / (2 * config.wavelength_nm * 1e-3 * config.f_number)
    rho2 = (fx * fx + fy * fy) / cutoff**2
    inside = rho2 <= 1
    np.testing.assert_array_equal(np.abs(pupil[~inside]), 0)
    np.testing.assert_allclose(np.abs(pupil[inside]), 1, atol=1e-14)
    np.testing.assert_allclose(pupil[inside], np.exp(2j * np.pi * waves * rho2[inside]))
    assert pupil[64, 64] == 1


def test_focused_otf_has_incoherent_optical_cutoff():
    config = OpticsConfig(size=128)
    otf = np.fft.fftshift(psf_to_otf(make_psf(config, 0.0)))
    frequencies = np.fft.fftshift(np.fft.fftfreq(config.size, config.pixel_pitch_um))
    fx, fy = np.meshgrid(frequencies, frequencies)
    incoherent_cutoff = 1 / (config.wavelength_nm * 1e-3 * config.f_number)
    outside = np.hypot(fx, fy) > incoherent_cutoff
    assert np.max(np.abs(otf[outside])) < 1e-12


@pytest.mark.parametrize("size", [64, 65])
def test_impulse_position_constant_and_flux_are_preserved(size):
    psf = make_psf(OpticsConfig(size=size), 0.5)
    centred_delta = np.zeros((size, size))
    centred_delta[size // 2, size // 2] = 1
    np.testing.assert_allclose(blur_image(centred_delta, psf), psf, atol=1e-15)
    origin_delta = np.zeros_like(centred_delta)
    origin_delta[0, 0] = 1
    np.testing.assert_allclose(blur_image(origin_delta, psf), np.fft.ifftshift(psf), atol=1e-15)
    np.testing.assert_allclose(blur_image(np.full_like(psf, 0.37), psf), 0.37, atol=1e-14)
    image = make_scene(size, seed=42)
    np.testing.assert_allclose(blur_image(image, psf).sum(), image.astype(float).sum(), rtol=1e-13)


def test_known_psf_restoration_improves_an_unseen_source():
    config = OpticsConfig(size=128)
    image = make_scene(config.size, scene_seed("test", 0))
    psf = make_psf(config, 0.5)
    observed = observe(image, psf, noise_std=0, seed=0)
    restored = wiener_restore(observed, psf, balance=1e-5)
    observed, restored = np.clip(observed, 0, 1), np.clip(restored, 0, 1)
    assert psnr(image, restored) > psnr(image, observed) + 5
    assert ssim(image, restored) > ssim(image, observed)
    noisy = observe(image, psf, noise_std=0.005, seed=0)
    regularised = wiener_restore(noisy, psf, balance=0.001)
    assert psnr(image, np.clip(regularised, 0, 1)) > psnr(image, np.clip(noisy, 0, 1)) + 2


def test_observation_is_reproducible_and_not_clipped():
    psf = make_psf(OpticsConfig(size=64), 0)
    zeros = np.zeros((64, 64))
    observation = observe(zeros, psf, noise_std=0.1, seed=8)
    assert observation.min() < 0 < observation.max()
    np.testing.assert_array_equal(observation, observe(zeros, psf, noise_std=0.1, seed=8))
    assert not np.array_equal(observation, observe(zeros, psf, noise_std=0.1, seed=9))


def test_source_seeds_are_disjoint_and_scene_generation_is_deterministic():
    seeds = [{scene_seed(split, i) for i in range(300)} for split in ("train", "val", "test")]
    assert seeds[0].isdisjoint(seeds[1])
    assert seeds[0].isdisjoint(seeds[2])
    assert seeds[1].isdisjoint(seeds[2])
    assert scene_seed("val", 4) == scene_seed("validation", 4)
    for split in ("train", "val", "test"):
        assert scene_seed(split, 999_999) > scene_seed(split, 0)
    image = make_scene(64, scene_seed("train", 0))
    assert image.dtype == np.float32
    assert 0 <= image.min() <= image.max() <= 1
    np.testing.assert_array_equal(image, make_scene(64, scene_seed("train", 0)))
    assert not np.array_equal(image, make_scene(64, scene_seed("val", 0)))
    with pytest.raises(ValueError):
        scene_seed("train", 1_000_000)
    with pytest.raises(ValueError):
        scene_seed("unknown", 0)
    with pytest.raises(ValueError):
        scene_seed("test", -1)


def test_external_srgb_is_decoded_to_linear_intensity(tmp_path):
    path = tmp_path / "gray.png"
    Image.fromarray(np.full((24, 36, 3), 128, dtype=np.uint8)).save(path)
    linear = load_grayscale(path, size=32)
    assert linear.shape == (32, 32)
    assert linear.dtype == np.float32
    expected = ((128 / 255 + 0.055) / 1.055) ** 2.4
    np.testing.assert_allclose(linear, expected, atol=1e-7)


def test_metrics_have_analytic_limits_and_reject_bad_pairs():
    image = np.full((32, 32), 0.5)
    assert np.isinf(psnr(image, image))
    np.testing.assert_allclose(ssim(image, image), 1.0, atol=1e-12)
    np.testing.assert_allclose(psnr(image, image + 0.1), 20.0, atol=1e-12)
    assert ssim(image, image + 0.1) < 1
    with pytest.raises(ValueError):
        psnr(image, np.zeros((16, 16)))
    with pytest.raises(ValueError):
        psnr(image, image, data_range=0)
    with pytest.raises(ValueError):
        ssim(np.ones((8, 8)), np.ones((8, 8)))


@pytest.mark.parametrize("kwargs", [
    {"size": 8}, {"size": 64.0}, {"size": True}, {"pixel_pitch_um": 2.3},
    {"wavelength_nm": 0}, {"f_number": -8}, {"pixel_pitch_um": np.inf},
])
def test_invalid_or_aliased_optical_sampling_is_rejected(kwargs):
    with pytest.raises(ValueError):
        OpticsConfig(**kwargs)


def test_invalid_observations_and_psfs_are_rejected():
    config = OpticsConfig(size=64)
    psf = make_psf(config)
    image = np.zeros_like(psf)
    for invalid in (np.full_like(image, np.nan), image.astype(complex), image[..., None]):
        with pytest.raises(ValueError):
            blur_image(invalid, psf)
    with pytest.raises(ValueError):
        blur_image(image[:-1], psf)
    with pytest.raises(ValueError):
        psf_to_otf(psf * 2)
    with pytest.raises(ValueError):
        psf_to_otf(-psf)
    with pytest.raises(ValueError):
        make_psf(config, np.nan)
    with pytest.raises(ValueError):
        observe(image, psf, noise_std=-0.1)
    with pytest.raises(ValueError):
        wiener_restore(image, psf, balance=0)
