"""First experiment: pupil -> PSF -> observation -> known-PSF Wiener."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from defocuslab.data import load_grayscale, make_scene
from defocuslab.optics import OpticsConfig, blur_image, make_psf, observe, pupil_function
from defocuslab.restoration import wiener_restore
from defocuslab.reporting import demo_figure, image_metrics, save_image, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--image", type=Path, help="Optional user image, converted to grayscale.")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--defocus", type=float, default=0.8, help="Pupil edge OPD in wavelengths.")
    parser.add_argument("--noise-std", type=float, default=0.005)
    parser.add_argument("--balance", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    try:
        config = OpticsConfig(size=args.size)
        clean = (load_grayscale(args.image, size=args.size) if args.image
                 else make_scene(size=args.size, seed=args.seed))
        psf = make_psf(config, args.defocus)
        pupil = pupil_function(config, args.defocus)
        blurred = blur_image(clean, psf)
        observed = observe(clean, psf, noise_std=args.noise_std, seed=args.seed + 10000)
        restored = wiener_restore(observed, psf, balance=args.balance)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "experiment": "known_psf_defocus_demo", "optics": asdict(config),
        "defocus_waves": args.defocus, "noise_std": args.noise_std,
        "wiener_balance": args.balance, "seed": args.seed,
        "source": str(args.image.resolve()) if args.image else "procedural linear-intensity scene",
        "boundary": "periodic circular convolution on the full image grid",
        "metric_convention": "PSNR/SSIM after fixed [0,1] clipping; data_range=1; raw arrays saved",
        "psf_sum": float(psf.sum()), "blurred": image_metrics(clean, blurred),
        "observed": image_metrics(clean, observed), "wiener": image_metrics(clean, restored),
        "limitations": ["synthetic monochromatic scalar incoherent model",
                        "defocus parameter known; not blind deconvolution",
                        "user images use assumed sRGB decoding, without camera calibration",
                        "no neural network is used by this demo"],
    }
    np.savez_compressed(args.output / "arrays.npz", clean=clean, pupil=pupil, psf=psf,
                        blurred=blurred, observed=observed, restored=restored)
    for name, array in [("original", clean), ("blurred", blurred),
                        ("observed", observed), ("wiener", restored)]:
        save_image(args.output / f"{name}.png", array)
    write_json(args.output / "metrics.json", metadata)
    demo_figure(args.output / "comparison.png", clean, pupil, psf, blurred,
                observed, restored, metadata)
    print(f"Observation PSNR: {metadata['observed']['psnr_db']:.3f} dB")
    print(f"Wiener PSNR:      {metadata['wiener']['psnr_db']:.3f} dB")
    print(f"PSF sum:          {psf.sum():.12f}")
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
