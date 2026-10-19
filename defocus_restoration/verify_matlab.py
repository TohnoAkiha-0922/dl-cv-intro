"""Compare MATLAB's saved arrays with the Python optical and inverse operators."""

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from defocuslab.optics import OpticsConfig, blur_image, make_psf, psf_to_otf
from defocuslab.reporting import write_json
from defocuslab.restoration import wiener_restore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat-file", type=Path,
                        default=Path("outputs/matlab/lesson1_result.mat"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/matlab/python-crosscheck.json"))
    args = parser.parse_args()
    if not args.mat_file.is_file():
        parser.error("MATLAB result is missing; run matlab/lesson1_defocus.m first")
    saved = loadmat(args.mat_file)

    def scalar(name):
        return float(saved[name].item())

    original = saved["original"]
    if original.ndim != 2 or original.shape[0] != original.shape[1]:
        parser.error("The MATLAB lesson requires a square grayscale image")
    config = OpticsConfig(size=original.shape[0],
                          pixel_pitch_um=scalar("pixel_pitch_um"),
                          wavelength_nm=1000 * scalar("wavelength_um"),
                          f_number=scalar("f_number"))
    psf = make_psf(config, scalar("defocus_waves"))
    arrays = {
        "psf": psf,
        "otf": psf_to_otf(psf),
        "blurred": blur_image(original, psf),
        "restored": wiener_restore(saved["observed"], psf, scalar("balance")),
    }
    checks = {}
    for name, actual in arrays.items():
        expected = saved[name]
        error = float(np.max(np.abs(actual - expected)))
        passed = bool(np.allclose(actual, expected, atol=1e-12, rtol=1e-10))
        checks[name] = {"max_absolute_error": error, "passed": passed}
        print(f"{name:8s} max absolute error: {error:.3e}; passed={passed}")
    result = {"mat_file": str(args.mat_file), "atol": 1e-12, "rtol": 1e-10,
              "checks": checks, "passed": all(c["passed"] for c in checks.values()),
              "scope": "same input arrays and ideal periodic operators; not camera validation"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
