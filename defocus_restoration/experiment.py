"""Choose Wiener balance on validation scenes, then test across defocus."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from defocuslab.data import make_scene, scene_seed
from defocuslab.optics import OpticsConfig, make_psf, observe
from defocuslab.restoration import wiener_restore
from defocuslab.reporting import image_metrics, sweep_figure, write_csv, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/baseline"))
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--test-count", type=int, default=8)
    parser.add_argument("--val-count", type=int, default=6)
    parser.add_argument("--defocus-values", type=float, nargs="+", default=[0, .2, .4, .6, .8, 1, 1.2])
    parser.add_argument("--balance-grid", type=float, nargs="+", default=[.00001, .0001, .001, .01])
    parser.add_argument("--noise-std", type=float, default=.005)
    parser.add_argument("--seed", type=int, default=20261019)
    args = parser.parse_args()
    if args.test_count < 1 or args.val_count < 1:
        parser.error("test-count and val-count must be positive")
    if not all(np.isfinite(args.balance_grid)) or min(args.balance_grid) <= 0:
        parser.error("Wiener balances must be finite and positive")
    try:
        config = OpticsConfig(size=args.size)
        psfs = {w: make_psf(config, w) for w in set(args.defocus_values + [0., .4, .8])}
        validation = []
        manifest = []
        for i in range(args.val_count):
            seed = scene_seed("val", i, base_seed=args.seed)
            clean = make_scene(args.size, seed)
            manifest.append({"split": "val", "index": i, "scene_seed": seed})
            for w in [0., .4, .8]:
                observed = observe(clean, psfs[w], args.noise_std, seed=seed + 73)
                for balance in args.balance_grid:
                    restored = wiener_restore(observed, psfs[w], balance)
                    validation.append({"scene_seed": seed, "defocus_waves": w,
                                       "balance": balance, **image_metrics(clean, restored)})
        scores = {k: float(np.mean([r["psnr_db"] for r in validation if r["balance"] == k]))
                  for k in args.balance_grid}
        selected = max(scores, key=scores.get)
        records = []
        for i in range(args.test_count):
            seed = scene_seed("test", i, base_seed=args.seed)
            clean = make_scene(args.size, seed)
            manifest.append({"split": "test", "index": i, "scene_seed": seed})
            for w in args.defocus_values:
                observed = observe(clean, psfs[w], args.noise_std, seed=seed + 73)
                restored = wiener_restore(observed, psfs[w], selected)
                for method, image in [("observed", observed), ("wiener", restored)]:
                    records.append({"split": "test", "scene_seed": seed,
                                    "defocus_waves": w, "method": method,
                                    **image_metrics(clean, image)})
    except ValueError as error:
        parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "test_metrics.csv", records)
    write_csv(args.output / "validation.csv", validation)
    write_json(args.output / "manifest.json", manifest)
    write_json(args.output / "summary.json", {
        "optics": asdict(config), "noise_std": args.noise_std,
        "selected_balance": selected,
        "selection": "maximum mean validation PSNR across 0,0.4,0.8 waves; test never tunes it",
        "validation_scores_db": scores, "seed": args.seed,
        "test_scenes": args.test_count, "test_defocus_waves": args.defocus_values,
        "metric_convention": "fixed [0,1] clipping for all methods, PSNR data_range=1",
        "boundary": "periodic; samples are generated scenes, not natural-image evidence",
    })
    sweep_figure(args.output / "defocus_curve.png", records)
    print(f"Validation-selected Wiener balance: {selected:g}")
    print(f"Saved {len(records)} test measurements: {args.output.resolve()}")


if __name__ == "__main__":
    main()
