"""Evaluate a saved conditional U-Net against blurred and known-PSF Wiener baselines."""

import argparse
from collections import defaultdict
import csv
from pathlib import Path
import sys


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--test-count", type=int, default=16, help="number of new, separate clean test scenes")
    parser.add_argument("--defocus", default="0,0.2,0.4,0.6,0.8,1.2",
                        help="test defocus in waves; default includes unseen 0.2, 0.6 and 1.2")
    parser.add_argument("--wiener-balance", type=float, default=0.01,
                        help="fixed positive regularizer; do not tune this on test results")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=2)
    return parser


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _representative(output, waves, target, observation, wiener, network, psf):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    import numpy as np

    name = "w" + f"{waves:g}".replace(".", "p")
    np.savez_compressed(output / f"{name}.npz", target=target, observation=observation,
                        wiener=wiener, network=network, psf=psf)
    figure, axes = plt.subplots(1, 4, figsize=(12, 3.4), constrained_layout=True)
    for axis, title, array in zip(axes, ("Target", "Observed", "Wiener (known PSF)", "Conditional U-Net"),
                                  (target, observation, wiener, network)):
        axis.imshow(np.clip(array, 0, 1), cmap="gray", vmin=0, vmax=1)
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    figure.suptitle(f"Synthetic test scene 0; defocus = {waves:g} waves; displays clipped [0, 1]")
    figure.savefig(output / f"{name}.png", dpi=150)
    plt.close(figure)


def _plot_curve(output, aggregates, defocus_values, training_defocus):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for method in ("blurred", "wiener", "network"):
        rows = [row for row in aggregates if row["method"] == method]
        rows.sort(key=lambda row: row["defocus_waves"])
        for axis, key in zip(axes, ("mean_psnr", "mean_ssim")):
            axis.plot([row["defocus_waves"] for row in rows], [row[key] for row in rows],
                      marker="o", label=method)
    for axis, label in zip(axes, ("PSNR (dB)", "SSIM")):
        axis.set_xlabel("Known defocus (waves)")
        axis.set_ylabel(label)
        axis.set_xticks(defocus_values)
        axis.set_xticklabels([f"{w:g}" + ("*" if w not in training_defocus else "") for w in defocus_values])
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Synthetic held-out scenes; * defocus absent from training; all outputs clipped [0, 1]")
    figure.savefig(output / "cross-defocus.png", dpi=160)
    plt.close(figure)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        import math
        import numpy as np
        import torch
        from defocuslab.learning import (
            TASK_DESCRIPTION, SyntheticDefocusDataset, load_checkpoint,
            parse_defocus, select_device, write_json,
        )
        from defocuslab.metrics import psnr, ssim
        from defocuslab.restoration import wiener_restore
    except ModuleNotFoundError as exc:
        if exc.name == "torch" or (exc.name and exc.name.startswith("torch.")):
            parser.exit(2, "PyTorch is optional and is not installed in this Python environment.\n"
                        "Install it with: python -m pip install -r requirements-learning.txt\n"
                        "The optics/Wiener demo works without PyTorch.\n")
        raise
    try:
        if not args.checkpoint.is_file():
            raise ValueError(f"checkpoint does not exist: {args.checkpoint}; run train.py first")
        if args.threads < 1:
            raise ValueError("threads must be positive")
        if not math.isfinite(args.wiener_balance) or args.wiener_balance <= 0:
            raise ValueError("wiener-balance must be finite and positive")
        defocus = parse_defocus(args.defocus)
        torch.set_num_threads(args.threads)
        device, device_note = select_device(args.device)
        model, optics, checkpoint = load_checkpoint(args.checkpoint, device)
        training = checkpoint["training"]
        dataset = SyntheticDefocusDataset(
            optics, "test", args.test_count, defocus, noise_std=training["noise_std"],
            base_seed=training["base_seed"], condition_scale=model.config.condition_scale,
        )
    except (ValueError, RuntimeError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"Configuration/checkpoint/device error: {exc}\n")
    args.output.mkdir(parents=True, exist_ok=True)
    representative_output = args.output / "representatives"
    representative_output.mkdir(exist_ok=True)
    train_defocus = tuple(training["train_defocus"])
    write_json(args.output / "test-manifest.json", dataset.manifest())
    write_json(args.output / "evaluation-config.json", {
        "checkpoint": str(args.checkpoint.resolve()), "best_epoch": checkpoint["best_epoch"],
        "optics": checkpoint["optics"], "noise_std": training["noise_std"],
        "base_seed": training["base_seed"], "model_config": checkpoint["model_config"],
        "test_defocus": list(defocus), "train_defocus": list(train_defocus),
        "wiener_balance": args.wiener_balance,
        "wiener_balance_policy": "fixed before test evaluation; default 0.01 chosen in the bundled independent validation example",
        "device": str(device), "device_note": device_note, "torch_version": str(torch.__version__),
        "ssim": "11x11 Gaussian window, sigma=1.5, population covariance, exclude 5-pixel boundary",
        "task": TASK_DESCRIPTION,
    })
    rows = []
    print(device_note)
    print(f"Optics and noise restored from checkpoint; {len(dataset)} held-out synthetic samples")
    with torch.no_grad():
        for index, record in enumerate(dataset.records):
            observation, target, psf = dataset.arrays(index)
            inputs, _ = dataset[index]
            network = model(inputs.unsqueeze(0).to(device))[0, 0].cpu().numpy()
            wiener = wiener_restore(observation, psf, balance=args.wiener_balance)
            waves = record["defocus_waves"]
            for method, array in (("blurred", observation), ("wiener", wiener), ("network", network)):
                clipped = np.clip(array, 0.0, 1.0)
                rows.append({**record, "seen_defocus": int(waves in train_defocus), "method": method,
                             "psnr": psnr(target, clipped, data_range=1.0),
                             "ssim": ssim(target, clipped, data_range=1.0)})
            if record["scene_index"] == 0:
                _representative(representative_output, waves, target, observation, wiener, network, psf)
    _write_csv(args.output / "per-sample.csv", rows)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["defocus_waves"], row["method"])].append(row)
    aggregates = []
    for (waves, method), group in sorted(groups.items()):
        psnrs = [row["psnr"] for row in group]
        ssims = [row["ssim"] for row in group]
        aggregates.append({"defocus_waves": waves, "seen_defocus": int(waves in train_defocus),
                           "method": method, "count": len(group),
                           "mean_psnr": float(np.mean(psnrs)), "std_psnr": float(np.std(psnrs)),
                           "mean_ssim": float(np.mean(ssims)), "std_ssim": float(np.std(ssims))})
    _write_csv(args.output / "aggregate.csv", aggregates)
    write_json(args.output / "summary.json", {
        "source_scene_count": args.test_count, "sample_count": len(dataset),
        "checkpoint_best_epoch": checkpoint["best_epoch"], "aggregates": aggregates,
        "task": TASK_DESCRIPTION,
    })
    _plot_curve(args.output, aggregates, defocus, train_defocus)
    for row in aggregates:
        print(f"w={row['defocus_waves']:g} {row['method']:7s}: "
              f"PSNR={row['mean_psnr']:.3f} dB SSIM={row['mean_ssim']:.4f}")
    print(f"Saved per-sample/aggregate results, manifests and images to {args.output}")
    print("Synthetic small-sample comparisons are a learning experiment, not a natural-image benchmark.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
