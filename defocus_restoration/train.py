"""Train the optional small conditional U-Net on synthetic teaching scenes."""

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
import sys


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/network"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--train-count", type=int, default=96, help="number of clean training scenes")
    parser.add_argument("--val-count", type=int, default=24, help="number of separate clean validation scenes")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-defocus", default="0,0.4,0.8", help="known defocus values in waves, comma-separated")
    parser.add_argument("--noise-std", type=float, default=0.005, help="unclipped additive Gaussian noise standard deviation")
    parser.add_argument("--pixel-pitch-um", type=float, default=2.0)
    parser.add_argument("--wavelength-nm", type=float, default=550.0)
    parser.add_argument("--f-number", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=2, help="PyTorch CPU threads")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        import math
        import torch
        from torch.utils.data import DataLoader
        from defocuslab.learning import (
            TASK_DESCRIPTION, SyntheticDefocusDataset, parse_defocus,
            save_checkpoint, seed_everything, select_device, write_json,
        )
        from defocuslab.model import ConditionalUNet, ModelConfig
        from defocuslab.optics import OpticsConfig
    except ModuleNotFoundError as exc:
        if exc.name == "torch" or (exc.name and exc.name.startswith("torch.")):
            parser.exit(2, "PyTorch is optional and is not installed in this Python environment.\n"
                        "Install it with: python -m pip install -r requirements-learning.txt\n"
                        "The optics/Wiener demo works without PyTorch.\n")
        raise
    try:
        if min(args.epochs, args.batch_size, args.threads) < 1 or args.seed < 0:
            raise ValueError("epochs, batch-size and threads must be positive; seed must be non-negative")
        if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
            raise ValueError("learning-rate must be finite and positive")
        defocus = parse_defocus(args.train_defocus)
        optics = OpticsConfig(size=args.size, pixel_pitch_um=args.pixel_pitch_um,
                              wavelength_nm=args.wavelength_nm, f_number=args.f_number)
        model_config = ModelConfig(base_channels=args.base_channels)
        torch.set_num_threads(args.threads)
        device, device_note = select_device(args.device)
        # Device probing consumes random numbers, so seed after the probe.
        seed_everything(args.seed)
        train_data = SyntheticDefocusDataset(optics, "train", args.train_count, defocus,
                                             args.noise_std, args.seed, model_config.condition_scale)
        val_data = SyntheticDefocusDataset(optics, "val", args.val_count, defocus,
                                           args.noise_std, args.seed, model_config.condition_scale)
    except (ValueError, RuntimeError) as exc:
        parser.exit(2, f"Configuration/device error: {exc}\n")
    args.output.mkdir(parents=True, exist_ok=True)
    training = {
        "epochs": args.epochs, "train_count": args.train_count, "val_count": args.val_count,
        "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "train_defocus": list(defocus), "val_defocus": list(defocus),
        "noise_std": args.noise_std, "base_seed": args.seed,
        "loss": "raw-output MSE", "checkpoint_selection": "lowest mean validation raw-output MSE",
    }
    runtime = {"torch_version": str(torch.__version__), "device": str(device),
               "device_note": device_note, "cpu_threads": args.threads}
    write_json(args.output / "config.json", {"optics": asdict(optics), "model": asdict(model_config),
                                             "training": training, "runtime": runtime, "task": TASK_DESCRIPTION})
    write_json(args.output / "train-manifest.json", train_data.manifest())
    write_json(args.output / "val-manifest.json", val_data.manifest())
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              generator=generator, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = ConditionalUNet(model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.MSELoss()
    best_mse = float("inf")
    best_epoch = 0
    print(device_note)
    print(f"Synthetic teaching data: {len(train_data)} training samples / {len(val_data)} validation samples")
    print(f"Network parameters: {sum(p.numel() for p in model.parameters()):,}; known defocus input")
    with (args.output / "loss.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("epoch", "train_mse", "val_mse", "selected_best"))
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_sum = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(inputs), targets)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss; inspect data and learning rate")
                loss.backward()
                optimizer.step()
                train_sum += loss.item() * inputs.shape[0]
            train_mse = train_sum / len(train_data)
            model.eval()
            val_sum = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    val_sum += criterion(model(inputs), targets).item() * inputs.shape[0]
            val_mse = val_sum / len(val_data)
            if not math.isfinite(val_mse):
                raise RuntimeError("Non-finite validation loss; no checkpoint selected")
            selected = val_mse < best_mse
            if selected:
                best_mse, best_epoch = val_mse, epoch
                save_checkpoint(args.output / "best.pt", model, optics, training, epoch, val_mse, runtime)
            writer.writerow({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse,
                             "selected_best": int(selected)})
            handle.flush()
            print(f"epoch {epoch:03d}/{args.epochs}: train_mse={train_mse:.7f} val_mse={val_mse:.7f}"
                  + (" [best]" if selected else ""))
    write_json(args.output / "training-summary.json", {"best_epoch": best_epoch, "best_val_mse": best_mse,
                                                       "checkpoint": "best.pt", "task": TASK_DESCRIPTION})
    print(f"Saved validation-selected checkpoint: {args.output / 'best.pt'}")
    print("This small synthetic experiment verifies the pipeline; it is not evidence of research-level performance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
