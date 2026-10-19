"""Shared reproducible data, device and checkpoint helpers for optional PyTorch."""

from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import make_scene, scene_seed
from .model import ConditionalUNet, ModelConfig
from .optics import OpticsConfig, make_psf, observe


TASK_DESCRIPTION = {
    "task": "known-defocus synthetic intensity restoration",
    "input": "raw observation, plus known defocus_waves / condition_scale plane",
    "target": "linear intensity in [0, 1]",
    "boundary": "periodic FFT convolution; circular network convolutions",
    "comparison": "Wiener and network both receive the true defocus parameter",
    "metric_policy": "clip every method output to [0, 1] before PSNR/SSIM",
    "limitation": "procedural teaching scenes; smoke results are not natural-image or research performance",
}


def parse_defocus(text):
    try:
        values = tuple(float(value.strip()) for value in text.split(","))
    except (ValueError, AttributeError) as exc:
        raise ValueError("defocus must be comma-separated numbers, e.g. 0,0.4,0.8") from exc
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("defocus values must be finite and non-negative")
    if len(set(values)) != len(values):
        raise ValueError("duplicate defocus values are not useful; supply each value once")
    return values


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def select_device(requested="auto"):
    """Probe a real convolution and backward kernel, including new GPU support.

    cuda.is_available() alone cannot prove that the installed PyTorch contains
    kernels compatible with the GPU's compute capability.
    """
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu or cuda")
    if requested == "cpu":
        return torch.device("cpu"), "CPU explicitly selected"
    if not torch.cuda.is_available():
        reason = "CUDA is unavailable in this Python/PyTorch environment"
    else:
        try:
            with torch.enable_grad():
                layer = torch.nn.Conv2d(2, 4, 3, padding=1).to("cuda")
                probe = torch.ones((1, 2, 16, 16), device="cuda", requires_grad=True)
                layer(probe).square().mean().backward()
                torch.cuda.synchronize()
            name = torch.cuda.get_device_name(0)
            del layer, probe
            return torch.device("cuda"), f"CUDA convolution/backward probe passed: {name}"
        except (RuntimeError, AssertionError) as exc:
            reason = f"CUDA compatibility probe failed: {exc}"
    if requested == "cuda":
        raise RuntimeError(
            reason + ". Select --device cpu, or install a PyTorch/CUDA build supporting your GPU."
        )
    warnings.warn(reason + "; --device auto falls back to CPU", RuntimeWarning)
    return torch.device("cpu"), reason + "; CPU fallback"


class SyntheticDefocusDataset(Dataset):
    """Fixed scenes and fixed noise, grouped by source scene before degradation.

    count is the number of clean source scenes. Each is crossed with every
    supplied defocus value. Different split names have disjoint source seeds.
    No random split is performed after making degraded image variants.
    """

    def __init__(self, optics, split, count, defocus_values, noise_std=0.005,
                 base_seed=20261019, condition_scale=2.0):
        if split == "validation":
            split = "val"
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train, val or test")
        if not isinstance(count, int) or not 1 <= count < 1_000_000:
            raise ValueError("scene count must be an integer in [1, 1000000)")
        if not math.isfinite(noise_std) or noise_std < 0:
            raise ValueError("noise_std must be finite and non-negative")
        if not math.isfinite(condition_scale) or condition_scale <= 0:
            raise ValueError("condition_scale must be finite and positive")
        self.optics = optics
        self.split = split
        self.count = count
        self.defocus_values = parse_defocus(",".join(str(x) for x in defocus_values))
        self.noise_std = float(noise_std)
        self.base_seed = int(base_seed)
        self.condition_scale = float(condition_scale)
        self.psfs = {w: make_psf(optics, defocus_waves=w) for w in self.defocus_values}
        self._scenes = {}
        split_id = {"train": 0, "val": 1, "test": 2}[split]
        self.records = []
        for index in range(count):
            source_seed = scene_seed(split, index, base_seed=base_seed)
            for w_index, waves in enumerate(self.defocus_values):
                noise_seed = int(np.random.SeedSequence(
                    [base_seed, split_id, index, w_index, 0xDEF0]
                ).generate_state(1)[0])
                self.records.append({
                    "split": split, "scene_index": index,
                    "scene_seed": source_seed, "noise_seed": noise_seed,
                    "defocus_waves": waves,
                })

    def __len__(self):
        return len(self.records)

    def arrays(self, index):
        record = self.records[index]
        source_seed = record["scene_seed"]
        if source_seed not in self._scenes:
            self._scenes[source_seed] = np.asarray(
                make_scene(size=self.optics.size, seed=source_seed), dtype=np.float32
            )
        target = self._scenes[source_seed]
        psf = self.psfs[record["defocus_waves"]]
        observation = np.asarray(observe(
            target, psf, noise_std=self.noise_std, seed=record["noise_seed"]
        ), dtype=np.float32)
        return observation, target, psf

    def __getitem__(self, index):
        observation, target, _ = self.arrays(index)
        observed_tensor = torch.from_numpy(observation).unsqueeze(0).unsqueeze(0)
        model_input = conditioned_input_local(
            observed_tensor, self.records[index]["defocus_waves"], self.condition_scale
        )[0]
        return model_input, torch.from_numpy(target).unsqueeze(0)

    def manifest(self):
        return {
            "optics": asdict(self.optics), "split": self.split,
            "source_scene_count": self.count, "sample_count": len(self),
            "base_seed": self.base_seed, "noise_std": self.noise_std,
            "defocus_values": list(self.defocus_values),
            "condition_scale": self.condition_scale,
            "source": "procedural linear-intensity teaching scenes",
            "records": self.records,
        }


# An alias keeps the data helper dependency explicit without a second formula.
from .model import conditioned_input as conditioned_input_local


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def save_checkpoint(path, model, optics, training, best_epoch, val_mse, runtime):
    """Serialize tensors and built-in values only, never custom Python objects."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": 1,
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "model_config": asdict(model.config), "optics": asdict(optics),
        "training": training, "best_epoch": int(best_epoch),
        "val_mse": float(val_mse), "runtime": runtime, "task": TASK_DESCRIPTION,
    }, path)


def load_checkpoint(path, device="cpu"):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format; use this project's best.pt")
    model = ConditionalUNet(ModelConfig(**payload["model_config"]))
    optics = OpticsConfig(**payload["optics"])
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, optics, payload
