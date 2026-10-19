"""Shared, headless figures and explicit evaluation conventions."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .metrics import psnr, ssim


def image_metrics(reference, prediction):
    """All displayed-method metrics use the same fixed [0, 1] clipping."""
    display = np.clip(prediction, 0.0, 1.0)
    return {"psnr_db": float(psnr(reference, display)),
            "ssim": float(ssim(reference, display))}


def save_image(path, array):
    Image.fromarray(np.round(np.clip(array, 0, 1) * 255).astype(np.uint8)).save(path)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, records):
    if not records:
        raise ValueError("Cannot write an empty experiment table.")
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def demo_figure(path, clean, pupil, psf, blurred, observed, restored, metadata):
    """Pupil phase and intensity PSF are distinct physical quantities."""
    pitch = metadata["optics"]["pixel_pitch_um"]
    size = clean.shape[0]
    # Extents describe pixel edges; centre index size//2 lies exactly at zero.
    spatial_edges = (np.array([0, size]) - size // 2 - 0.5) * pitch
    extent = [*spatial_edges, *spatial_edges]
    frequencies = np.fft.fftshift(np.fft.fftfreq(size, d=pitch))
    delta_frequency = 1.0 / (size * pitch)
    frequency_edges = [frequencies[0] - delta_frequency / 2,
                       frequencies[-1] + delta_frequency / 2]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    panels = [
        (clean, "Original linear intensity", "gray", 0, 1),
        (np.ma.masked_where(abs(pupil) == 0, np.angle(pupil)),
         "Pupil phase [rad], wrapped", "twilight", -np.pi, np.pi),
        (np.log10(np.maximum(psf / psf.max(), 1e-6)),
         "Intensity PSF: log10(h / max(h))", "magma", -6, 0),
        (blurred, "Blurred, before noise", "gray", 0, 1),
        (observed, f"Observation: {metadata['observed']['psnr_db']:.2f} dB", "gray", 0, 1),
        (restored, f"Wiener: {metadata['wiener']['psnr_db']:.2f} dB", "gray", 0, 1),
    ]
    for i, (ax, panel) in enumerate(zip(axes.flat, panels)):
        array, title, cmap, low, high = panel
        if i == 1:
            im = ax.imshow(array, cmap=cmap, vmin=low, vmax=high,
                           extent=[*frequency_edges, *frequency_edges], origin="lower")
            ax.set_xlabel("f_x [cycles/um]")
            ax.set_ylabel("f_y [cycles/um]")
        else:
            im = ax.imshow(array, cmap=cmap, vmin=low, vmax=high,
                           extent=extent, origin="lower")
            ax.set_xlabel("x [um]"); ax.set_ylabel("y [um]")
        ax.set_title(title, fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Known-PSF defocus restoration | defocus = {metadata['defocus_waves']:g} waves\n"
                 "Synthetic scalar incoherent model; periodic boundary; display clipped to [0, 1]",
                 fontsize=13)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def sweep_figure(path, records):
    levels = sorted(set(row["defocus_waves"] for row in records))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    for method, label in [("observed", "Observation"), ("wiener", "Wiener")]:
        for ax, key, ylabel in [(axes[0], "psnr_db", "Mean PSNR [dB]"),
                                (axes[1], "ssim", "Mean SSIM [unitless]")]:
            means = [np.mean([r[key] for r in records
                              if r["method"] == method and r["defocus_waves"] == w])
                     for w in levels]
            ax.plot(levels, means, marker="o", label=label)
            ax.set_xlabel("Pupil-edge defocus OPD [waves]")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.2); ax.legend()
    fig.suptitle("Defocus sensitivity on independent synthetic test scenes\n"
                 "Mean over scenes; all predictions clipped to [0, 1]; source: this run")
    fig.savefig(path, dpi=160)
    plt.close(fig)
