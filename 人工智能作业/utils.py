"""检测结果、训练曲线和网络结构可视化（无窗口环境也可运行）。"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D


def draw_box(ax, x, y, w, h, color, label=None):
    """绘制像素坐标的中心点格式框；由调用方将归一化坐标换算为像素。"""
    ax.add_patch(patches.Rectangle(
        (x - w / 2, y - h / 2), w, h, linewidth=1.3,
        edgecolor=color, facecolor="none",
    ))
    if label is not None:
        ax.text(x - w / 2, y - h / 2 - 0.8, label, color=color,
                fontsize=7, weight="bold", bbox=dict(facecolor="black", alpha=0.65, pad=0.6))


def label_to_boxes(label, B=2):
    """每个有目标网格只返回一个真值，忽略重复的 B 个标签槽。"""
    label = label.detach().cpu()
    result = []
    for gy, gx in (label[..., 4] > 0.5).nonzero().tolist():
        result.append((*label[gy, gx, :4].tolist(),
                       int(label[gy, gx, 5 * B:].argmax())))
    return result


def _plot_sample(ax, sample, title):
    image, label, boxes = sample
    img = image.detach().cpu().squeeze(0).numpy()
    height, width = img.shape
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    for x, y, w, h, cls_id in label_to_boxes(label):
        draw_box(ax, x * width, y * height, w * width, h * height,
                 "#39db85", f"GT {cls_id}")
    for x, y, w, h, confidence, cls_id in boxes:
        draw_box(ax, x * width, y * height, w * width, h * height,
                 "#ff635e", f"{cls_id}:{confidence:.2f}")
    ax.set(xlim=(-4, width + 3), ylim=(height + 3, -5), title=title)
    ax.axis("off")


def visualize_predictions(samples, save_path="predictions.png", titles=None):
    if not samples:
        raise ValueError("没有可以绘制的样本")
    n = min(len(samples), 8)
    columns = min(n, 4)
    rows = (n + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4.2 * rows),
                             squeeze=False)
    for i, ax in enumerate(axes.flat):
        if i < n:
            _plot_sample(ax, samples[i], titles[i] if titles else f"Sample {i + 1}")
        else:
            ax.axis("off")
    fig.legend(handles=[Line2D([0], [0], color="#39db85", label="Ground truth"),
                        Line2D([0], [0], color="#ff635e", label="Prediction: digit / confidence")],
               loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_loss_curves(history, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    epochs = [row["epoch"] for row in history]
    groups = [("Loss components", ["loss", "conf", "cls"]),
              ("Coordinate loss", ["coord"]),
              ("Training and validation", ["loss", "val_loss"])]
    colors = {"loss": "#193c65", "conf": "#bf7037", "cls": "#54836c",
              "coord": "#9364a3", "val_loss": "#c45454"}
    for ax, (title, keys) in zip(axes, groups):
        for key in keys:
            ax.plot(epochs, [row[key] for row in history], label=key, color=colors[key])
        ax.set(title=title, xlabel="Epoch", ylabel="Loss per image")
        ax.grid(alpha=0.18)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_network(save_path):
    labels = ["Input\n1 x 64 x 64", "Conv 3x3 / 32\nLeakyReLU + Pool\n32 x 32 x 32",
              "Conv 3x3 / 64\nLeakyReLU + Pool\n64 x 16 x 16",
              "Conv 3x3 / 128\nLeakyReLU + Pool\n128 x 8 x 8",
              "Flatten\n8192", "Linear 512\nLeakyReLU\nDropout 0.1",
              "Linear 980\nBox decoding\n7 x 7 x 20"]
    fig, ax = plt.subplots(figsize=(14, 2.3))
    for i, label in enumerate(labels):
        x = i * 1.95
        ax.add_patch(patches.FancyBboxPatch((x, 0.35), 1.65, 1.25,
                     boxstyle="round,pad=0.06", facecolor="#edf2f7", edgecolor="#52718e"))
        ax.text(x + 0.825, 0.975, label, ha="center", va="center", fontsize=9)
        if i < len(labels) - 1:
            ax.annotate("", xy=(x + 1.9, 0.975), xytext=(x + 1.68, 0.975),
                        arrowprops=dict(arrowstyle="->", color="#52718e"))
    ax.set(xlim=(-0.2, 13.7), ylim=(0.1, 1.85))
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
