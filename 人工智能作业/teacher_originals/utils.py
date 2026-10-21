"""
utils.py
可视化工具：在图像上画边界框。
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def draw_box(ax, x, y, w, h, color, label=None):
    """
    在 ax 上画一个框。
    x, y 为中心点，w, h 为宽高，均为归一化坐标。
    需要乘以图像尺寸转换为像素坐标。
    """
    # 这里假设图像是 64x64，调用方传入的是归一化坐标
    # 实际使用时由调用方换算
    rect = patches.Rectangle(
        (x - w / 2, y - h / 2), w, h,
        linewidth=2, edgecolor=color, facecolor='none'
    )
    ax.add_patch(rect)
    if label is not None:
        ax.text(x - w / 2, y - h / 2 - 2, label,
                color=color, fontsize=8, weight='bold')


def visualize_predictions(samples, save_path='predictions.png'):
    """
    可视化预测结果。
    samples: list of (image, label, boxes)
        image: (1, 64, 64) 张量
        label: (S, S, 5B+C) 张量
        boxes: nms 返回的列表
    """
    n = min(len(samples), 8)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i in range(n):
        image, label, boxes = samples[i]
        img = image.squeeze(0).numpy()
        ax = axes[i]
        ax.imshow(img, cmap='gray')

        H, W = img.shape
        S, B, C = 7, 2, 10

        # 画真实框（绿色）
        for gy in range(S):
            for gx in range(S):
                conf = label[gy, gx, 4]
                if conf > 0.5:
                    cx = label[gy, gx, 0].item() * W
                    cy = label[gy, gx, 1].item() * H
                    bw = label[gy, gx, 2].item() * W
                    bh = label[gy, gx, 3].item() * H
                    draw_box(ax, cx, cy, bw, bh, 'green', 'GT')

        # 画预测框（红色）
        for box in boxes:
            x, y, w, h, conf, cls_id = box
            cx = x * W
            cy = y * H
            bw = w * W
            bh = h * H
            draw_box(ax, cx, cy, bw, bh, 'red', f'{cls_id}:{conf:.2f}')

        ax.set_title(f'Sample {i+1}')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f'可视化结果已保存到 {save_path}')