"""一个用于 CIFAR-10 的小型卷积神经网络（CNN）。

CIFAR-10 是 32x32 的 RGB 彩色图，共 10 类（飞机/汽车/鸟/猫/鹿/狗/青蛙/马/船/卡车）。
"""

import torch
import torch.nn as nn


class CIFAR10CNN(nn.Module):
    """VGG 风格的小 CNN，足够小，几秒一个 epoch，但能学到 ~85%+ 的准确率。"""

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # 特征提取部分：卷积 -> 激活 -> 归一化，堆叠后下采样
        self.features = nn.Sequential(
            # block 1：3 通道 -> 32 通道，尺寸保持 32x32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),          # 32x32 -> 16x16
            nn.Dropout(0.25),

            # block 2：32 通道 -> 64 通道，16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),          # 16x16 -> 8x8
            nn.Dropout(0.25),
        )

        # 分类部分：把卷积得到的 64x8x8 特征图拍平，接全连接层输出 10 个分数
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入 x 形状为 (batch, 3, 32, 32)，输出 (batch, 10) 的类别分数。"""
        return self.classifier(self.features(x))
