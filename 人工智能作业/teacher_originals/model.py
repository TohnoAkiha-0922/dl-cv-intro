"""
model.py
简化 YOLOv1 模型。
backbone: 若干 Conv-LeakyReLU-MaxPool 块
head: 全连接层，输出 S*S*(5B+C) 维向量
"""

import torch
import torch.nn as nn


class YOLOv1(nn.Module):
    def __init__(self, S=7, B=2, C=10, in_channels=1):
        """
        参数：
            S: 网格数
            B: 每个网格预测的框数
            C: 类别数
            in_channels: 输入通道数（灰度图为 1）
        """
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.out_dim = S * S * (5 * B + C)  # 7*7*20 = 980

        # ============ TODO 1: 补全 backbone ============
        # 建议 3-5 个卷积块，每个块：Conv2d -> LeakyReLU -> MaxPool2d
        # 输入 (1, 64, 64)，经过若干次下采样后，得到一个特征图
        # 例如：
        #   64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4
        # 最后一层通道数建议 128 或 256
        self.backbone = nn.Sequential(
            # TODO: 在这里写你的卷积层
            # 示例：
            # nn.Conv2d(in_channels, 32, 3, padding=1),
            # nn.LeakyReLU(0.1),
            # nn.MaxPool2d(2),
            # ...
        )

        # ============ TODO 2: 计算展平后的特征维度 ============
        # 你需要根据 backbone 的输出尺寸，算出 flatten 后的维度
        # 然后用它作为全连接层的输入
        # 提示：可以用一个假输入测试一下：x = torch.randn(1, in_channels, 64, 64)
        #       feat = self.backbone(x); print(feat.shape)
        self.feat_dim = None  # TODO: 替换为实际维度

        # ============ TODO 3: 补全头部 ============
        # 全连接层，把 feat_dim 映射到 out_dim
        # 建议：Linear(feat_dim, 512) -> LeakyReLU -> Linear(512, out_dim)
        self.head = nn.Sequential(
            # TODO: 在这里写你的全连接层
        )

    def forward(self, x):
        """
        输入：x (batch, in_channels, 64, 64)
        输出：(batch, S, S, 5B+C)
        """
        # TODO 4: 实现 forward
        # 1. 过 backbone
        # 2. flatten
        # 3. 过 head
        # 4. reshape 为 (batch, S, S, 5B+C)
        raise NotImplementedError("请实现 forward")