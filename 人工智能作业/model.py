"""简化 YOLOv1：64×64 灰度图 -> (N, 7, 7, 20)。

与教师数据一致，输出框采用整幅图归一化的 (cx, cy, w, h)。
类别输出保留为 logits，供交叉熵使用；坐标和置信度经过 sigmoid。
"""

import math

import torch
from torch import nn


class YOLOv1(nn.Module):
    def __init__(self, S=7, B=2, C=10, in_channels=1):
        super().__init__()
        if min(S, B, C, in_channels) < 1:
            raise ValueError("S、B、C 和输入通道数必须为正整数")
        self.S, self.B, self.C = S, B, C
        self.out_dim = S * S * (5 * B + C)

        # 三个卷积块：64 -> 32 -> 16 -> 8。
        layers = []
        channels = in_channels
        for out_channels in (32, 64, 128):
            layers.extend([
                nn.Conv2d(channels, out_channels, 3, padding=1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.MaxPool2d(2),
            ])
            channels = out_channels
        self.backbone = nn.Sequential(*layers)
        with torch.no_grad():
            self.feat_dim = self.backbone(
                torch.zeros(1, in_channels, 64, 64)
            ).numel()
        self.head = nn.Sequential(
            nn.Linear(self.feat_dim, 512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, self.out_dim),
        )

        gy, gx = torch.meshgrid(torch.arange(S), torch.arange(S), indexing="ij")
        self.register_buffer("grid", torch.stack((gx, gy), dim=-1).float()
                             .view(1, S, S, 1, 2))
        # 数字实际宽高约 8~14 像素；初始框和低目标概率有助于稳定训练。
        with torch.no_grad():
            bias = self.head[-1].bias.view(S, S, 5 * B + C)
            bias.zero_()
            for box_id in range(B):
                bias[..., box_id * 5 + 2:box_id * 5 + 4] = math.log(0.17 / 0.83)
                bias[..., box_id * 5 + 4] = math.log(0.05 / 0.95)

    def forward(self, x):
        features = self.backbone(x)
        raw = self.head(torch.flatten(features, start_dim=1))
        raw = raw.reshape(x.size(0), self.S, self.S, 5 * self.B + self.C)
        boxes = raw[..., :5 * self.B].reshape(
            x.size(0), self.S, self.S, self.B, 5
        )
        # 先预测网格内偏移，再解码为全图坐标；损失、NMS 不再二次解码。
        xy = (boxes[..., :2].sigmoid() + self.grid) / self.S
        wh = boxes[..., 2:4].sigmoid()
        confidence = boxes[..., 4:5].sigmoid()
        decoded = torch.cat((xy, wh, confidence), dim=-1)
        return torch.cat((decoded.flatten(-2), raw[..., 5 * self.B:]), dim=-1)
