"""
create_dataset.py
合成手写数字检测数据集。
从 MNIST 中随机选取数字图片，随机粘贴到空白画布上，生成 YOLO 格式标签。
"""

import random
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image
import numpy as np


class DigitDetectionDataset(Dataset):
    """
    合成手写数字检测数据集。

    每张图片：64x64 灰度画布，上面随机粘贴 0~max_digits 个 MNIST 数字。
    标签：形状 (S, S, 5*B + C) = (7, 7, 20)
        - 每个网格负责预测中心点落在该网格内的目标。
        - 5*B = 10：B=2 个框，每个框 (x, y, w, h, conf)
        - C = 10：类别 one-hot（0-9）
    """

    def __init__(self, mnist_root='./data', train=True,
                 canvas_size=64, S=7, B=2, C=10,
                 max_digits=5, max_scale=0.5,
                 transform=None, seed=None):
        """
        参数：
            mnist_root: MNIST 数据存放路径
            train: True 使用训练集，False 使用测试集
            canvas_size: 画布大小（64）
            S: 网格数（7）
            B: 每个网格预测的框数（2）
            C: 类别数（10）
            max_digits: 每张图最多粘贴的数字个数
            max_scale: 数字相对画布的最大缩放比例
            transform: 对最终画布图像的变换（可选）
            seed: 随机种子（可选）
        """
        self.canvas_size = canvas_size
        self.S = S
        self.B = B
        self.C = C
        self.max_digits = max_digits
        self.max_scale = max_scale
        self.transform = transform

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        # 加载 MNIST 原始数据（不做归一化，保持 0-255 便于粘贴）
        self.mnist = datasets.MNIST(
            root=mnist_root, train=train, download=True,
            transform=transforms.ToTensor()
        )
        # ToTensor 会把像素缩到 [0,1]，粘贴时再乘 255 还原

    def __len__(self):
        # 合成数据集长度可以自定义，这里设为 MNIST 长度的一半
        return len(self.mnist) // 2

    def _paste_digit(self, canvas, digit_img, top, left):
        """
        将 digit_img 粘贴到 canvas 的 (top, left) 位置。
        处理边界裁剪，返回实际粘贴区域的 (top, left, h, w)。
        """
        H, W = canvas.shape
        dh, dw = digit_img.shape
        # 计算实际可粘贴区域
        y1, y2 = max(0, top), min(H, top + dh)
        x1, x2 = max(0, left), min(W, left + dw)
        if y1 >= y2 or x1 >= x2:
            return None
        # 对应 digit_img 的裁剪区域
        dy1, dy2 = y1 - top, y2 - top
        dx1, dx2 = x1 - left, x2 - left
        patch = digit_img[dy1:dy2, dx1:dx2]
        # 取最大值粘贴，避免覆盖已有数字（简单策略）
        canvas[y1:y2, x1:x2] = np.maximum(canvas[y1:y2, x1:x2], patch)
        return y1, x1, y2 - y1, x2 - x1

    def __getitem__(self, idx):
        """
        返回：
            image: (1, canvas_size, canvas_size) 的 float32 张量
            label: (S, S, 5*B + C) 的 float32 张量
        """
        canvas = np.zeros((self.canvas_size, self.canvas_size), dtype=np.float32)
        # 标签初始化为全 0
        label = np.zeros((self.S, self.S, 5 * self.B + self.C), dtype=np.float32)

        # 随机决定这张图上有几个数字
        n_digits = random.randint(0, self.max_digits)

        for _ in range(n_digits):
            # 随机选一个 MNIST 样本
            mnist_idx = random.randint(0, len(self.mnist) - 1)
            img, cls = self.mnist[mnist_idx]
            img = (img.squeeze(0).numpy() * 255).astype(np.float32)  # (28,28)

            # 随机缩放
            scale = random.uniform(0.3, self.max_scale)
            new_size = max(8, int(28 * scale))
            pil_img = Image.fromarray(img.astype(np.uint8)).resize(
                (new_size, new_size), Image.BILINEAR
            )
            digit_img = np.array(pil_img, dtype=np.float32)

            # 随机位置
            top = random.randint(0, self.canvas_size - new_size)
            left = random.randint(0, self.canvas_size - new_size)

            # 粘贴
            result = self._paste_digit(canvas, digit_img, top, left)
            if result is None:
                continue
            y1, x1, h, w = result

            # 计算边界框（归一化到 [0,1]）
            cx = (x1 + w / 2.0) / self.canvas_size
            cy = (y1 + h / 2.0) / self.canvas_size
            bw = w / self.canvas_size
            bh = h / self.canvas_size

            # 确定中心点落在哪个网格
            grid_x = int(cx * self.S)
            grid_y = int(cy * self.S)
            grid_x = min(grid_x, self.S - 1)
            grid_y = min(grid_y, self.S - 1)

            # 写入标签
            # 每个网格的 5*B + C 维：
            # [x1,y1,w1,h1,conf1, x2,y2,w2,h2,conf2, cls0..cls9]
            # 这里两个框都填相同值（简化处理），conf 只在负责框上为 1
            base = 5 * self.B
            # 第一个框
            label[grid_y, grid_x, 0] = cx
            label[grid_y, grid_x, 1] = cy
            label[grid_y, grid_x, 2] = bw
            label[grid_y, grid_x, 3] = bh
            label[grid_y, grid_x, 4] = 1.0  # conf
            # 第二个框（简化：同样填值，conf 设 0，由模型自己学）
            label[grid_y, grid_x, 5] = cx
            label[grid_y, grid_x, 6] = cy
            label[grid_y, grid_x, 7] = bw
            label[grid_y, grid_x, 8] = bh
            label[grid_y, grid_x, 9] = 1.0  # 让两个框都学，简化训练
            # 类别 one-hot
            label[grid_y, grid_x, base + int(cls)] = 1.0

        # 图像归一化到 [0,1]
        canvas = canvas / 255.0
        image = torch.from_numpy(canvas).unsqueeze(0)  # (1, H, W)

        if self.transform is not None:
            image = self.transform(image)

        label = torch.from_numpy(label)
        return image, label