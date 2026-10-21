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
                 transform=None, seed=None, num_samples=None,
                 mnist_slice=None):
        """
        参数：
            mnist_root: MNIST 数据存放路径
            train: True 使用训练集，False 使用测试集
            canvas_size: 画布大小（64）
            S: 网格数（7）
            B: 每个网格预测的框数（2）
            C: 类别数（10）
            max_digits: 每张图最多粘贴的数字个数
            max_scale: 相对原始 28×28 MNIST 图的最大缩放比例
            transform: 对最终画布图像的变换（可选）
            seed: 随机种子（可选）
            num_samples: 合成图数量；默认原始 MNIST 数量的一半
            mnist_slice: 可选的原始 MNIST (start, stop) 划分
        """
        self.canvas_size = canvas_size
        self.S = S
        self.B = B
        self.C = C
        self.max_digits = max_digits
        self.max_scale = max_scale
        self.transform = transform

        # 使用独立的、由样本索引确定的随机数，不重置模型/全局随机状态。
        # 同一 seed、idx 得到同一张图，验证集和测试集可以稳定复现。
        self.seed = seed if seed is not None else random.SystemRandom().randrange(2**31)
        self.num_samples = num_samples
        if S < 1 or B < 1 or C != 10 or canvas_size < 14 or max_digits < 0:
            raise ValueError("要求 S、B 为正，C=10，canvas_size>=14，max_digits>=0")
        if not 0.3 <= max_scale <= 1.0 or int(28 * max_scale) > canvas_size:
            raise ValueError("max_scale 必须在 [0.3,1] 内且数字能放入画布")
        if num_samples is not None and num_samples < 1:
            raise ValueError("num_samples 必须为正")

        # 加载 MNIST 原始数据（不做归一化，保持 0-255 便于粘贴）
        self.mnist = datasets.MNIST(
            root=mnist_root, train=train, download=True,
            transform=transforms.ToTensor()
        )
        if mnist_slice is not None:
            start, stop = mnist_slice
            self.mnist.data = self.mnist.data[start:stop]
            self.mnist.targets = self.mnist.targets[start:stop]
            if not len(self.mnist):
                raise ValueError("MNIST 划分不能为空")
        # ToTensor 会把像素缩到 [0,1]，粘贴时再乘 255 还原

    def __len__(self):
        # 合成数据集长度可以自定义，这里设为 MNIST 长度的一半
        return self.num_samples if self.num_samples is not None else len(self.mnist) // 2

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
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        rng = random.Random(self.seed + int(idx) * 1000003)
        canvas = np.zeros((self.canvas_size, self.canvas_size), dtype=np.float32)
        # 标签初始化为全 0
        label = np.zeros((self.S, self.S, 5 * self.B + self.C), dtype=np.float32)

        # 随机决定这张图上有几个数字
        n_digits = rng.randint(0, self.max_digits)
        skipped_collisions = 0

        for _ in range(n_digits):
            # 随机选一个 MNIST 样本
            mnist_idx = rng.randint(0, len(self.mnist) - 1)
            img, cls = self.mnist[mnist_idx]
            img = (img.squeeze(0).numpy() * 255).astype(np.float32)  # (28,28)

            # 随机缩放
            scale = rng.uniform(0.3, self.max_scale)
            new_size = max(8, int(28 * scale))
            pil_img = Image.fromarray(img.astype(np.uint8)).resize(
                (new_size, new_size), Image.Resampling.BILINEAR
            )
            digit_img = np.array(pil_img, dtype=np.float32)

            # 随机位置
            top = rng.randint(0, self.canvas_size - new_size)
            left = rng.randint(0, self.canvas_size - new_size)

            # YOLOv1 每格只有一组类别，无法表示同格两个目标。
            # 在粘贴前跳过冲突目标，避免可见数字与标签不一致。
            candidate_gx = min(int((left + new_size / 2) / self.canvas_size * self.S), self.S - 1)
            candidate_gy = min(int((top + new_size / 2) / self.canvas_size * self.S), self.S - 1)
            if label[candidate_gy, candidate_gx, 4] > 0.5:
                skipped_collisions += 1
                continue

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
            # 两个框槽都保存相同真值；loss.py 用 IoU 动态选择唯一负责框。
            base = 5 * self.B
            # 第一个框
            label[grid_y, grid_x, 0] = cx
            label[grid_y, grid_x, 1] = cy
            label[grid_y, grid_x, 2] = bw
            label[grid_y, grid_x, 3] = bh
            label[grid_y, grid_x, 4] = 1.0  # conf
            for box_id in range(1, self.B):
                offset = box_id * 5
                label[grid_y, grid_x, offset:offset + 5] = [cx, cy, bw, bh, 1.0]
            # 类别 one-hot
            label[grid_y, grid_x, base + int(cls)] = 1.0

        # 图像归一化到 [0,1]
        canvas = canvas / 255.0
        image = torch.from_numpy(canvas).unsqueeze(0)  # (1, H, W)

        if self.transform is not None:
            image = self.transform(image)

        label = torch.from_numpy(label)
        self.last_metadata = {
            "attempted_digits": n_digits, "skipped_collisions": skipped_collisions,
            "objects": int((label[..., 4] > 0.5).sum()),
        }
        return image, label
