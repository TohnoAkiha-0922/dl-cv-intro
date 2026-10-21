"""
loss.py
YOLOv1 多任务损失函数。
包含：坐标损失、置信度损失、类别损失。
学生需要实现核心的匹配逻辑和损失计算。
"""

import torch
import torch.nn as nn


def compute_iou(box1, box2):
    """
    计算两个框的 IoU。
    框格式：(x_center, y_center, w, h)，均已归一化。

    TODO: 实现 IoU 计算
    提示：
        1. 把中心点+宽高 转成 左上角+右下角
        2. 计算交集面积
        3. 计算并集面积
        4. 返回 交集/并集
    """
    raise NotImplementedError("请实现 compute_iou")


class YOLOv1Loss(nn.Module):
    def __init__(self, S=7, B=2, C=10,
                 lambda_coord=5.0, lambda_noobj=0.5):
        """
        参数：
            S, B, C: 与模型一致
            lambda_coord: 坐标损失权重
            lambda_noobj: 无目标框的置信度损失权重
        """
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, pred, target):
        """
        参数：
            pred:   (batch, S, S, 5B+C) 模型输出
            target: (batch, S, S, 5B+C) 真实标签
        返回：
            total_loss, coord_loss, conf_loss, cls_loss
        """
        batch = pred.size(0)
        device = pred.device

        # 拆分预测
        # pred 形状 (B, S, S, 5*B+C)
        # 前 5*B 是框信息，后 C 是类别
        pred_boxes = pred[..., :5 * self.B].view(batch, self.S, self.S, self.B, 5)
        pred_cls = pred[..., 5 * self.B:]

        # 拆分目标
        target_boxes = target[..., :5 * self.B].view(batch, self.S, self.S, self.B, 5)
        target_cls = target[..., 5 * self.B:]

        # ============ TODO 1: 构建负责框的 mask ============
        # 对每个网格，找出与真实框 IoU 最大的预测框，作为"负责框"
        # 提示：
        #   - target_boxes[..., 4] 是 conf，只有负责目标的网格 conf=1
        #   - 对每个网格，取 target 中 conf=1 的框作为真实框
        #   - 计算每个预测框与真实框的 IoU，取最大的那个
        #   - 构建 obj_mask（形状 (batch, S, S)）和 noobj_mask
        obj_mask = None   # TODO
        noobj_mask = None # TODO

        # ============ TODO 2: 坐标损失 ============
        # 只对负责框计算 (x,y,w,h) 的 MSE
        # 建议 x,y 用较小权重，w,h 用较大权重（因为 w,h 对误差更敏感）
        # 或者直接用 lambda_coord 统一加权
        coord_loss = None  # TODO

        # ============ TODO 3: 置信度损失 ============
        # 有目标的框：conf 应接近 1
        # 无目标的框：conf 应接近 0，用 lambda_noobj 加权
        conf_loss = None  # TODO

        # ============ TODO 4: 类别损失 ============
        # 只对负责的网格计算交叉熵
        cls_loss = None  # TODO

        total_loss = coord_loss + conf_loss + cls_loss
        return total_loss, coord_loss, conf_loss, cls_loss