"""IoU 匹配与简化 YOLOv1 多任务损失。"""

import torch
from torch import nn
from torch.nn import functional as F


def compute_iou(box1, box2):
    """计算中心点格式框的 IoU；支持 (..., 4) 张量及广播。

    零面积/不相交框的结果为 0，负宽高按 0 处理，避免无效交并比。
    """
    if not isinstance(box1, torch.Tensor):
        if isinstance(box2, torch.Tensor):
            box1 = torch.as_tensor(box1, device=box2.device, dtype=box2.dtype)
        else:
            box1 = torch.as_tensor(box1, dtype=torch.float32)
    if not box1.is_floating_point():
        box1 = box1.float()
    box2 = torch.as_tensor(box2, device=box1.device, dtype=box1.dtype)
    size1 = box1[..., 2:4].clamp_min(0)
    size2 = box2[..., 2:4].clamp_min(0)
    left_top = torch.maximum(box1[..., :2] - size1 / 2,
                             box2[..., :2] - size2 / 2)
    right_bottom = torch.minimum(box1[..., :2] + size1 / 2,
                                 box2[..., :2] + size2 / 2)
    intersection = (right_bottom - left_top).clamp_min(0).prod(dim=-1)
    union = size1.prod(dim=-1) + size2.prod(dim=-1) - intersection
    # 浮点舍入可能使完全相同框的结果略大于 1，显式保持 IoU 的定义域。
    return (intersection / union.clamp_min(torch.finfo(box1.dtype).eps)).clamp(0, 1)


class YOLOv1Loss(nn.Module):
    def __init__(self, S=7, B=2, C=10, lambda_coord=5.0,
                 lambda_noobj=0.5, xy_weight=1.0, wh_weight=2.0):
        super().__init__()
        self.S, self.B, self.C = S, B, C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.xy_weight, self.wh_weight = xy_weight, wh_weight

    def forward(self, pred, target):
        """返回加权后的 total、coord、conf、cls，均按图像数归一化。

        target 中两个框可能重复 conf=1，但每个有目标网格只选一个预测框
        负责。另一预测框和空网格中的框均计入无目标置信度损失。
        """
        expected = (self.S, self.S, 5 * self.B + self.C)
        if pred.shape != target.shape or tuple(pred.shape[1:]) != expected:
            raise ValueError(f"预测和标签必须同为 (N, {expected})")
        batch = pred.size(0)
        if batch == 0:
            raise ValueError("不能计算空 batch 的损失")
        pred_boxes = pred[..., :5 * self.B].reshape(
            batch, self.S, self.S, self.B, 5
        )
        target_boxes = target[..., :5 * self.B].reshape_as(pred_boxes)
        target_cls = target[..., 5 * self.B:]
        obj_mask = target_boxes[..., 4].amax(dim=-1) > 0.5
        # 从目标中选 conf 最大的一项，兼容只填一个框和重复两个框的标签。
        gt_index = target_boxes[..., 4].argmax(dim=-1)
        gt = target_boxes.gather(
            -2, gt_index[..., None, None].expand(*gt_index.shape, 1, 5)
        ).squeeze(-2)
        with torch.no_grad():
            ious = compute_iou(pred_boxes[..., :4], gt[..., None, :4])
            best_index = ious.argmax(dim=-1)
            responsible = F.one_hot(best_index, self.B).bool() & obj_mask[..., None]
        noobj_mask = ~responsible

        xy_error = (pred_boxes[..., :2] - gt[..., None, :2]).square().sum(-1)
        wh_error = (pred_boxes[..., 2:4] - gt[..., None, 2:4]).square().sum(-1)
        coord_loss = self.lambda_coord * (
            (self.xy_weight * xy_error + self.wh_weight * wh_error)
            * responsible
        ).sum() / batch

        confidence = pred_boxes[..., 4]
        conf_loss = (((confidence - 1).square() * responsible).sum()
                     + self.lambda_noobj * (confidence.square() * noobj_mask).sum()) / batch
        if obj_mask.any():
            cls_loss = F.cross_entropy(
                pred[..., 5 * self.B:][obj_mask],
                target_cls[obj_mask].argmax(dim=-1), reduction="sum"
            ) / batch
        else:
            # 与计算图相连的 0，使全空白 batch 仍可安全反向传播。
            cls_loss = pred[..., 5 * self.B:].sum() * 0.0
        total_loss = coord_loss + conf_loss + cls_loss
        return total_loss, coord_loss, conf_loss, cls_loss
