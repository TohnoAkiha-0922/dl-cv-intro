"""按类别执行非极大值抑制，使用全图归一化中心点格式框。"""

import torch

from loss import compute_iou


def nms(pred, conf_thresh=0.5, iou_thresh=0.5, B=2, C=10):
    """(S,S,5B+C) -> [(cx,cy,w,h,confidence,class_id), ...]。

    按题目要求用框的目标置信度筛选/排序，类别由 logits 的 argmax 决定。
    仅抑制同类别重叠框；不同数字即使相互遮挡也应保留。
    """
    if not 0 <= conf_thresh <= 1 or not 0 <= iou_thresh <= 1:
        raise ValueError("置信度和 IoU 阈值必须在 [0,1] 内")
    if pred.ndim != 3 or pred.shape[0] != pred.shape[1] or pred.shape[-1] != 5 * B + C:
        raise ValueError("pred 形状必须为 (S, S, 5*B+C)")
    pred = pred.detach()
    candidate_boxes = pred[..., :5 * B].reshape(-1, B, 5)
    classes = pred[..., 5 * B:].argmax(-1).reshape(-1, 1).expand(-1, B)
    valid = (torch.isfinite(candidate_boxes).all(-1)
             & (candidate_boxes[..., 4] > conf_thresh)
             & (candidate_boxes[..., 2:4] > 0).all(-1))
    boxes = candidate_boxes[valid]
    class_ids = classes[valid]
    if boxes.numel() == 0:
        return []
    # NMS 最多 98 个框，转到 CPU，避免反复触发 GPU/CPU 同步。
    boxes = boxes.cpu()
    class_ids = class_ids.cpu()
    order = boxes[:, 4].argsort(descending=True, stable=True)
    keep = []
    while order.numel():
        current = int(order[0])
        keep.append(current)
        rest = order[1:]
        if not rest.numel():
            break
        overlap = compute_iou(boxes[current, :4], boxes[rest, :4])
        suppress = (class_ids[rest] == class_ids[current]) & (overlap > iou_thresh)
        order = rest[~suppress]
    return [(*[float(v) for v in boxes[i]], int(class_ids[i])) for i in keep]
