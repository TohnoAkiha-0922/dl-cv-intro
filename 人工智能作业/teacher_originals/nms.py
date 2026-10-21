"""
nms.py
非极大值抑制（NMS）。
学生需要实现核心逻辑。
"""

import torch


def nms(pred, conf_thresh=0.5, iou_thresh=0.5):
    """
    对单个样本的预测结果做 NMS。

    参数：
        pred: (S, S, 5B+C) 的预测张量
        conf_thresh: 置信度阈值，低于此值的框丢弃
        iou_thresh: IoU 阈值，高于此值的重叠框只保留分数最高的

    返回：
        boxes: list of (x, y, w, h, conf, cls_id)
               均为归一化坐标，cls_id 为 0-9 的整数
    """
    S = pred.shape[0]
    B = 2
    C = 10

    boxes = []

    # ============ TODO 1: 收集所有候选框 ============
    # 遍历每个网格，对每个预测框：
    #   - 读取 (x, y, w, h, conf)
    #   - 如果 conf > conf_thresh，加入 boxes 列表
    #   - 类别取 pred 中类别概率最大的那个
    # 提示：pred[gy, gx, :5*B] 是框信息，pred[gy, gx, 5*B:] 是类别
    # TODO

    if len(boxes) == 0:
        return []

    # ============ TODO 2: 按置信度排序 ============
    # 将 boxes 按 conf 从高到低排序
    # TODO

    # ============ TODO 3: 逐个筛选 ============
    # 遍历排序后的 boxes：
    #   - 保留当前框
    #   - 计算它与剩余框的 IoU
    #   - 去掉 IoU > iou_thresh 的框
    # 提示：可以复用 loss.py 中的 compute_iou，或在这里重新实现
    # TODO

    return boxes