"""
train.py
训练与评估脚本。
学生需要调整超参数（batch_size, lr, epochs），并运行训练。
"""

import torch
from torch.utils.data import DataLoader
from create_dataset import DigitDetectionDataset
from model import YOLOv1
from loss import YOLOv1Loss
from nms import nms
from utils import visualize_predictions


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_coord = 0.0
    total_conf = 0.0
    total_cls = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        pred = model(images)
        loss, coord, conf, cls = criterion(pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_coord += coord.item()
        total_conf += conf.item()
        total_cls += cls.item()

    n = len(loader)
    return total_loss / n, total_coord / n, total_conf / n, total_cls / n


def evaluate(model, loader, device, conf_thresh=0.5, iou_thresh=0.5):
    """
    在测试集上评估，返回可视化所需的样本。
    """
    model.eval()
    samples = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            pred = model(images)
            # 对每个样本做 NMS
            for i in range(images.size(0)):
                boxes = nms(pred[i], conf_thresh=conf_thresh, iou_thresh=iou_thresh)
                samples.append((images[i].cpu(), labels[i].cpu(), boxes))
            if len(samples) >= 8:
                break
    return samples


def main():
    # ============ TODO: 调整超参数 ============
    batch_size = 16      # 根据笔记本内存调整，可从 8 或 16 开始
    lr = 1e-3            # 学习率
    epochs = 30          # 训练轮数，20-50 应该能看到效果
    S, B, C = 7, 2, 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 数据集
    train_set = DigitDetectionDataset(train=True, S=S, B=B, C=C, seed=42)
    test_set = DigitDetectionDataset(train=False, S=S, B=B, C=C, seed=123)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # 模型、损失、优化器
    model = YOLOv1(S=S, B=B, C=C).to(device)
    criterion = YOLOv1Loss(S=S, B=B, C=C)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 训练循环
    for epoch in range(epochs):
        loss, coord, conf, cls = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        print(f'Epoch {epoch+1}/{epochs} | '
              f'Loss {loss:.4f} | Coord {coord:.4f} | '
              f'Conf {conf:.4f} | Cls {cls:.4f}')

    # 保存权重
    torch.save(model.state_dict(), 'yolov1_digit.pth')
    print('模型已保存到 yolov1_digit.pth')

    # 评估与可视化
    samples = evaluate(model, test_loader, device)
    visualize_predictions(samples, save_path='predictions.png')
    print('可视化结果已保存到 predictions.png')


if __name__ == '__main__':
    main()