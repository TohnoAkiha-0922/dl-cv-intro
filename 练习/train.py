"""训练 CIFAR-10 分类器。

完整流程：加载数据 -> 定义模型 -> 循环 {前向传播, 算损失, 反向传播, 更新参数} -> 验证。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import CIFAR10CNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 1e-3

# CIFAR-10 的逐通道均值/标准差，用来做标准化，让训练更稳、更快收敛
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.1770)


def get_loaders():
    # 训练时做数据增强（随机裁剪 + 水平翻转），测试时只做标准化
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()          # 清空上一步的梯度
        outputs = model(images)        # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()                # 反向传播，算出每个参数的梯度
        optimizer.step()               # 按梯度更新参数

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


def main():
    print(f"Using device: {DEVICE}")
    train_loader, test_loader = get_loaders()

    model = CIFAR10CNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, test_loader, criterion)
        print(f"Epoch {epoch:2d}/{EPOCHS}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

    torch.save(model.state_dict(), "cifar10_cnn.pth")
    print(f"\nDone. Model saved to cifar10_cnn.pth (final val_acc={val_acc:.4f})")


if __name__ == "__main__":
    main()
