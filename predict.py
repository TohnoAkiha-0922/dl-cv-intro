"""用训练好的模型对一张图片做分类推理。

用法：python predict.py 图片路径
（默认用上级目录的 image.jpg）
"""

import sys

import torch
from torchvision import transforms
from PIL import Image

from model import CIFAR10CNN

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.1770)


def predict(image_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CIFAR10CNN().to(device)
    model.load_state_dict(torch.load("cifar10_cnn.pth", map_location=device))
    model.eval()

    # 和训练时测试集一样的预处理：缩放到 32x32、转张量、标准化
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)  # 加 batch 维度 (1,3,32,32)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        top3 = probs.topk(3)

    print(f"Predicting: {image_path}")
    for rank, (p, idx) in enumerate(zip(top3.values[0], top3.indices[0]), start=1):
        print(f"  {rank}. {CLASSES[idx]:<10s}  {p.item():.4f}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "../image.jpg"
    predict(path)
