"""加载训练权重进行快速演示，不修改正式测试指标。"""

import argparse
import json
from pathlib import Path

import torch

from create_dataset import DigitDetectionDataset
from model import YOLOv1
from nms import nms
from utils import visualize_predictions

PROJECT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=PROJECT / "outputs/yolov1_digit.pth")
    parser.add_argument("--data-root", type=Path, default=PROJECT / "data")
    parser.add_argument("--output", type=Path, default=PROJECT / "demo_predictions.png")
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--conf-thresh", type=float, default=None)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if not 1 <= args.count <= 8:
        parser.error("count 必须在 1~8 内")
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    threshold = args.conf_thresh
    if threshold is None:
        metrics_file = args.weights.parent / "metrics.json"
        threshold = json.loads(metrics_file.read_text(encoding="utf-8"))["test"]["conf_thresh"] if metrics_file.exists() else 0.5
    model = YOLOv1().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    model.eval()
    dataset = DigitDetectionDataset(str(args.data_root), train=False,
                                    seed=args.seed, num_samples=args.count)
    pairs = [dataset[i] for i in range(args.count)]
    with torch.no_grad():
        predictions = model(torch.stack([image for image, _ in pairs]).to(device)).cpu()
    samples = []
    for i, ((image, label), prediction) in enumerate(zip(pairs, predictions)):
        boxes = nms(prediction, conf_thresh=threshold, iou_thresh=args.iou_thresh)
        samples.append((image, label, boxes))
        print(f"Sample {i}: {boxes}")
    visualize_predictions(samples, args.output,
                          [f"Test #{i}" for i in range(args.count)])
    print(f"Device: {device} | confidence threshold: {threshold} | saved: {args.output}")


if __name__ == "__main__":
    main()
