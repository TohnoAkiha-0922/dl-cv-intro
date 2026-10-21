"""可复现的训练、验证集选模、独立测试、结果导出。运行 python train.py --help。"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from create_dataset import DigitDetectionDataset
from model import YOLOv1
from loss import YOLOv1Loss, compute_iou
from nms import nms
from utils import label_to_boxes, visualize_predictions, plot_loss_curves, plot_network

PROJECT = Path(__file__).resolve().parent


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cache_dataset(dataset, name):
    """一次合成并缓存到内存，避免每个 epoch 重复 PIL 缩放的 CPU 开销。"""
    images, labels = [], []
    collisions = objects = empty = attempted = 0
    for i in range(len(dataset)):
        image, label = dataset[i]
        images.append(image)
        labels.append(label)
        info = dataset.last_metadata
        collisions += info["skipped_collisions"]
        objects += info["objects"]
        attempted += info["attempted_digits"]
        empty += int(info["objects"] == 0)
    stats = {"images": len(dataset), "objects": objects, "empty_images": empty,
             "attempted_digits": attempted, "skipped_collisions": collisions}
    print(f"{name}: {stats}", flush=True)
    return TensorDataset(torch.stack(images), torch.stack(labels)), stats


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    totals = np.zeros(4, dtype=np.float64)
    count = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        losses = criterion(model(images), labels)
        if not torch.isfinite(losses[0]):
            raise RuntimeError("训练损失出现 NaN/Inf")
        losses[0].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        totals += np.array([value.detach().item() for value in losses]) * images.size(0)
        count += images.size(0)
    if not count:
        raise ValueError("训练集为空")
    return tuple(totals / count)


@torch.no_grad()
def validation_loss(model, loader, criterion, device):
    model.eval()
    total, count = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        total += criterion(model(images), labels)[0].item() * images.size(0)
        count += images.size(0)
    return total / count


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    predictions = []
    for images, _ in loader:
        predictions.append(model(images.to(device)).cpu())
    return torch.cat(predictions)


def match_detections(boxes, ground_truth, iou_thresh=0.5):
    """同类别且 IoU>=阈值的一对一贪心匹配。每个真值最多匹配一次。"""
    used = set()
    hits, matched_ious = [], []
    for box in boxes:
        eligible = [i for i, gt in enumerate(ground_truth)
                    if i not in used and gt[4] == box[5]]
        if eligible:
            overlaps = compute_iou(torch.tensor(box[:4]),
                                   torch.tensor([ground_truth[i][:4] for i in eligible]))
            value, index = overlaps.max(dim=0)
            if value.item() >= iou_thresh:
                used.add(eligible[int(index)])
                hits.append(True)
                matched_ious.append(float(value))
                continue
        hits.append(False)
    tp = len(used)
    return tp, len(boxes) - tp, len(ground_truth) - tp, matched_ious, hits


def score_predictions(predictions, dataset, conf_thresh=0.5, iou_thresh=0.5):
    totals = [0, 0, 0]
    all_ious, records = [], []
    for idx, pred in enumerate(predictions):
        _, label = dataset[idx]
        boxes = nms(pred, conf_thresh=conf_thresh, iou_thresh=iou_thresh)
        ground_truth = label_to_boxes(label)
        tp, fp, fn, ious, hits = match_detections(boxes, ground_truth)
        totals = [a + b for a, b in zip(totals, (tp, fp, fn))]
        all_ious.extend(ious)
        records.append({"index": idx, "tp": tp, "fp": fp, "fn": fn,
                        "ground_truth": ground_truth, "predictions": boxes, "hits": hits})
    tp, fp, fn = totals
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    metrics = {"images": len(predictions), "tp": tp, "fp": fp, "fn": fn,
               "precision": precision, "recall": recall,
               "f1": 2 * precision * recall / max(precision + recall, 1e-12),
               "mean_matched_iou": float(np.mean(all_ious)) if all_ious else 0.0,
               "conf_thresh": conf_thresh, "nms_iou_thresh": iou_thresh,
               "matching_iou_thresh": 0.5}
    return metrics, records


def export_results(model, val_loader, test_loader, val_set, test_set, device, output,
                   conf_thresh=None, iou_thresh=0.5):
    val_predictions = predict_all(model, val_loader, device)
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6] if conf_thresh is None else [conf_thresh]
    calibration = [score_predictions(val_predictions, val_set, t, iou_thresh)[0]
                   for t in thresholds]
    chosen = max(calibration, key=lambda row: row["f1"])
    threshold = chosen["conf_thresh"]
    predictions = predict_all(model, test_loader, device)
    metrics, records = score_predictions(predictions, test_set, threshold, iou_thresh)
    payload = {"test": metrics, "validation_threshold_search": calibration}
    (output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output / "test_predictions.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8")

    # 同时展示最先出现的非空测试图和空白图，图像编号可追溯，不挑最好样本。
    selected = [r for r in records if r["ground_truth"]][:7]
    blank = next((r for r in records if not r["ground_truth"]), None)
    if blank is not None:
        selected.append(blank)
    samples = [(*test_set[r["index"]], r["predictions"]) for r in selected]
    titles = [f"Test #{r['index']} | TP {r['tp']} FP {r['fp']} FN {r['fn']}" for r in selected]
    visualize_predictions(samples, output / "predictions.png", titles)
    for i, (sample, title) in enumerate(zip(samples[:3], titles[:3]), start=1):
        visualize_predictions([sample], output / f"test_{i:02d}.png", [title])
    failure = next((r for r in records if r["ground_truth"] and (r["fp"] or r["fn"])), None)
    if failure is not None:
        sample = (*test_set[failure["index"]], failure["predictions"])
        visualize_predictions([sample], output / "failure_case.png",
                              [f"Test #{failure['index']} | TP {failure['tp']} FP {failure['fp']} FN {failure['fn']}"])
        (output / "failure_case.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
    print("TEST:", json.dumps(metrics), flush=True)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-samples", type=int, default=12000)
    parser.add_argument("--val-samples", type=int, default=1200)
    parser.add_argument("--test-samples", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--refresh-every", type=int, default=1,
                        help="每隔多少轮重新合成训练图；0 表示始终复用，验证和测试不变")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--data-root", type=Path, default=PROJECT / "data")
    parser.add_argument("--output", type=Path, default=PROJECT / "outputs")
    parser.add_argument("--conf-thresh", type=float, default=None,
                        help="省略时仅在验证集上选择 F1 最佳的阈值")
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--weights", type=Path, default=None)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.train_samples,
           args.val_samples, args.test_samples) < 1 or args.lr <= 0:
        parser.error("样本数、epochs、batch-size 和 lr 必须为正")
    if args.refresh_every < 0:
        parser.error("refresh-every 不能为负")
    return args


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
    print(f"Device: {device}", flush=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({"S": 7, "B": 2, "C": 10, "torch_version": torch.__version__,
                   "device_used": str(device),
                   "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                   "coordinate_format": "full-image normalized cx,cy,w,h",
                   "lambda_coord": 5.0, "lambda_noobj": 0.5,
                   "xy_weight": 1.0, "wh_weight": 2.0})

    def make_set(train, count, seed, split=None):
        return DigitDetectionDataset(str(args.data_root), train=train,
                                     seed=seed, num_samples=count, mnist_slice=split)

    stats = {}
    train_set = None
    train_source = None
    if not args.eval_only:
        train_source = make_set(True, args.train_samples, args.seed, (0, 50000))
        train_set, stats["train"] = cache_dataset(train_source, "train")
    val_set, stats["validation"] = cache_dataset(
        make_set(True, args.val_samples, 123, (50000, 60000)), "validation")
    test_set, stats["test"] = cache_dataset(
        make_set(False, args.test_samples, 321), "test")
    if not args.eval_only:
        (args.output / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    kwargs = dict(batch_size=args.batch_size, num_workers=0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, shuffle=False, **kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **kwargs)
    model = YOLOv1().to(device)
    config["parameters"] = sum(p.numel() for p in model.parameters())
    weights = args.weights or args.output / "yolov1_digit.pth"
    if args.eval_only:
        model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    else:
        (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        train_loader = DataLoader(train_set, shuffle=True, **kwargs)
        criterion = YOLOv1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)
        history = []
        best = float("inf")
        start = time.perf_counter()
        stats["training_epochs"] = []
        epoch_stats = stats["train"]
        for epoch in range(1, args.epochs + 1):
            tick = time.perf_counter()
            if epoch > 1 and args.refresh_every and (epoch - 1) % args.refresh_every == 0:
                train_source.seed = args.seed + (epoch - 1) * 10000019
                train_set, epoch_stats = cache_dataset(train_source, f"train epoch {epoch}")
                train_loader = DataLoader(train_set, shuffle=True, **kwargs)
            stats["training_epochs"].append(dict(epoch=epoch, seed=train_source.seed, **epoch_stats))
            loss, coord, conf, cls = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val = validation_loss(model, val_loader, criterion, device)
            row = {"epoch": epoch, "loss": loss, "coord": coord, "conf": conf,
                   "cls": cls, "val_loss": val, "lr": optimizer.param_groups[0]["lr"],
                   "seconds": time.perf_counter() - tick}
            history.append(row)
            if val < best:
                best = val
                config["best_epoch"] = epoch
                torch.save(model.state_dict(), weights)
            scheduler.step()
            with (args.output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=row.keys())
                writer.writeheader()
                writer.writerows(history)
            print(f"Epoch {epoch:02d}/{args.epochs} | loss={loss:.4f} coord={coord:.4f} "
                  f"conf={conf:.4f} cls={cls:.4f} val={val:.4f} ({row['seconds']:.1f}s)", flush=True)
        torch.save(model.state_dict(), args.output / "last_epoch.pth")
        config["training_seconds"] = time.perf_counter() - start
        config["best_val_loss"] = best
        (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        (args.output / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        plot_loss_curves(history, args.output / "loss_curves.png")
        model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
        visualize_predictions([(*train_set[i], []) for i in range(8)], args.output / "dataset_examples.png")
    plot_network(args.output / "network.png")
    export_results(model, val_loader, test_loader, val_set, test_set, device, args.output,
                   args.conf_thresh, args.iou_thresh)


if __name__ == "__main__":
    main()
