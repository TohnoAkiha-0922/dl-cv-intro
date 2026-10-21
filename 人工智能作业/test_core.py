"""关键数学逻辑和数据一致性测试：python -m unittest -v test_core。"""

import math
import random
import unittest
from pathlib import Path

import torch

from create_dataset import DigitDetectionDataset
from loss import compute_iou, YOLOv1Loss
from model import YOLOv1
from nms import nms
from train import match_detections


class IoUTests(unittest.TestCase):
    def test_identical_disjoint_and_zero_area(self):
        box = [0.5, 0.5, 0.2, 0.2]
        self.assertAlmostEqual(float(compute_iou(box, box)), 1.0, places=5)
        self.assertLessEqual(float(compute_iou(box, box)), 1.0)
        self.assertEqual(float(compute_iou(box, [0.1, 0.1, 0.1, 0.1])), 0.0)
        self.assertEqual(float(compute_iou([0, 0, 0, 0], [0, 0, 0, 0])), 0.0)

    def test_partial_overlap_and_broadcast(self):
        result = compute_iou(torch.tensor([[0.5, 0.5, 0.4, 0.4],
                                          [0.7, 0.5, 0.4, 0.4]]),
                             torch.tensor([0.5, 0.5, 0.4, 0.4]))
        torch.testing.assert_close(result, torch.tensor([1.0, 1 / 3]))


class LossTests(unittest.TestCase):
    def test_empty_images_have_only_negative_confidence_loss(self):
        pred = torch.zeros(1, 1, 1, 20)
        pred[..., 4], pred[..., 9] = 0.8, 0.2
        pred.requires_grad_()
        total, coord, conf, cls = YOLOv1Loss(S=1)(pred, torch.zeros_like(pred))
        self.assertAlmostEqual(float(total.detach()), 0.34, places=6)
        self.assertEqual(float(coord.detach()), 0)
        self.assertEqual(float(cls.detach()), 0)
        total.backward()
        self.assertAlmostEqual(float(pred.grad[..., 4]), 0.8, places=6)
        self.assertAlmostEqual(float(pred.grad[..., 9]), 0.2, places=6)
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_only_best_iou_box_is_responsible_with_duplicated_targets(self):
        pred = torch.zeros(1, 1, 1, 20)
        pred[..., :5] = torch.tensor([0.1, 0.1, 0.2, 0.2, 0.4])
        pred[..., 5:10] = torch.tensor([0.5, 0.5, 0.2, 0.2, 0.8])
        target = torch.zeros_like(pred)
        target[..., :5] = torch.tensor([0.5, 0.5, 0.2, 0.2, 1.0])
        target[..., 5:10] = target[..., :5]
        target[..., 14] = 1
        pred.requires_grad_()
        total, coord, conf, cls = YOLOv1Loss(S=1)(pred, target)
        self.assertAlmostEqual(float(coord.detach()), 0.0, places=6)
        self.assertAlmostEqual(float(conf.detach()), 0.12, places=6)
        self.assertAlmostEqual(float(cls.detach()), math.log(10), places=6)
        torch.testing.assert_close(total, coord + conf + cls)
        total.backward()
        self.assertGreater(float(pred.grad[..., 4]), 0)
        self.assertLess(float(pred.grad[..., 9]), 0)
        self.assertTrue((pred.grad[..., :4] == 0).all())

    def test_incorrect_shape_rejected(self):
        with self.assertRaises(ValueError):
            YOLOv1Loss()(torch.zeros(1, 7, 7, 19), torch.zeros(1, 7, 7, 19))


class NMSTests(unittest.TestCase):
    @staticmethod
    def five_boxes():
        pred = torch.zeros(3, 3, 20)
        for gy, gx, slot, confidence, xy, cls_id in [
            (0, 0, 0, .95, .3, 1), (0, 0, 1, .90, .3, 1),
            (0, 1, 0, .85, .31, 1), (0, 2, 0, .80, .8, 1),
            (1, 0, 0, .70, .3, 2),
        ]:
            pred[gy, gx, 5 * slot:5 * slot + 5] = torch.tensor([xy, xy, .2, .2, confidence])
            pred[gy, gx, 10 + cls_id] = 3
        return pred

    def test_five_boxes_three_overlap_and_different_classes_survive(self):
        boxes = nms(self.five_boxes())
        self.assertEqual(len(boxes), 3)
        self.assertEqual([box[5] for box in boxes], [1, 1, 2])
        for actual, expected in zip([box[4] for box in boxes], [.95, .8, .7]):
            self.assertAlmostEqual(actual, expected, places=5)

    def test_high_iou_threshold_keeps_all(self):
        self.assertEqual(len(nms(self.five_boxes(), iou_thresh=1.0)), 5)

    def test_empty_and_low_confidence(self):
        self.assertEqual(nms(torch.zeros(7, 7, 20)), [])
        self.assertEqual(nms(self.five_boxes(), conf_thresh=.99), [])


class EvaluationTests(unittest.TestCase):
    def test_one_truth_cannot_match_two_predictions(self):
        gt = [(0.5, 0.5, 0.2, 0.2, 3)]
        pred = [(0.5, 0.5, 0.2, 0.2, .9, 3), (0.5, 0.5, 0.2, 0.2, .8, 3)]
        tp, fp, fn, _, hits = match_detections(pred, gt)
        self.assertEqual((tp, fp, fn), (1, 1, 0))
        self.assertEqual(hits, [True, False])

    def test_wrong_class_does_not_count_as_correct(self):
        tp, fp, fn, _, _ = match_detections([(0.5, 0.5, .2, .2, .9, 4)],
                                           [(0.5, 0.5, .2, .2, 3)])
        self.assertEqual((tp, fp, fn), (0, 1, 1))


class ModelAndDatasetTests(unittest.TestCase):
    def test_model_output_and_backward(self):
        torch.set_num_threads(2)
        model = YOLOv1()
        self.assertEqual(model.feat_dim, 8192)
        output = model(torch.rand(2, 1, 64, 64))
        self.assertEqual(tuple(output.shape), (2, 7, 7, 20))
        box = output[..., :10].reshape(2, 7, 7, 2, 5)
        self.assertTrue(((box >= 0) & (box <= 1)).all())
        grid_x = torch.arange(7).view(1, 1, 7, 1)
        self.assertTrue(((box[..., 0] * 7 >= grid_x)
                         & (box[..., 0] * 7 <= grid_x + 1)).all())
        loss = YOLOv1Loss()(output, torch.zeros_like(output))[0]
        loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                            for p in model.parameters()))

    @unittest.skipUnless((Path(__file__).parent / "data/MNIST/raw/train-images-idx3-ubyte").exists(),
                         "先下载 MNIST 后运行数据集集成检查")
    def test_dataset_repeatability_collision_handling_and_global_rng(self):
        random.seed(1234)
        state = random.getstate()
        dataset = DigitDetectionDataset(str(Path(__file__).parent / "data"),
                                        S=1, max_digits=5, seed=42, num_samples=24)
        self.assertEqual(random.getstate(), state)
        saw_collision = False
        saw_blank = False
        for index in range(len(dataset)):
            image, label = dataset[index]
            info = dataset.last_metadata
            saw_collision |= info["skipped_collisions"] > 0
            saw_blank |= info["objects"] == 0
            torch.testing.assert_close(image, dataset[index][0])
            torch.testing.assert_close(label, dataset[index][1])
            self.assertLessEqual(int(label[..., 4].sum()), 1)
            self.assertLessEqual(int(label[..., 10:].sum()), 1)
            self.assertEqual(tuple(image.shape), (1, 64, 64))
            self.assertTrue(0 <= float(image.min()) <= float(image.max()) <= 1)
        self.assertTrue(saw_collision)
        self.assertTrue(saw_blank)


if __name__ == "__main__":
    unittest.main()
