"""Optional PyTorch checks: data separation, conditioning and real optimization."""

from dataclasses import asdict
import importlib.util
import tempfile
from pathlib import Path
import unittest


HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "optional PyTorch is not installed")
class LearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def test_condition_plane_carries_known_parameter(self):
        import torch
        from defocuslab.model import conditioned_input
        observation = torch.randn(2, 1, 19, 23)
        inputs = conditioned_input(observation, [0.4, 1.2])
        self.assertEqual(tuple(inputs.shape), (2, 2, 19, 23))
        torch.testing.assert_close(inputs[:, :1], observation)
        torch.testing.assert_close(inputs[0, 1], torch.full((19, 23), 0.2))
        torch.testing.assert_close(inputs[1, 1], torch.full((19, 23), 0.6))
        with self.assertRaises(ValueError):
            conditioned_input(observation, [0, 0.4, 0.8])

    def test_forward_preserves_odd_image_shape_and_residual_initialization(self):
        import torch
        from defocuslab.model import ConditionalUNet, conditioned_input
        observation = torch.randn(2, 1, 33, 35)
        model = ConditionalUNet()
        output = model(conditioned_input(observation, [0.0, 0.8]))
        self.assertEqual(tuple(output.shape), tuple(observation.shape))
        torch.testing.assert_close(output, observation)

    def test_step_changes_parameters_and_checkpoint_round_trip(self):
        import torch
        from defocuslab.learning import load_checkpoint, save_checkpoint
        from defocuslab.model import ConditionalUNet, ModelConfig, conditioned_input
        from defocuslab.optics import OpticsConfig
        torch.manual_seed(9)
        model = ConditionalUNet(ModelConfig(base_channels=4))
        inputs = conditioned_input(torch.rand(2, 1, 24, 24), [0.4, 0.8])
        target = torch.zeros(2, 1, 24, 24)
        before = model.head.bias.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.mse_loss(model(inputs), target).backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.head.bias))
        model.eval()
        with torch.no_grad():
            expected = model(inputs)
        optics = OpticsConfig(size=24)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "best.pt"
            save_checkpoint(path, model, optics, {"noise_std": 0.005}, 1, 0.1, {"device": "cpu"})
            restored, restored_optics, payload = load_checkpoint(path)
            self.assertEqual(asdict(optics), asdict(restored_optics))
            self.assertEqual(payload["best_epoch"], 1)
            with torch.no_grad():
                torch.testing.assert_close(restored(inputs), expected, rtol=0, atol=0)

    def test_datasets_group_all_degradations_by_source_split(self):
        import numpy as np
        from defocuslab.learning import SyntheticDefocusDataset
        from defocuslab.optics import OpticsConfig
        optics = OpticsConfig(size=24)
        datasets = [SyntheticDefocusDataset(optics, split, 3, (0.0, 0.8))
                    for split in ("train", "val", "test")]
        seed_sets = [set(record["scene_seed"] for record in dataset.records) for dataset in datasets]
        for i in range(3):
            self.assertEqual(len(seed_sets[i]), 3)
            for j in range(i + 1, 3):
                self.assertTrue(seed_sets[i].isdisjoint(seed_sets[j]))
        observation1, target1, _ = datasets[0].arrays(0)
        observation2, target2, _ = datasets[0].arrays(1)
        np.testing.assert_array_equal(target1, target2)
        self.assertFalse(np.array_equal(observation1, observation2))
        np.testing.assert_array_equal(observation1, datasets[0].arrays(0)[0])
        self.assertFalse(np.array_equal(target1, datasets[1].arrays(0)[1]))


if __name__ == "__main__":
    unittest.main()
