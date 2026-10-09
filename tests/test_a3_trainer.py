"""CPU checks for the Assignment 3 TrainConfig fields."""
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from data import TokenDatasetConfig
from lr_schedules import build_scheduler
from model_config import LMConfig
from train import TrainConfig, build_model, checked_train_config, train, training_run_name


def mock_wandb():
    wandb = Mock()
    wandb.init.return_value = SimpleNamespace(id="test-run", url="test-run")
    wandb.util.generate_id.return_value = "test-run"
    return wandb


def lrs(lr_schedule, total_steps, warmup_steps, **kwargs):
    optimizer = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
    scheduler = build_scheduler(optimizer, lr_schedule, warmup_steps, total_steps, **kwargs)
    values = []
    for _ in range(total_steps):
        values.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return np.array(values)


class SchedulerTests(unittest.TestCase):
    def test_invsqrt_ignores_horizon(self):
        short, long = lrs("invsqrt", 50, 10), lrs("invsqrt", 200, 10)
        np.testing.assert_allclose(short, long[:50])
        self.assertAlmostEqual(long[10], 1.0)
        self.assertAlmostEqual(long[40], (10 / 40) ** 0.5)

    def test_sgdr_cycles_and_floor(self):
        values = lrs("sgdr20", 100, 10, min_lr_ratio=0.1)
        self.assertAlmostEqual(values[10], 1.0)
        self.assertAlmostEqual(values[30], 1.0)
        self.assertAlmostEqual(values[20], 0.55)
        self.assertGreaterEqual(values[10:].min(), 0.1)

    def test_dip_keeps_base_cooldown(self):
        base = lrs("wsd0.2", 100, 0)
        dipped = lrs("wsd0.2", 100, 0, lr_dip=(0.4, 0.5, 0.6, 0.1))
        np.testing.assert_allclose(dipped[:40], base[:40])
        np.testing.assert_allclose(dipped[60:], base[60:])
        self.assertAlmostEqual(dipped[50], 0.1)


class TrainerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        root = Path(self.folder.name)
        rng = np.random.default_rng(0)
        for split in ("train", "val"):
            path = root / split
            path.mkdir()
            rng.integers(0, 32, size=(40, 8), dtype=np.uint16).tofile(path / "tokens.bin")
            (path / "metadata.json").write_text(
                json.dumps(dict(dtype="uint16", num_sequences=40, seq_len=8))
            )
        self.config = TrainConfig(
            model_config=LMConfig("tiny", 32, 8, 16, 32, 1, 2, 2, head_dim=8),
            train_dataset=TokenDatasetConfig("train", "test", 8, path=str(root / "train")),
            val_dataset=TokenDatasetConfig("val", "test", 8, path=str(root / "val")),
            num_train_sequences=40, batch_size=4, precision="fp32", num_evals=2,
            data_seed=None, lr_schedule="wsd0.0", warmup_steps=3,
            wandb_online=False, metric_loggers=(), torch_compile_mode=None,
            model_dir=str(root / "models"),
        )

    def run_cpu(self, config):
        with patch("torch.cuda.device_count", return_value=0):
            return train(config)

    def step_state(self, config, step):
        path = Path(config.model_dir) / training_run_name(config) / f"step_{step}.pt"
        return torch.load(path, map_location="cpu", weights_only=False)

    def assert_same(self, model, state):
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)

    def test_fork_and_stop_reproduce_uninterrupted_run(self):
        prefix = replace(self.config, keep_checkpoint_steps=(4, 7))
        full = self.run_cpu(prefix)
        fork = dict(fork_from_run=training_run_name(prefix), fork_from_step=4)
        self.assert_same(self.run_cpu(replace(self.config, **fork)), full.state_dict())
        stopped = self.run_cpu(replace(self.config, stop_at_step=7, **fork))
        self.assert_same(stopped, self.step_state(prefix, 7)["model_state"])

    def test_fork_uses_branch_hyperparameters_and_order(self):
        prefix = replace(self.config, keep_checkpoint_steps=(4,))
        full = self.run_cpu(prefix).state_dict()
        fork = dict(fork_from_run=training_run_name(prefix), fork_from_step=4)
        for change in (dict(learning_rate=0.03), dict(weight_decay=0.0), dict(batch_order_seed=1)):
            branch = self.run_cpu(replace(self.config, **fork, **change)).state_dict()
            self.assertTrue(any(not torch.equal(branch[k], full[k]) for k in full), change)

    def test_ema_survives_fork_and_is_logged(self):
        config = replace(self.config, ema_decays=(0.5, 0.9), keep_checkpoint_steps=(5,),
                         wandb_online=True)
        with patch.dict(sys.modules, {"wandb": mock_wandb()}):
            import wandb
            self.run_cpu(config)
            full = [c.args[0] for c in wandb.log.call_args_list if "val_ema0.9_loss" in c.args[0]]
            self.assertIn("ema_state", self.step_state(config, 5))
            wandb.log.reset_mock()
            self.run_cpu(replace(config, keep_checkpoint_steps=(),
                                 fork_from_run=training_run_name(config), fork_from_step=5))
            forked = [c.args[0] for c in wandb.log.call_args_list if "val_ema0.9_loss" in c.args[0]]
        self.assertEqual(full[-1]["val_ema0.9_loss"], forked[-1]["val_ema0.9_loss"])
        self.assertNotEqual(full[-1]["val_ema0.9_loss"], full[-1]["val_loss"])

    def test_connectivity_endpoints_and_ensemble(self):
        from experiments.a3_optimization.connectivity import measure_pair
        from model_io import load_model
        from train import evaluate

        a = replace(self.config, save_model=True)
        b = replace(a, batch_order_seed=3)
        self.run_cpu(a)
        self.run_cpu(b)
        root = Path(a.model_dir)
        path_a, path_b = root / training_run_name(a), root / training_run_name(b)
        tokens = np.fromfile(Path(a.val_dataset.path) / "tokens.bin", dtype=np.uint16)
        batches = list(torch.as_tensor(tokens.astype(np.int64)).reshape(40, 8).split(8))
        kwargs = dict(device="cpu", precision="fp32", batches=batches)
        row = measure_pair(path_a, path_b, **kwargs)
        config = SimpleNamespace(precision="fp32")
        for path, loss in ((path_a, row["interpolation"][0]), (path_b, row["interpolation"][-1])):
            self.assertAlmostEqual(loss, evaluate(load_model(path), batches, config, "cpu").item(), places=5)
        same = measure_pair(path_a, path_a, **kwargs)
        self.assertAlmostEqual(same["ensemble"], same["interpolation"][0], places=5)
        self.assertLess(row["ensemble"], max(row["interpolation"][0], row["interpolation"][-1]))

    def test_resume_restores_ema_and_batch_order(self):
        config = replace(self.config, keep_checkpoint_steps=(5,),
                         ema_decays=(0.5, 0.9), batch_order_seed=3,
                         lr_schedule="linear", stop_at_step=8)
        full = self.run_cpu(config)
        latest = Path(config.model_dir) / training_run_name(config) / "latest.pt"
        expected = torch.load(latest, map_location="cpu", weights_only=False)
        # Simulate interruption after step 5 using the real kept checkpoint.
        torch.save(self.step_state(config, 5), latest)
        (latest.parent / "model.pt").unlink()
        resumed = self.run_cpu(config)
        self.assert_same(resumed, full.state_dict())
        actual = torch.load(latest, map_location="cpu", weights_only=False)
        self.assertEqual(actual["completed_steps"], 8)
        for decay, values in expected["ema_state"].items():
            for a, b in zip(actual["ema_state"][decay], values):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_sharpness_logger_runs_in_training_loop(self):
        from metric_logging import AFTER_BACKWARD, MetricLogger
        from sharpness_logging import SharpnessLogger

        logger = SharpnessLogger(every=1, cold_steps=8, warm_steps=4)
        config = replace(self.config, num_train_sequences=8, batch_size=8, num_epochs=4.0,
                         optimizer_name="sgd", beta1=0.0, weight_decay=0.0, grad_norm=None,
                         lr_schedule="step2x0.5", warmup_steps=None, embedding_init_scale=16.0,
                         metric_loggers=(MetricLogger(AFTER_BACKWARD, logger),), wandb_online=True)
        # Built before training: patch.dict drops modules first imported inside it.
        initial = build_model(replace(config, embedding_init_scale=1.0), device="cpu")
        with patch.dict(sys.modules, {"wandb": mock_wandb()}):
            import wandb
            model = self.run_cpu(config)
            rows = [c.args[0] for c in wandb.log.call_args_list
                    if "logging/sharpness/lambda_max" in c.args[0]]
        self.assertEqual(len(rows), 4)
        for row, lr in zip(rows, (0.003, 0.003, 0.0015, 0.0015)):
            self.assertAlmostEqual(row["logging/sharpness/eta_lambda"],
                                   lr * row["logging/sharpness/lambda_max"])
        self.assertGreater(model.get_input_embeddings().weight.std().item(),
                           8 * initial.get_input_embeddings().weight.std().item())

    def test_new_fields_change_run_names(self):
        c = self.config
        variants = [c, replace(c, warmup_steps=5), replace(c, lr_schedule="sgdr20", min_lr_ratio=0.1),
                    replace(c, lr_dip=(0.4, 0.5, 0.6, 0.1)), replace(c, ema_decays=(0.99,)),
                    replace(c, batch_order_seed=1), replace(c, stop_at_step=5),
                    replace(c, fork_from_run="a", fork_from_step=4),
                    replace(c, fork_from_run="b", fork_from_step=4)]
        self.assertEqual(len({training_run_name(v) for v in variants}), len(variants))

    def test_invalid_fields_are_rejected(self):
        for change in (dict(fork_from_run="a"), dict(fork_from_run="a", fork_from_step=0),
                       dict(min_lr_ratio=0.1), dict(lr_dip=(0.6, 0.5, 0.7, 0.1)),
                       dict(ema_decays=(1.0,)), dict(stop_at_step=3, keep_checkpoint_steps=(4,))):
            with self.assertRaises((ValueError, TypeError), msg=change):
                checked_train_config(replace(self.config, **change))


if __name__ == "__main__":
    unittest.main()
