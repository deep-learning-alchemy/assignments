"""CPU checks of the student sharpness logger against an explicit Hessian.

Run with: python -B -m unittest discover -s tests -p 'test_a3_sharpness.py'
"""

import os
import pickle
import unittest
from types import SimpleNamespace

import torch

from unittest.mock import patch

from torch.func import functional_call
from torch.nn.attention import SDPBackend, sdpa_kernel

from lr_schedules import build_scheduler
from metric_logging import AFTER_BACKWARD, MetricLogger, importable_metric_loggers
from optimizers import build_optimizer
from sharpness_logging import ProbeHessian, SharpnessLogger, adam_scales, top_eigenpair
from model_config import LMConfig
from modeling import AutoregressiveLM, initialize_model
from train import TrainConfig, build_model, causal_lm_loss, training_run_name

INPUTS = torch.tensor([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1], [0, 0, 1, 1, 2, 2],
                       [7, 8, 9, 10, 7, 8], [3, 1, 4, 1, 5, 9]])


def tiny_model(seed=0):
    torch.manual_seed(seed)
    config = LMConfig(name="tiny", vocab_size=11, context_length=6, hidden_size=8,
                      intermediate_size=12, num_hidden_layers=1, num_attention_heads=2,
                      num_key_value_heads=2, head_dim=4)
    model = AutoregressiveLM(config)
    initialize_model(model)
    with torch.no_grad():  # move away from the symmetric initialization
        for p in model.parameters():
            p.add_(0.3 * torch.randn_like(p))
    return model.eval()


def dense_hessian(model, input_ids):
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    flat = torch.cat([p.detach().reshape(-1) for _, p in named])

    def loss(vector):
        tensors, offset = {}, 0
        for name, p in named:
            tensors[name] = vector[offset:offset + p.numel()].view(p.shape)
            offset += p.numel()
        with sdpa_kernel(SDPBackend.MATH):
            return causal_lm_loss(functional_call(model, tensors, (input_ids,)), input_ids)

    return torch.autograd.functional.hessian(loss, flat)


class SharpnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = tiny_model()
        cls.H = dense_hessian(cls.model, INPUTS)
        cls.top = float(torch.linalg.eigvalsh(cls.H)[-1])

    def test_hvp_matches_dense(self):
        v = torch.randn(self.H.shape[0])
        with ProbeHessian(self.model, INPUTS, chunk=2) as op:
            torch.testing.assert_close(op.hvp(v), self.H @ v, rtol=1e-4, atol=1e-5)

    def test_top_eigenvalue_and_warm_start(self):
        with ProbeHessian(self.model, INPUTS, chunk=3) as op:
            value, vector, residual = top_eigenpair(op.hvp, op.size, op.device, steps=60)
            self.assertAlmostEqual(value, self.top, delta=1e-4 * abs(self.top))
            self.assertLess(residual, 1e-3)
            again, _, _ = top_eigenpair(op.hvp, op.size, op.device, steps=3, start=vector)
            self.assertAlmostEqual(again, self.top, delta=1e-3 * abs(self.top))

    def test_preconditioned_matches_dense(self):
        model = tiny_model()
        optimizer = build_optimizer(model, optimizer_name="adamw", learning_rate=1e-3,
                                    weight_decay=0.0, beta1=0.9, beta2=0.95)
        causal_lm_loss(model(input_ids=INPUTS), INPUTS).backward()
        optimizer.step()
        optimizer.zero_grad()
        H = dense_hessian(model, INPUTS)
        with ProbeHessian(model, INPUTS) as op:
            s = adam_scales(optimizer, op.params)
            dense = float(torch.linalg.eigvalsh(s[:, None] * H * s[None, :])[-1])
            value, _, _ = top_eigenpair(lambda v: s * op.hvp(s * v), op.size, op.device, steps=80)
        self.assertAlmostEqual(value, dense, delta=1e-3 * abs(dense))
        sgd = build_optimizer(model, optimizer_name="sgd", learning_rate=0.1, weight_decay=0.0,
                              beta1=0.0, beta2=0.95)
        self.assertIsNone(adam_scales(sgd, list(model.parameters())))

    def test_logger_records_and_leaves_model_alone(self):
        model = tiny_model()
        model.train()
        optimizer = build_optimizer(model, optimizer_name="sgd", learning_rate=0.1, weight_decay=0.0,
                                    beta1=0.0, beta2=0.95)
        before = [p.detach().clone() for p in model.parameters()]
        logger = SharpnessLogger(every=5, cold_steps=60)
        ctx = SimpleNamespace(config=SimpleNamespace(num_micro_batches=1), step=10, model=model,
                              optimizer=optimizer, input_ids=INPUTS, loss=torch.tensor(1.0), val_batches={})
        logger.setup(ctx)
        stats = logger(ctx)
        self.assertAlmostEqual(stats["sharpness/lambda_max"], float(torch.linalg.eigvalsh(dense_hessian(model, INPUTS))[-1]), delta=1e-3 * abs(self.top))
        self.assertAlmostEqual(stats["sharpness/eta_lambda"], 0.1 * stats["sharpness/lambda_max"])
        self.assertNotIn("sharpness/lambda_max_pre", stats)
        masses = [v for k, v in stats.items() if k.startswith("sharpness/mass_")]
        self.assertAlmostEqual(sum(masses), 1.0, places=4)
        self.assertIn("sharpness/mass_embedding", stats)
        self.assertTrue(model.training)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        for p, q in zip(model.parameters(), before):
            torch.testing.assert_close(p, q)
        self.assertEqual(logger(SimpleNamespace(**{**ctx.__dict__, "step": 11})), {})

    def test_heldout_rows_are_the_reference_probe(self):
        # The probe behind the reference answers: positions 600,256 + [0, 64) of
        # the seed-42 permutation of the 9.6M-row source.
        import numpy as np
        from utils import DATA_DIR
        path = os.path.join(DATA_DIR, "dclm_9p6m_ctx1024", "train", "tokens.bin")
        if not os.path.exists(path):
            self.skipTest("needs the 9.6M-row source")
        source = np.memmap(path, dtype=np.uint16, mode="r").reshape(-1, 1024)
        order = np.random.default_rng(42).permutation(len(source))
        expected = torch.from_numpy(np.stack([source[i] for i in order[600_256:600_320]]).astype("int64"))
        from sharpness_logging import heldout_rows
        rows = heldout_rows(TrainConfig())
        self.assertEqual(rows.shape, (64, 1024))
        torch.testing.assert_close(rows, expected)
        with self.assertRaises(ValueError):
            heldout_rows(TrainConfig(num_train_sequences=700_000))

    def test_logger_instance_survives_modal_transport(self):
        loggers = importable_metric_loggers((MetricLogger(AFTER_BACKWARD, SharpnessLogger(every=7, probe="heldout")),))
        restored = pickle.loads(pickle.dumps(loggers))[0].fn
        self.assertIsInstance(restored, SharpnessLogger)
        self.assertEqual((restored.every, restored.probe), (7, "heldout"))
        # The default A1 logger still travels as an import path.
        default = importable_metric_loggers(TrainConfig.metric_loggers)
        self.assertEqual(default[0].fn, "module_rms_logging:log_module_rms")

    def test_kinds(self):
        model = tiny_model()
        optimizer = build_optimizer(model, optimizer_name="adamw", learning_rate=1e-3,
                                    weight_decay=0.0, beta1=0.9, beta2=0.95)
        causal_lm_loss(model(input_ids=INPUTS), INPUTS).backward()
        optimizer.step()
        optimizer.zero_grad()
        ctx = SimpleNamespace(config=SimpleNamespace(num_micro_batches=1), step=0, model=model,
                              optimizer=optimizer, input_ids=INPUTS, loss=torch.tensor(1.0), val_batches={})
        stats = SharpnessLogger(kinds=("pre",))(ctx)
        self.assertIn("sharpness/eta_lambda_pre", stats)
        self.assertNotIn("sharpness/eta_lambda", stats)


class ConfigTests(unittest.TestCase):
    def test_step_schedule(self):
        p = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([p], lr=0.01)
        scheduler = build_scheduler(optimizer, "step3x0.5", warmup_steps=0, total_steps=10)
        lrs = []
        for _ in range(6):
            lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
        self.assertEqual(lrs, [0.01, 0.01, 0.01, 0.005, 0.005, 0.005])
        with self.assertRaises(ValueError):
            build_scheduler(optimizer, "step3", warmup_steps=0, total_steps=10)

    def test_embedding_init_scale(self):
        base = build_model(TrainConfig(model_name="d4"), device="cpu")
        scaled = build_model(TrainConfig(model_name="d4", embedding_init_scale=16.0), device="cpu")
        torch.testing.assert_close(scaled.get_input_embeddings().weight, 16 * base.get_input_embeddings().weight)
        torch.testing.assert_close(scaled.get_output_embeddings().weight, base.get_output_embeddings().weight)
        self.assertNotEqual(training_run_name(TrainConfig()), training_run_name(TrainConfig(embedding_init_scale=16.0)))
        self.assertIn("-emb16", training_run_name(TrainConfig(embedding_init_scale=16.0)))


class StarterTests(unittest.TestCase):
    def test_measurement_steps_match_tokens(self):
        from experiments.a3_optimization.p3_edge_of_stability import measurement_steps
        base = measurement_steps(64)
        self.assertEqual(base[:3], [0, 16, 32])
        self.assertEqual(base[-1], 2700)
        self.assertEqual([s // 4 for s in measurement_steps(16)[:3]], base[:3])

    def test_starter_runs_are_valid(self):
        from experiments.a3_optimization.p3_edge_of_stability import RUNS, full_batch, minibatch
        from train import checked_train_config
        for config in RUNS + [full_batch(64, 500, every=40, optimizer_name="adamw",
                                         embedding_init_scale=1.0, learning_rate=1e-5),
                              full_batch(32, 1500, learning_rate=0.01, lr_schedule="step750x0.5"),
                              minibatch(batch_size=256), minibatch(lr_schedule="linear")]:
            checked_train_config(config)
        self.assertEqual(full_batch(32, 10).beta1, 0.0)
        self.assertEqual(full_batch(32, 10, optimizer_name="adamw").beta1, TrainConfig.beta1)

    def test_history_reads_preconditioned_only_runs(self):
        from experiments.a3_optimization import sharpness
        rows = [{"optimizer_step": 0, "train_loss": 7.0},
                {"optimizer_step": 1, "train_loss": 6.0, "learning_rate": 0.003,
                 "logging/sharpness/lambda_max_pre": 1e4, "logging/sharpness/eta_lambda_pre": 30.0},
                {"optimizer_step": 2, "train_loss": 5.0}]
        run = SimpleNamespace(created_at="2026", scan_history=lambda: rows)
        with patch("wandb.Api") as api:
            api.return_value.runs.return_value = [run]
            history = sharpness.history(TrainConfig())
        self.assertEqual(len(history), 1)
        self.assertEqual((history[0]["step"], history[0]["train_loss"], history[0]["eta_lambda_pre"]), (1, 6.0, 30.0))
        self.assertEqual(history.losses, [(0, 7.0), (1, 6.0), (2, 5.0)])


if __name__ == "__main__":
    unittest.main()
