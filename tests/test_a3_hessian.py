"""CPU checks of the Problem 4 Hessian tools against an explicit Hessian of a tiny model.

Run with: python -B -m unittest discover -s tests -p 'test_a3_hessian.py'
"""

import unittest

import numpy as np
import torch
from torch.func import functional_call
from torch.nn.attention import SDPBackend, sdpa_kernel

from experiments.a3_optimization import hessian as hs
from experiments.a3_optimization.hessian_measure import ARMS, measure, scaled
from model_config import LMConfig
from modeling import AutoregressiveLM, initialize_model
from train import causal_lm_loss


def tiny_model(seed=0):
    torch.manual_seed(seed)
    config = LMConfig(name="tiny", vocab_size=11, context_length=6, hidden_size=8,
                      intermediate_size=12, num_hidden_layers=2, num_attention_heads=2,
                      num_key_value_heads=2, head_dim=4)
    model = AutoregressiveLM(config)
    initialize_model(model)
    with torch.no_grad():  # move away from the symmetric initialization
        for p in model.parameters():
            p.add_(0.3 * torch.randn_like(p))
    return model.eval()


def dense_hessian(model, input_ids):
    names, params = hs.trainable(model)
    shapes = [p.shape for p in params]
    flat = torch.cat([p.detach().reshape(-1) for p in params])

    def loss(vector):
        tensors, offset = {}, 0
        for name, shape in zip(names, shapes):
            tensors[name] = vector[offset:offset + shape.numel()].view(shape)
            offset += shape.numel()
        with sdpa_kernel(SDPBackend.MATH):
            return causal_lm_loss(functional_call(model, tensors, (input_ids,)), input_ids)

    return torch.autograd.functional.hessian(loss, flat)


class HessianToolsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = tiny_model()
        cls.inputs = torch.tensor([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1],
                                   [0, 0, 1, 1, 2, 2], [7, 8, 9, 10, 7, 8], [3, 1, 4, 1, 5, 9]])
        cls.H = dense_hessian(cls.model, cls.inputs)
        cls.size = cls.H.shape[0]
        cls.names, params = hs.trainable(cls.model)
        cls.sizes = [p.numel() for p in params]
        cls.matvec = staticmethod(lambda v: cls.H @ v)

    def test_probe_hessian_matches_dense(self):
        v = torch.randn(self.size)
        for retain in (False, True):
            with hs.ProbeHessian(self.model, self.inputs, chunk=2, retain=retain) as H:
                exact = self.H @ v
                torch.testing.assert_close(H.hvp(v), exact, rtol=1e-4, atol=1e-5 * float(exact.abs().max()))
                self.assertEqual(H.size, self.size)

    def test_extreme_eigenpairs(self):
        exact = torch.linalg.eigvalsh(self.H.double())
        values, residuals, vectors = hs.top_eigenpairs(self.matvec, self.size, "cpu", k=3,
                                                       steps=self.size)
        np.testing.assert_allclose(values, exact[-3:].flip(0).numpy(), rtol=1e-3, atol=1e-4)
        self.assertLess(max(residuals), 1e-3)
        low, _, _ = hs.bottom_eigenpairs(self.matvec, self.size, "cpu", steps=self.size)
        self.assertAlmostEqual(low[0], float(exact[0]), places=3)

    def test_quadrature_weights_and_density(self):
        run, _ = hs.lanczos(self.matvec, self.size, "cpu", steps=self.size, seed=3, k=0)
        self.assertAlmostEqual(sum(run["weights"]), 1.0, places=6)
        # sum_j w_j theta_j^2 = q^T H^2 q for the (normalized) start vector q
        gen = torch.Generator().manual_seed(3)
        q = hs.random_signs(self.size, gen, "cpu")
        q /= q.norm()
        moment = sum(w * t * t for w, t in zip(run["weights"], run["ritz"]))
        self.assertLess(abs(moment / float((self.H @ q).square().sum()) - 1), 1e-6)
        span = max(run["ritz"]) - min(run["ritz"])
        grid = np.linspace(min(run["ritz"]) - 0.1 * span, max(run["ritz"]) + 0.1 * span, 4001)
        mass = hs.density([run], grid, width=0.01 * span).sum() * (grid[1] - grid[0])
        self.assertAlmostEqual(mass, 1.0, places=3)

    def test_partitions_cover_every_coordinate(self):
        for label in (hs.family_of, hs.layer_of, lambda n: n):
            blocks = hs.partition(self.names, self.sizes, label)
            covered = sorted(r for ranges in blocks.values() for r in ranges)
            self.assertEqual(sum(hs.block_size(r) for r in blocks.values()), self.size)
            self.assertEqual(covered[0][0], 0)
            self.assertTrue(all(a[1] == b[0] for a, b in zip(covered, covered[1:])))
        families = set(hs.partition(self.names, self.sizes, hs.family_of))
        self.assertEqual(families, set(hs.FAMILIES))

    def test_restrict_is_the_principal_block(self):
        ranges = hs.partition(self.names, self.sizes, hs.family_of)["attention"]
        index = torch.cat([torch.arange(a, b) for a, b in ranges])
        v = torch.randn(len(index))
        torch.testing.assert_close(hs.restrict(self.matvec, self.size, ranges)(v),
                                   self.H[index][:, index] @ v, rtol=1e-5, atol=1e-6)

    def test_block_energy_estimates_frobenius_norms(self):
        blocks = hs.partition(self.names, self.sizes, hs.layer_of)
        keys, E = hs.block_energy(self.matvec, self.size, blocks, "cpu", probes=3000)
        index = {k: torch.cat([torch.arange(a, b) for a, b in blocks[k]]) for k in keys}
        exact = np.array([[float(self.H[index[c]][:, index[b]].square().sum()) for b in keys]
                          for c in keys])
        np.testing.assert_allclose(E.mean(0), exact, rtol=0.1, atol=1e-3 * exact.max())
        # a finer partition sums exactly to a coarser one
        fine = hs.partition(self.names, self.sizes, lambda n: n)
        names, F = hs.block_energy(self.matvec, self.size, fine, "cpu", probes=2, seed=5)
        groups, coarse = hs.coarsen(names, F, hs.layer_of)
        self.assertEqual(groups, keys)
        self.assertAlmostEqual(coarse.sum(), F.sum(), places=6)

    def test_measure_runs_on_a_tiny_model(self):
        out = measure(self.model, probe=self.inputs, top_k=2, families=True,
                      tensors=[self.names[0]], density_runs=1, energy=("family", "layer"),
                      lanczos_steps=40, block_steps=20, density_steps=20)
        exact = torch.linalg.eigvalsh(self.H.double())
        self.assertAlmostEqual(out["lambda_max"], float(exact[-1]), places=3)
        self.assertEqual(set(out["energy"]), {"family", "layer"})
        E = np.array(out["energy"]["family"]["per_probe"])
        self.assertEqual(E.shape, (4, 6, 6))

    def test_scaled_restores_weights(self):
        before = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        name = self.names[1]
        with scaled(self.model, {name: 3.0}):
            torch.testing.assert_close(dict(self.model.named_parameters())[name], 3.0 * before[name])
        for n, p in self.model.named_parameters():
            self.assertTrue(torch.equal(p, before[n]))

    def test_projection_arms(self):
        basis, _ = torch.linalg.qr(torch.randn(50, 4))
        basis = basis.T
        g = torch.randn(50)
        top, removed = ARMS["top"](g, basis), ARMS["removed"](g, basis)
        torch.testing.assert_close(top + removed, g)
        torch.testing.assert_close(ARMS["top"](top, basis), top)
        self.assertLess(float((basis @ removed).abs().max()), 1e-5)

    def test_continuation_schedule_matches_the_trainer(self):
        from experiments.a3_optimization.hessian_checkpoints import build_optimizer
        from experiments.a3_optimization.hessian_measure import _scheduler
        from lr_schedules import set_scheduler_to_completed_steps
        from train import TrainConfig

        config = TrainConfig(optimizer_name="sgd", learning_rate=1.0, weight_decay=0.0)
        optimizer = build_optimizer(tiny_model(), config)
        scheduler = _scheduler(config, optimizer, 9375, lr_scale=2.0)
        set_scheduler_to_completed_steps(scheduler, 1526)
        # linear decay after 93 warmup steps, peak 2 x 1.0
        self.assertAlmostEqual(scheduler.get_last_lr()[0], 2 * (1 - (1526 - 93) / (9375 - 93)))

    def test_launcher_jobs_are_well_formed(self):
        import inspect

        from experiments.a3_optimization import p4_inside_the_hessian as p4
        from experiments.a3_optimization.hessian_measure import continue_projected, measure

        jobs = [j for stage in p4.STAGES.values() for j in stage]
        self.assertEqual(len({j["name"] for j in jobs}), len(jobs))
        allowed = {"measure": set(inspect.signature(measure).parameters) | {"run", "step", "scale"},
                   "continue": set(inspect.signature(continue_projected).parameters)}
        for job in jobs:
            (kind,) = set(job) - {"name"}
            self.assertLessEqual(set(job[kind]), allowed[kind], job["name"])


if __name__ == "__main__":
    unittest.main()
