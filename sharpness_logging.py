"""Measure the sharpness during training, for the edge-of-stability problem.

The sharpness is lambda_max, the top eigenvalue of the Hessian H of the mean
next-token loss on a fixed set of probe sequences. For AdamW the logger also
measures the preconditioned sharpness lambda_max(D^-1/2 H D^-1/2), where
D = sqrt(v_hat) + eps comes from the optimizer state, so D^-1/2 H D^-1/2 is the
curvature that Adam's per-coordinate step eta / D actually sees.

Enable it by adding a logger to TrainConfig.metric_loggers:

    from metric_logging import AFTER_BACKWARD, MetricLogger
    from sharpness_logging import SharpnessLogger

    TrainConfig(
        ...,
        metric_loggers=TrainConfig.metric_loggers
        + (MetricLogger(AFTER_BACKWARD, SharpnessLogger(every=25)),),
    )

At a measured update t the parameters are those after t optimizer steps, and
eta is the learning rate that step t + 1 is about to use. Results go to wandb
under logging/sharpness/* (with wandb_online=True) and are printed as
SHARPNESS lines:

    lambda_max, eta_lambda           lambda_max(H) and eta * lambda_max(H)
    lambda_max_pre, eta_lambda_pre   the preconditioned versions (AdamW, from update 1)
    probe_loss                       the loss on the probe, in FP32
    residual                         ||H u - lambda u|| / lambda for the top vector u
    mass_<family>[_pre]              share of the top eigenvector's squared norm in
                                     each parameter family (embedding, attention,
                                     mlp, norm, head). A reading with nearly all its
                                     mass in the embedding comes from one rare
                                     token's row, whose Adam second moment is tiny.

Probes:
    probe="batch"    the batch about to be stepped on. In a full-batch run
                     (batch_size == num_train_sequences, one micro-batch) this
                     is the whole training set, so H is the Hessian of the
                     training objective.
    probe="heldout"  64 random training-source sequences that the run never
                     trains on: positions 600,256 to 600,319 of the data_seed
                     shuffle of the full 9.6M-sequence source (training uses the
                     first num_train_sequences <= 600,000 positions). Do not probe
                     with validation rows: the validation set is an unshuffled
                     prefix of a few documents, and its rare tokens make the
                     preconditioned reading jump.

Each measurement runs Lanczos iteration on Hessian-vector products (16 steps
the first time, then 10, warm-started from the previous top eigenvector), in
FP32 with the math attention kernel. On 64 sequences of a d8 model one
measurement takes about a minute on an RTX A6000.
"""

from contextlib import contextmanager
import math

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

HELDOUT_START, HELDOUT_SEQUENCES = 600_256, 64
FAMILIES = (("embed_tokens", "embedding"), ("lm_head", "head"), ("self_attn", "attention"),
            ("mlp", "mlp"), ("norm", "norm"))


def family(name):
    return next((label for key, label in FAMILIES if key in name), "other")


class SharpnessLogger:
    def __init__(self, every=25, steps=None, probe="batch", kinds=("raw", "pre"), chunk=4,
                 cold_steps=16, warm_steps=10):
        """Measure at every `every`-th update, or at the updates listed in `steps`.

        kinds: "raw" for lambda_max(H), "pre" for the preconditioned sharpness
        (AdamW only, from update 1). Each kind costs one eigensolve.

        chunk is the number of probe sequences per Hessian-vector-product pass;
        lower it if a measurement runs out of GPU memory.
        """
        if probe not in ("batch", "heldout"):
            raise ValueError(f"probe must be 'batch' or 'heldout', got {probe!r}.")
        self.every = every
        self.steps = None if steps is None else frozenset(int(s) for s in steps)
        self.probe = probe
        self.kinds = tuple(kinds)
        self.chunk = chunk
        self.cold_steps = cold_steps
        self.warm_steps = warm_steps
        self._reset()

    def _reset(self):
        self.vectors = {}
        self.heldout = None

    def setup(self, ctx):
        self._reset()
        if self.probe == "batch" and ctx.config.num_micro_batches != 1:
            raise ValueError("probe='batch' needs num_micro_batches=1 (it sees only one micro-batch).")

    def close(self):
        self._reset()

    def measured(self, step):
        return step in self.steps if self.steps is not None else step % self.every == 0

    def __call__(self, ctx):
        if not self.measured(ctx.step) or not torch.isfinite(ctx.loss).item():
            return {}
        device = next(ctx.model.parameters()).device
        tokens = ctx.input_ids if self.probe == "batch" else self._heldout(ctx.config, device)
        lr = ctx.optimizer.param_groups[0]["lr"]
        stats = {}
        with ProbeHessian(ctx.model, tokens, chunk=self.chunk) as op:
            stats["probe_loss"] = op.loss
            kinds = {"": op.hvp} if "raw" in self.kinds else {}
            scales = adam_scales(ctx.optimizer, op.params) if "pre" in self.kinds else None
            if scales is not None:
                kinds["_pre"] = lambda v: scales * op.hvp(scales * v)
            for suffix, matvec in kinds.items():
                start = self.vectors.get(suffix)
                steps = self.cold_steps if start is None else self.warm_steps
                value, vector, residual = top_eigenpair(matvec, op.size, device, steps, start)
                self.vectors[suffix] = vector
                stats[f"lambda_max{suffix}"] = value
                for label, share in op.mass(vector).items():
                    stats[f"mass_{label}{suffix}"] = share
                stats[f"eta_lambda{suffix}"] = lr * value
                if suffix == "":
                    stats["residual"] = residual
        print("SHARPNESS step=%d lr=%.6g train_loss=%.5f " % (ctx.step, lr, float(ctx.loss))
              + " ".join(f"{k}={v:.6g}" for k, v in stats.items()), flush=True)
        return {f"sharpness/{k}": v for k, v in stats.items()}

    def _heldout(self, config, device):
        if self.heldout is None:
            self.heldout = heldout_rows(config).to(device=device, dtype=torch.long)
        return self.heldout


def heldout_rows(config):
    """The 64 held-out probe sequences (see the module docstring)."""
    import numpy as np

    from data import global_shuffle_prefix_indices, load_token_dataset

    if config.num_train_sequences > HELDOUT_START:
        raise ValueError("The held-out probe overlaps the training rows.")
    source = load_token_dataset(config.train_dataset, "train", data_seed=None)
    seed = 42 if config.data_seed is None else config.data_seed
    order = global_shuffle_prefix_indices(source.total_sequences, HELDOUT_START + HELDOUT_SEQUENCES, seed)
    rows = np.stack([np.asarray(source.tokens[i]) for i in order[HELDOUT_START:]])
    return torch.from_numpy(rows.astype(np.int64))


@contextmanager
def fp32_evaluation(model, device):
    """Eval mode, FP32, math attention (it has a double backward), RNG left untouched."""
    modes = [(module, module.training) for module in model.modules()]
    devices = [device.index] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices), torch.autocast(device_type=device.type, enabled=False), \
            sdpa_kernel(SDPBackend.MATH), torch.enable_grad():
        try:
            model.eval()
            yield
        finally:
            for module, training in modes:
                module.training = training


class ProbeHessian:
    """v -> H v for the mean next-token loss over `probe` ([sequences, tokens]).

    The probe is processed `chunk` sequences at a time; each product rebuilds one
    chunk's double-backward graph, so memory is bounded by a single chunk. The
    model's .grad fields are never touched.
    """

    def __init__(self, model, probe, chunk=4):
        from train import causal_lm_loss

        self.loss_fn = causal_lm_loss
        self.model = model
        self.chunks = list(probe.split(chunk))
        self.weights = [len(c) / len(probe) for c in self.chunks]
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.families = [family(n) for n, _ in named]
        self.params = [p for _, p in named]
        if any(p.dtype != torch.float32 for p in self.params):
            raise ValueError("Sharpness needs FP32 weights (precision='mp').")
        self.size = sum(p.numel() for p in self.params)
        self.device = self.params[0].device
        self.loss = None
        self._context = None

    def __enter__(self):
        self._context = fp32_evaluation(self.model, self.device)
        self._context.__enter__()
        self.loss = 0.0
        for w, chunk in zip(self.weights, self.chunks):
            with torch.no_grad():
                self.loss += w * float(self.loss_fn(self.model(input_ids=chunk), chunk))
        return self

    def hvp(self, vector):
        out = torch.zeros_like(vector)
        for w, chunk in zip(self.weights, self.chunks):
            loss = self.loss_fn(self.model(input_ids=chunk), chunk)
            grads = torch.autograd.grad(loss, self.params, create_graph=True)
            flat = torch.cat([g.reshape(-1) for g in grads])
            products = torch.autograd.grad(flat, self.params, grad_outputs=vector, allow_unused=True)
            out.add_(torch.cat([(h if h is not None else torch.zeros_like(p)).reshape(-1)
                                for p, h in zip(self.params, products)]), alpha=w)
        return out.detach()

    def mass(self, vector):
        """Squared norm of a unit vector in each parameter family."""
        out, offset = {}, 0
        for label, p in zip(self.families, self.params):
            piece = vector[offset:offset + p.numel()]
            out[label] = out.get(label, 0.0) + float(piece.square().sum())
            offset += p.numel()
        return out

    def __exit__(self, *exc):
        context, self._context = self._context, None
        if context is not None:
            context.__exit__(*exc)
        return False


def adam_scales(optimizer, params):
    """D^-1/2 as a flat vector, D = sqrt(v / (1 - beta2^t)) + eps; None for SGD or before step 1."""
    groups = {id(p): g for g in optimizer.param_groups for p in g["params"]}
    scales = []
    for p in params:
        group, state = groups[id(p)], optimizer.state.get(p, {})
        if "exp_avg_sq" not in state or float(state.get("step", 0)) <= 0:
            return None
        beta2 = group["betas"][1]
        correction = -math.expm1(float(state["step"]) * math.log(beta2)) if beta2 else 1.0
        v = state["exp_avg_sq"].detach().float()
        scales.append((v / correction).sqrt().add(group["eps"]).rsqrt().reshape(-1))
    return torch.cat(scales)


def top_eigenpair(matvec, size, device, steps, start=None, seed=0):
    """Top eigenvalue of a symmetric operator by fully reorthogonalized Lanczos.

    Returns (value, unit vector, relative residual ||A u - value u|| / |value|).
    """
    if start is None:
        gen = torch.Generator(device=device).manual_seed(seed)
        q = torch.randint(0, 2, (size,), generator=gen, device=device).float().mul_(2).sub_(1)
    else:
        q = start.detach().clone().float()
    q /= q.norm()
    basis = torch.empty((steps, size), device=device, dtype=torch.float32)
    alphas, betas = [], []
    for i in range(steps):
        basis[i].copy_(q)
        z = matvec(q)
        if not torch.isfinite(z).all():
            raise FloatingPointError("Nonfinite Hessian-vector product.")
        alphas.append(float(torch.dot(q, z)))
        for _ in range(2):  # full reorthogonalization
            z -= basis[:i + 1].T @ (basis[:i + 1] @ z)
        beta = float(z.norm())
        if i + 1 == steps or beta < 1e-10 * max(1.0, abs(alphas[-1])):
            break
        betas.append(beta)
        q = z / beta
    n = len(alphas)
    T = torch.diag(torch.tensor(alphas, dtype=torch.float64))
    if n > 1:
        b = torch.tensor(betas[:n - 1], dtype=torch.float64)
        T += torch.diag(b, 1) + torch.diag(b, -1)
    values, rotation = torch.linalg.eigh(T)
    vector = (basis[:n].T @ rotation[:, -1].to(device=device, dtype=torch.float32))
    vector /= vector.norm()
    value = float(values[-1])
    residual = float((matvec(vector) - value * vector).norm()) / max(abs(value), 1e-30)
    return value, vector, residual
