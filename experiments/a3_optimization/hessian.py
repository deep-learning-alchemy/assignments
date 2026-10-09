"""Hessian tools for Problem 4, Inside the Hessian.

Everything acts on the flat vector of trainable parameters, in `model.named_parameters()`
order, and differentiates the mean next-token loss on a probe in FP32 with math attention
and TF32 off. Decoupled weight decay is not part of the loss, so it is not part of H.

* ProbeHessian      H v for the mean loss over the probe.
* lanczos           Ritz values, quadrature weights and top Ritz pairs with residuals.
* top_eigenpairs / bottom_eigenpairs
* density           eigenvalue density from Lanczos runs (stochastic Lanczos quadrature).
* family_of / layer_of / partition
                    named blocks of the parameter vector, as {block: [(start, stop), ...]}.
* restrict          the principal block H_bb of one block.
* block_energy      Hutchinson estimate of ||H_cb||_F^2 for every pair of blocks.
* hutchinson        tr(H)/d and tr(H^2)/d.

sharpness_logging.ProbeHessian (Problem 3) computes the same products during training;
this version also keeps the probe gradient, the parameter layout and (optionally) the
double-backward graphs, which the block and spectrum measurements need. Its six
families split the QK-norm gains from the other RMSNorm gains.
"""

from contextlib import ExitStack, contextmanager
import math
import re

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from train import causal_lm_loss


def exact_numerics():
    """FP32 matmuls everywhere: TF32 changes Hessian-vector products at the 1e-3 level."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def trainable(model):
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    return [name for name, _ in named], [p for _, p in named]


def flatten(tensors, params):
    return torch.cat([(t if t is not None else torch.zeros_like(p)).reshape(-1)
                      for p, t in zip(params, tensors)])


@contextmanager
def _evaluation(model, device):
    """Eval mode, FP32, math attention, and the caller's RNG state left untouched."""
    modes = [(module, module.training) for module in model.modules()]
    with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []), \
            torch.autocast(device_type=device.type, enabled=False), sdpa_kernel(SDPBackend.MATH):
        try:
            model.eval()
            yield
        finally:
            for module, training in modes:
                module.training = training


class _ChunkHessian:
    """Double-backward graph of the mean loss on one chunk of probe sequences."""

    def __init__(self, model, input_ids):
        self.model, self.input_ids = model, input_ids
        self.names, self.params = trainable(model)
        if any(p.dtype != torch.float32 for p in self.params):
            raise ValueError("Hessian measurements need FP32 weights.")

    def __enter__(self):
        self._stack = ExitStack()
        self._stack.enter_context(_evaluation(self.model, self.params[0].device))
        self._stack.enter_context(torch.enable_grad())
        loss = causal_lm_loss(self.model(input_ids=self.input_ids), self.input_ids)
        self.loss = float(loss.detach())
        grads = torch.autograd.grad(loss, self.params, create_graph=True, allow_unused=True)
        self.gradient = flatten(grads, self.params)
        return self

    def hvp(self, vector):
        with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):
            products = torch.autograd.grad(self.gradient, self.params, grad_outputs=vector,
                                           retain_graph=True, allow_unused=True)
        return flatten(products, self.params).detach()

    def __exit__(self, *exc):
        self._stack.close()
        self.gradient = None
        return False


class ProbeHessian:
    """Hessian of the mean loss over `probe` (a [sequences, tokens] LongTensor).

    Use as a context manager; the model must not change inside it. The probe is split
    into chunks of `chunk` sequences. With retain=True every chunk's double-backward graph
    stays alive (fast; for 64 sequences this needs more than 80 GB). With retain=False each
    product rebuilds one chunk's graph at a time (about 3x slower, memory of one chunk).

        with ProbeHessian(model, probe) as H:
            H.loss, H.gradient, H.size, H.names
            Hv = H.hvp(v)          # v: flat FP32 vector on the model's device
    """

    def __init__(self, model, probe, chunk=4, retain=None):
        self.model = model
        self.chunks = list(probe.split(chunk))
        self.weights = [len(c) / len(probe) for c in self.chunks]
        self.retain = len(probe) <= 16 if retain is None else retain
        self.names, self.params = trainable(model)
        self.sizes = [p.numel() for p in self.params]
        self.size = sum(self.sizes)
        self.device = self.params[0].device
        self.loss = self.gradient = None
        self._live = None

    def __enter__(self):
        self._stack = ExitStack()
        self.loss, self.gradient = 0.0, torch.zeros(self.size, device=self.device)
        if self.retain:
            self._live = [self._stack.enter_context(_ChunkHessian(self.model, c)) for c in self.chunks]
            for w, op in zip(self.weights, self._live):
                self.loss += w * op.loss
                self.gradient.add_(op.gradient.detach(), alpha=w)
        else:
            for w, c in zip(self.weights, self.chunks):
                with _ChunkHessian(self.model, c) as op:
                    self.loss += w * op.loss
                    self.gradient.add_(op.gradient.detach(), alpha=w)
        return self

    def hvp(self, vector):
        if vector.shape != (self.size,) or vector.dtype != torch.float32:
            raise ValueError("hvp expects a flat FP32 vector of length H.size.")
        out = torch.zeros_like(vector)
        if self.retain:
            for w, op in zip(self.weights, self._live):
                out.add_(op.hvp(vector), alpha=w)
        else:
            for w, c in zip(self.weights, self.chunks):
                with _ChunkHessian(self.model, c) as op:
                    out.add_(op.hvp(vector), alpha=w)
        if not torch.isfinite(out).all():
            raise FloatingPointError("Nonfinite Hessian-vector product.")
        return out

    def __exit__(self, *exc):
        self._stack.close()
        self._live = None
        return False


def random_signs(size, generator, device):
    return torch.randint(0, 2, (size,), generator=generator, device=device).float().mul_(2).sub_(1)


def lanczos(matvec, size, device, steps=48, seed=0, k=1):
    """Lanczos with full reorthogonalization from a random-sign start.

    Returns (result, vectors). result["ritz"] and result["weights"] are the Ritz values
    and their Gauss quadrature weights; for a random start q, sum_j w_j f(theta_j) is an
    unbiased estimate of tr f(H) / size. An unweighted histogram of Ritz values is not a
    density estimate: extreme Ritz values converge first. The top-k Ritz pairs are returned
    with residuals ||H u - theta u|| / |theta| (k extra products).
    """
    device = torch.device(device)
    q = random_signs(size, torch.Generator(device=device).manual_seed(seed), device)
    q /= q.norm()
    limit = min(steps, size)
    basis = torch.empty((limit, size), device=device)
    alphas, betas = [], []
    with torch.no_grad():
        for i in range(limit):
            basis[i].copy_(q)
            z = matvec(q)
            alphas.append(float(q @ z))
            z = z - alphas[-1] * q - (betas[-1] * basis[i - 1] if i else 0)
            for _ in range(2):
                z -= basis[:i + 1].T @ (basis[:i + 1] @ z)
            beta = float(z.norm())
            if i + 1 == limit or beta < 1e-10 * max(1.0, abs(alphas[-1])):
                break
            betas.append(beta)
            q = z / beta
        n = len(alphas)
        T = torch.diag(torch.tensor(alphas, dtype=torch.float64))
        if n > 1:
            b = torch.tensor(betas[:n - 1], dtype=torch.float64)
            T += torch.diag(b, 1) + torch.diag(b, -1)
        values, rotation = torch.linalg.eigh(T)
        top, residuals, vectors = [], [], []
        for j in range(1, min(k, n) + 1):
            theta = float(values[-j])
            u = basis[:n].T @ rotation[:, -j].to(device=device, dtype=torch.float32)
            u /= u.norm()
            residuals.append(float((matvec(u) - theta * u).norm()) / max(abs(theta), 1e-30))
            top.append(theta)
            vectors.append(u)
    return {"iterations": n, "ritz": values.tolist(), "weights": rotation[0].square().tolist(),
            "lambda_max": float(values[-1]), "lambda_min": float(values[0]),
            "top": top, "residuals": residuals}, vectors


def top_eigenpairs(matvec, size, device, k=1, steps=48, seed=0):
    """(values, residuals, vectors) of the k largest eigenvalues."""
    result, vectors = lanczos(matvec, size, device, steps=max(steps, 2 * k + 10), seed=seed, k=k)
    return result["top"], result["residuals"], vectors


def bottom_eigenpairs(matvec, size, device, k=1, steps=48, seed=0):
    """(values, residuals, vectors) of the k smallest eigenvalues (Lanczos on -H)."""
    values, residuals, vectors = top_eigenpairs(lambda v: -matvec(v), size, device, k, steps, seed)
    return [-v for v in values], residuals, vectors


def density(runs, grid, width):
    """Smoothed eigenvalue density from independent Lanczos runs.

    runs: list of lanczos() results; grid: points at which to evaluate; width: Gaussian
    kernel width in the same units as grid. Returns the density per unit eigenvalue,
    averaged over runs (it integrates to about 1). To see outliers over many orders of
    magnitude, apply the same monotone transform (e.g. a symmetric log) to grid and to
    the Ritz values before calling.
    """
    grid = np.asarray(grid, dtype=float)
    out = np.zeros_like(grid)
    for run in runs:
        for theta, w in zip(run["ritz"], run["weights"]):
            out += w * np.exp(-(grid - theta) ** 2 / (2 * width ** 2))
    return out / (len(runs) * width * math.sqrt(2 * math.pi))


def hutchinson(matvec, size, device, probes=16, seed=1):
    """tr(H)/d and tr(H^2)/d from E[z^T H z] = tr H and E||H z||^2 = tr H^2 (z random signs).
    Returns per-probe samples so standard errors can be formed."""
    gen = torch.Generator(device=device).manual_seed(seed)
    trace, square = [], []
    for _ in range(probes):
        z = random_signs(size, gen, device)
        hz = matvec(z)
        trace.append(float(z @ hz) / size)
        square.append(float(hz.square().sum()) / size)
    return {"trace_per_param": trace, "squared_per_param": square}


# ----- blocks -----------------------------------------------------------------------

FAMILIES = ("embedding", "attention", "qk_norm", "mlp", "norm", "head")


def family_of(name):
    """The six parameter families of the problem."""
    if "embed_tokens" in name:
        return "embedding"
    if "lm_head" in name:
        return "head"
    if "q_norm" in name or "k_norm" in name:
        return "qk_norm"
    if "self_attn" in name:
        return "attention"
    if "norm" in name:
        return "norm"
    return "mlp"


def layer_of(name):
    """'L0'..'L7' for block parameters; 'embedding', 'final_norm', 'head' otherwise."""
    m = re.search(r"layers\.(\d+)\.", name)
    if m:
        return f"L{int(m.group(1))}"
    return "embedding" if "embed_tokens" in name else "head" if "lm_head" in name else "final_norm"


def partition(names, sizes, label):
    """{block: [(start, stop), ...]} grouping parameter tensors by label(name).

    label may be family_of, layer_of, or any function of the tensor name (use
    `lambda name: name` for one block per tensor). Blocks keep first-appearance order.
    """
    blocks, offset = {}, 0
    for name, n in zip(names, sizes):
        blocks.setdefault(label(name), []).append((offset, offset + n))
        offset += n
    return blocks


def block_size(ranges):
    return sum(stop - start for start, stop in ranges)


def gather(vector, ranges):
    return torch.cat([vector[start:stop] for start, stop in ranges])


def scatter(values, ranges, size):
    full = values.new_zeros(size)
    offset = 0
    for start, stop in ranges:
        full[start:stop] = values[offset:offset + stop - start]
        offset += stop - start
    return full


def restrict(matvec, size, ranges):
    """v -> H_bb v for the principal block of the coordinates in `ranges`: the Hessian of
    the loss as a function of this block alone, everything else frozen. Its top eigenvalue
    is not the global top eigenvector restricted to the block."""
    return lambda v: gather(matvec(scatter(v, ranges, size)), ranges)


def block_energy(matvec, size, blocks, device, probes=4, seed=11):
    """Hutchinson estimate of E[k, c, b] = ||H_cb||_F^2 for every pair of blocks.

    For z with i.i.d. random signs on block b and zero elsewhere, E||(H z)_c||^2 =
    ||H_cb||_F^2. Returns (block names, array [probes, B, B]) with one estimate per probe;
    average over the first axis for the estimate, and use the spread for standard errors.
    Summing all cells gives tr(H^2) = ||H||_F^2. Estimates for a fine partition (e.g. one
    block per tensor) sum exactly to any coarser partition.
    """
    keys = list(blocks)
    out = np.zeros((probes, len(keys), len(keys)))
    gen = torch.Generator(device=device).manual_seed(seed)
    for j, b in enumerate(keys):
        for k in range(probes):
            hz = matvec(scatter(random_signs(block_size(blocks[b]), gen, device), blocks[b], size))
            hz2 = hz.square()
            for i, c in enumerate(keys):
                out[k, i, j] = float(sum(hz2[start:stop].sum(dtype=torch.float64)
                                         for start, stop in blocks[c]))
    return keys, out


def coarsen(keys, energy, label):
    """Sum a [..., B, B] block-energy array into the blocks given by label(key)."""
    groups = list(dict.fromkeys(label(k) for k in keys))
    index = np.array([groups.index(label(k)) for k in keys])
    out = np.zeros(energy.shape[:-2] + (len(groups), len(groups)))
    for i in range(len(keys)):
        for j in range(len(keys)):
            out[..., index[i], index[j]] += energy[..., i, j]
    return groups, out


def block_fractions(vector, blocks):
    """Share of ||vector||^2 in each block (e.g. where an eigenvector lives)."""
    total = float(vector.square().sum())
    return {b: float(gather(vector, r).square().sum()) / total for b, r in blocks.items()}
