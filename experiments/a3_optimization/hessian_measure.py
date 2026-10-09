"""Problem 4 measurements: one checkpoint's Hessian, and continuations with a projected update.

    from experiments.a3_optimization.hessian_measure import measure, continue_projected
    result = measure(model, extremes=True, families=True, energy=("family", "layer"))
    records = continue_projected("sgdm", 1526, "removed", k=10, steps=300)

measure: every option is independent, so you pay only for what you ask for. Cost is set
by the number of Hessian-vector products, about 2 seconds each on an H100/H200 with the
64-row probe: extremes ~150 products (~5 min), families 6 x 24 (~5 min), each density run
96 (~3 min), energy 4 per block: "family" (6 blocks) ~1 min, "layer" (11) ~1.5 min,
"tensor" (one block per parameter tensor, 91; it sums exactly to the coarser two)
~12 min. Every eigenvalue comes with its residual ||Hu - λu|| / |λ|; report it.

continue_projected (part (c)): every `refresh` steps the top-k eigenvectors of H on the
probe are recomputed with Lanczos and held fixed until the next refresh. At each step
the minibatch gradient g is passed through `transform(g, basis)` before gradient
clipping and the optimizer step. Built-in transforms (ARMS):

    "full"     g                      (the ordinary update)
    "top"      U U^T g                (only the top-k component)
    "removed"  g - U U^T g            (the top-k component removed)

where the rows of `basis` (U^T, shape [k, d]) are orthonormal eigenvectors. Pass your own
function to try something else. Every arm sees the same batches, in the run's own
training order, and the run's own learning-rate schedule (optionally scaled).

Records, one dict per step and per refresh, are returned and, if `log` is given, appended
to that JSONL file as they are produced:
    {"kind": "refresh", "step", "lr", "lambda_max", "top", "residuals",
     "heavy_ball_threshold", "basis_overlap", "probe_loss", "probe_gradient_fraction"}
    {"kind": "step", "step", "lr", "loss", "gradient_norm", "top_fraction",
     "clipped_norm", ["val_loss"]}
"""

import json
import time
from contextlib import contextmanager

import torch

import train as train_module
from experiments.a3_optimization import hessian as hs
from experiments.a3_optimization import hessian_checkpoints as checkpoints
from lr_schedules import build_scheduler, set_scheduler_to_completed_steps
from train import PermutedBatches, causal_lm_loss
from utils import create_batches


def measure(model, *, probe=None, extremes=True, top_k=5, families=True, tensors=(),
            density_runs=0, energy=(), energy_probes=4, hutchinson_probes=0, lanczos_steps=48, bottom_steps=96,
            block_steps=24, density_steps=96):
    """Measure the Hessian of the probe loss at the model's current weights.

    extremes       top-k and bottom eigenpairs of H, each with its residual and the share
                   of the eigenvector's squared norm in each parameter family.
    families       top eigenvalue of each family's principal block H_bb.
    tensors        parameter names whose own blocks' top eigenvalues to report, along
                   with each tensor's RMS and gradient norm.
    density_runs   number of independent Lanczos runs (Ritz values and quadrature weights)
                   for hessian.density.
    energy         partitions among "family", "layer", "tensor": for each, Hutchinson
                   estimates of ||H_cb||_F^2 for every pair of its blocks, per probe.
    hutchinson_probes  tr(H)/d and tr(H^2)/d samples.
    """
    hs.exact_numerics()
    device = next(model.parameters()).device
    probe = checkpoints.probe(device) if probe is None else probe
    t0 = time.time()
    out = {"probe_rows": len(probe)}
    with hs.ProbeHessian(model, probe) as H:
        out.update(size=H.size, probe_loss=H.loss, gradient_norm=float(H.gradient.norm()),
                   names=H.names, sizes=H.sizes)
        fam = hs.partition(H.names, H.sizes, hs.family_of)
        if extremes:
            # the bottom end converges more slowly (it sits closer to the bulk)
            for label, fn, k, n in (("top", hs.top_eigenpairs, top_k, lanczos_steps),
                                    ("bottom", hs.bottom_eigenpairs, 1, bottom_steps)):
                values, residuals, vectors = fn(H.hvp, H.size, device, k=k, steps=n)
                out[label] = {"values": values, "residuals": residuals,
                              "family_fractions": [hs.block_fractions(u, fam) for u in vectors],
                              "gradient_overlap": [float(u @ H.gradient) for u in vectors]}
            out["lambda_max"], out["lambda_min"] = out["top"]["values"][0], out["bottom"]["values"][0]
        if families:
            out["families"] = {}
            for name, ranges in fam.items():
                values, residuals, _ = hs.top_eigenpairs(
                    hs.restrict(H.hvp, H.size, ranges), hs.block_size(ranges), device, steps=block_steps)
                out["families"][name] = {"lambda_max": values[0], "residual": residuals[0],
                                         "size": hs.block_size(ranges)}
        if tensors:
            by_tensor = hs.partition(H.names, H.sizes, lambda n: n)
            params = dict(zip(H.names, H.params))
            out["tensors"] = {}
            for name in tensors:
                ranges = by_tensor[name]
                values, residuals, _ = hs.top_eigenpairs(
                    hs.restrict(H.hvp, H.size, ranges), hs.block_size(ranges), device, steps=block_steps)
                out["tensors"][name] = {"lambda_max": values[0], "residual": residuals[0],
                                        "rms": float(params[name].detach().square().mean().sqrt()),
                                        "gradient_norm": float(hs.gather(H.gradient, ranges).norm())}
        if density_runs:
            out["density_runs"] = []
            for seed in range(density_runs):
                run, _ = hs.lanczos(H.hvp, H.size, device, steps=density_steps, seed=100 + seed, k=0)
                out["density_runs"].append({k: run[k] for k in ("iterations", "ritz", "weights")})
        if isinstance(energy, str):
            energy = (energy,)
        if energy:
            out["energy"] = {}
            for part in energy:
                label = {"family": hs.family_of, "layer": hs.layer_of, "tensor": lambda n: n}[part]
                keys, E = hs.block_energy(H.hvp, H.size, hs.partition(H.names, H.sizes, label),
                                          device, probes=energy_probes)
                out["energy"][part] = {"blocks": keys, "per_probe": E.tolist()}
        if hutchinson_probes:
            out["hutchinson"] = hs.hutchinson(H.hvp, H.size, device, probes=hutchinson_probes)
    out["seconds"] = time.time() - t0
    return out


def measure_checkpoint(run, step=checkpoints.FINAL_STEP, device="cuda", scale=None, **options):
    """measure() on `step` of `run` (see checkpoints.load), tagged with run, step, tokens.
    scale: optional {parameter name: factor} applied to the weights before measuring."""
    model, _, _ = checkpoints.load(run, step, device)
    with scaled(model, scale or {}):
        result = measure(model, **options)
    return {"run": str(run), "step": step, "tokens": step * checkpoints.TOKENS_PER_STEP,
            "scale": scale or {}, **result}


@contextmanager
def scaled(model, factors):
    """Temporarily multiply named parameters by constants, e.g.
    `with scaled(model, {"model.norm.weight": 0.25, ...}): measure(model)`.
    The weights are restored exactly on exit (they are copied, not divided back)."""
    params = dict(model.named_parameters())
    saved = {name: params[name].detach().clone() for name in factors}
    try:
        with torch.no_grad():
            for name, c in factors.items():
                params[name].mul_(c)
        yield model
    finally:
        with torch.no_grad():
            for name, value in saved.items():
                params[name].copy_(value)


ARMS = {
    "full": lambda g, basis: g,
    "top": lambda g, basis: basis.T @ (basis @ g),
    "removed": lambda g, basis: g - basis.T @ (basis @ g),
}


def continue_projected(run, step, transform="full", *, k=10, steps=300, refresh=25,
                       lr_scale=1.0, reset_momentum=True, eval_every=25, lanczos_steps=48,
                       log=None, device="cuda"):
    hs.exact_numerics()
    transform = ARMS[transform] if isinstance(transform, str) else transform
    model, optimizer, config = checkpoints.load(run, step, device, optimizer=True)
    names, params = hs.trainable(model)
    batches = _training_batches(config)
    scheduler = _scheduler(config, optimizer, len(batches), lr_scale)
    set_scheduler_to_completed_steps(scheduler, step)
    beta = config.beta1 if config.optimizer_name == "sgd" else None
    if reset_momentum:
        for state in optimizer.state.values():
            state.pop("momentum_buffer", None)

    val, probe = checkpoints.validation_rows(config, device), checkpoints.probe(device)
    records = []
    sink = open(log, "a") if log else None

    def record(**fields):
        records.append(fields)
        if sink:
            sink.write(json.dumps(fields) + "\n")
            sink.flush()

    def val_loss():
        with torch.no_grad(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            model.eval()
            losses = [float(causal_lm_loss(model(input_ids=c), c)) for c in val.split(16)]
        return sum(losses) / len(losses)

    record(kind="start", run=str(run), step=step, k=k, refresh=refresh, lr_scale=lr_scale,
           reset_momentum=reset_momentum, val_loss=val_loss())
    basis, t0 = None, time.time()
    for i in range(steps):
        lr = scheduler.get_last_lr()[0]
        if i % refresh == 0:
            model.eval()
            with hs.ProbeHessian(model, probe) as H:
                values, residuals, vectors = hs.top_eigenpairs(H.hvp, H.size, device, k=k,
                                                               steps=lanczos_steps, seed=i)
                new = torch.stack(vectors)
                coeff = new @ H.gradient
                record(kind="refresh", step=step + i, lr=lr, lambda_max=values[0], top=values,
                       residuals=residuals, probe_loss=H.loss,
                       heavy_ball_threshold=None if beta is None else 2 * (1 + beta) / lr,
                       basis_overlap=None if basis is None else float((new @ basis.T).square().sum() / k),
                       probe_gradient_fraction=float(coeff.square().sum() / H.gradient.square().sum()))
            basis = new
        model.train()
        batch = batches[(step + i) % len(batches)].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = causal_lm_loss(model(input_ids=batch), batch)
        loss.backward()
        g = hs.flatten([p.grad for p in params], params)
        coeff = basis @ g
        new_g = transform(g, basis)
        offset = 0
        for p in params:
            p.grad.copy_(new_g[offset:offset + p.numel()].view_as(p))
            offset += p.numel()
        clipped = (float(torch.nn.utils.clip_grad_norm_(params, config.grad_norm))
                   if config.grad_norm is not None else None)
        optimizer.step()
        scheduler.step()
        fields = dict(kind="step", step=step + i + 1, lr=lr, loss=float(loss),
                      gradient_norm=float(g.norm()), clipped_norm=clipped,
                      top_fraction=float(coeff.square().sum() / g.square().sum().clamp_min(1e-30)))
        if (i + 1) % eval_every == 0 or i + 1 == steps:
            fields.update(val_loss=val_loss(), seconds=time.time() - t0)
            print(json.dumps({k: fields[k] for k in ("step", "loss", "val_loss", "top_fraction")}), flush=True)
        record(**fields)
    if sink:
        sink.close()
    return records


def _training_batches(config):
    """The run's training batches in its training order (as in train.train), one full
    batch per step."""
    data, _ = train_module.load_train_and_val_datasets(config)
    data = train_module.select_train_sequences(data, config)
    if config.data_seed is not None:
        data = data.shuffle(seed=config.data_seed)
    batches = create_batches(data, config.batch_size)
    if config.batch_order_seed is not None:
        if config.num_micro_batches != 1:
            raise ValueError("Continuations of runs with micro-batches and batch_order_seed "
                             "are not supported.")
        batches = PermutedBatches(batches, config.batch_order_seed)
    return batches


def _scheduler(config, optimizer, steps_per_epoch, lr_scale):
    """The run's learning-rate schedule (as in train.train), with the peak scaled."""
    total = int(config.num_epochs * steps_per_epoch)
    warmup = (config.warmup_steps if config.warmup_steps is not None
              else int(total * config.warmup_percent))
    scheduler = build_scheduler(
        optimizer, lr_schedule=config.lr_schedule, warmup_steps=warmup, total_steps=total,
        **({"min_lr_ratio": config.min_lr_ratio} if config.min_lr_ratio else {}),
        **({"lr_dip": tuple(config.lr_dip)} if config.lr_dip is not None else {}))
    for s in (scheduler, getattr(scheduler, "base_scheduler", None)):
        if s is not None:
            s.base_lrs = [lr * lr_scale for lr in s.base_lrs]
    return scheduler
