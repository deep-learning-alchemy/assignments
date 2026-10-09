"""Validation loss along the line between two checkpoints, and of their ensemble.

A checkpoint is named by (run_name, step): step N is the kept training
checkpoint `step_N.pt`, and step None is the run's final saved model.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import modal

from modal_utils import (
    MODAL_ENVIRONMENT,
    MODAL_MODEL_DIR,
    VOLUME_MOUNTS,
    build_image,
    timestamped_modal_app_name,
)

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
EVAL_BATCH_SIZE = 64


def checkpoint_path(model_dir, run_name, step=None):
    run_dir = Path(model_dir) / run_name
    return run_dir if step is None else run_dir / f"step_{step}.pt"


def _val_batches():
    from data import DEFAULT_DATA_SEED, dclm_val_dataset, load_token_dataset
    from utils import create_batches

    dataset = load_token_dataset(dclm_val_dataset(), role="val", data_seed=DEFAULT_DATA_SEED)
    return list(create_batches(dataset, EVAL_BATCH_SIZE))


def _ensemble_loss(model_a, model_b, batches, device, precision):
    """NLL of the 0.5/0.5 mixture of the two models' next-token probabilities."""
    import torch
    import torch.nn.functional as F
    from utils import autocast_context

    total_nll, total_tokens = 0.0, 0
    with torch.no_grad():
        for input_ids in batches:
            input_ids = input_ids.to(device)
            targets = input_ids[..., 1:].unsqueeze(-1)
            log_probs = []
            for model in (model_a, model_b):
                with autocast_context(precision, device=device):
                    logits = model(input_ids=input_ids)
                log_probs.append(
                    F.log_softmax(logits[..., :-1, :].float(), dim=-1).gather(-1, targets)
                )
            mixture = torch.logsumexp(torch.stack(log_probs), dim=0) - math.log(2.0)
            total_nll -= mixture.sum().item()
            total_tokens += targets.numel()
    return total_nll / total_tokens


def measure_pair(
    path_a, path_b, *, alphas=ALPHAS, ensemble=True, device="cuda", precision="mp", batches=None
):
    """Return validation losses of (1 - alpha) * theta_a + alpha * theta_b and the ensemble."""
    import torch
    from model_io import load_model
    from train import evaluate

    model = load_model(path_a, device=device).eval()
    other = load_model(path_b, device=device).eval()
    params = list(model.parameters())
    theta_a = [p.detach().clone() for p in params]
    theta_b = [p.detach().clone() for p in other.parameters()]
    if batches is None:
        batches = _val_batches()
    config = SimpleNamespace(precision=precision)

    row = dict(a=str(path_a), b=str(path_b), alphas=list(alphas), interpolation=[])
    with torch.no_grad():
        for alpha in alphas:
            torch._foreach_copy_(params, [torch.lerp(x, y, alpha) for x, y in zip(theta_a, theta_b)])
            row["interpolation"].append(evaluate(model, batches, config, device).item())
        torch._foreach_copy_(params, theta_a)
    if ensemble:
        row["ensemble"] = _ensemble_loss(model, other, batches, device, precision)
    return row


def measure_pairs(pairs, *, model_dir, **kwargs):
    """pairs: list of ((run_a, step_a), (run_b, step_b)); returns one row per pair."""
    rows = []
    for (run_a, step_a), (run_b, step_b) in pairs:
        row = measure_pair(
            checkpoint_path(model_dir, run_a, step_a),
            checkpoint_path(model_dir, run_b, step_b),
            **kwargs,
        )
        row.update(run_a=run_a, step_a=step_a, run_b=run_b, step_b=step_b)
        print(json.dumps(row), flush=True)
        rows.append(row)
    return rows


app = modal.App("dl-alchemy-a3-connectivity")


@app.function(image=build_image(), volumes=VOLUME_MOUNTS, gpu="H100", timeout=6 * 60 * 60)
def _measure_pairs_remote(pairs, kwargs):
    pairs = [tuple(tuple(ref) for ref in pair) for pair in pairs]
    return measure_pairs(pairs, model_dir=str(MODAL_MODEL_DIR), **kwargs)


def measure_pairs_on_modal(pairs, **kwargs):
    """Run measure_pairs on one Modal H100 against your Modal checkpoint volume."""
    with modal.enable_output():
        with app.run(
            name=timestamped_modal_app_name("dl-alchemy-a3-connectivity"),
            environment_name=MODAL_ENVIRONMENT,
        ):
            return _measure_pairs_remote.remote(pairs, kwargs)
