"""Fetch logged curves from W&B for plotting.

    from experiments.a3_optimization.wandb_runs import load_runs
    runs = load_runs("a3-river-valley", keys=("val_loss", "val_ema0.99_loss", "learning_rate"))
    for run in runs:
        steps, losses = run["history"]["val_loss"]
"""

import numpy as np
import wandb

from utils import WANDB_ENTITY, WANDB_PROJECT

DEFAULT_KEYS = ("train_loss", "val_loss", "learning_rate")


def load_runs(tag, keys=DEFAULT_KEYS, *, finished_only=False):
    """Latest W&B run per run name with this tag; history maps key -> (steps, values)."""
    api = wandb.Api()
    latest = {}
    for run in api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", filters={"tags": {"$in": [tag]}}):
        if finished_only and run.state != "finished":
            continue
        if run.name not in latest or run.created_at > latest[run.name].created_at:
            latest[run.name] = run
    loaded = []
    for name, run in sorted(latest.items()):
        history = {}
        for key in keys:
            rows = [row for row in run.scan_history(keys=["optimizer_step", key])]
            steps = np.array([row["optimizer_step"] + 1 for row in rows])
            values = np.array([row[key] for row in rows], dtype=float)
            history[key] = (steps, values)
        loaded.append(dict(name=name, state=run.state, url=run.url, config=dict(run.config),
                           history=history))
    return loaded
