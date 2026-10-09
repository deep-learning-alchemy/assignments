"""Checkpoints, the probe, and validation rows for Problem 4, Inside the Hessian.

Two runs ship with the handout, each with model and optimizer state at the 13 steps in
STEPS (step 0 is rebuilt from the run's seed):

* "adamw": the default d8 recipe (AdamW, peak LR 3e-3, weight decay 0.1, linear decay).
* "sgdm":  SGD with momentum 0.9, peak LR 1, no weight decay, same schedule and data.

`load` also accepts your own runs: a TrainConfig, the run name printed by the launcher
(its directory under your model volume), or a path. Runs you train keep only their final
weights unless you set `keep_checkpoint_steps`.
"""

import json
import os
from dataclasses import fields
from functools import cache
from pathlib import Path

import numpy as np
import torch

import train as train_module
from model_config import LMConfig
from sharpness_logging import heldout_rows
from train import TrainConfig, build_model, training_run_name
from utils import MODEL_DIR

SHIPPED = {
    "adamw": "model-d8-lr0.003-tok614M-a3-curvature-v1-base-s42",
    "sgdm": "model-d8-lr1.0-tok614M-wd0.0-sgd-a3-curvature-v1-sgdm-lr1",
}
# Where the shipped runs live: the course's read-only shared volume on Modal.
SHIPPED_ROOT = Path(os.environ.get("A3_HESSIAN_RUNS", "/root/shared_data/a3_hessian/runs"))

TOKENS_PER_STEP = 64 * 1024
# 0, 0.2M, 1M, 4M, 10M, 20M, 50M, 100M, 200M, 300M, 400M, 500M, 614M tokens
STEPS = (0, 3, 15, 61, 153, 305, 763, 1526, 3052, 4578, 6104, 7630, 9375)
FINAL_STEP = 9375

VALIDATION_ROWS = 64


def run_dir(run):
    """Directory of a shipped run ("adamw", "sgdm"), one of your runs, or a path."""
    if isinstance(run, TrainConfig):
        run = training_run_name(run)
    if run in SHIPPED:
        path = SHIPPED_ROOT / SHIPPED[run]
        if not path.exists():
            raise FileNotFoundError(
                f"Shipped run {run!r} not found at {path}. On Modal it is on the shared course "
                "volume; elsewhere set A3_HESSIAN_RUNS to the directory that holds it.")
        return path
    path = Path(run) if Path(run).is_absolute() else Path(MODEL_DIR) / run
    if not (path / "metadata.json").exists():
        raise FileNotFoundError(f"Run {run!r} not found at {path} (no metadata.json). If it is "
                                "one of your training runs, has its training job finished?")
    return path


def train_config(run):
    """The TrainConfig a run was trained with, rebuilt from its metadata.json."""
    meta = json.loads((run_dir(run) / "metadata.json").read_text())
    known = {f.name for f in fields(TrainConfig)}
    keep = {k: v for k, v in meta.items() if k in known and k not in (
        "model_config", "train_dataset", "val_dataset", "metric_loggers", "keep_checkpoint_steps",
        "model_dir", "wandb_tags", "init_checkpoint_path", "fork_from_run", "fork_from_step")}
    return TrainConfig(model_config=LMConfig.from_dict(meta["model_config"]), **keep)


def load(run, step=FINAL_STEP, device="cuda", optimizer=False):
    """(model, optimizer or None, config) at `step` of `run`, model in eval mode, FP32.

    Step 0 is rebuilt from the run's model seed (initialization runs on the CPU, so it
    is identical on any GPU). Other steps read step_<step>.pt; for a run without kept
    steps, the final weights are in model.pt. The optimizer is rebuilt with the run's
    settings and its saved state loaded (needed to continue training).
    """
    config = train_config(run)
    model = build_model(config, device=device)
    opt = None
    if step:
        path = run_dir(run) / f"step_{step}.pt"
        if not path.exists() and step == FINAL_STEP and (run_dir(run) / "model.pt").exists():
            path = run_dir(run) / "model.pt"
            if optimizer:
                raise FileNotFoundError(f"{run_dir(run)} has no optimizer state at step {step}.")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state"])
        if optimizer:
            opt = build_optimizer(model, config)
            opt.load_state_dict(payload["optimizer_state"])
        del payload
    elif optimizer:
        opt = build_optimizer(model, config)
    model.float().eval()
    return model, opt, config


def build_optimizer(model, config):
    return train_module.build_optimizer(
        model, optimizer_name=config.optimizer_name, learning_rate=config.learning_rate,
        weight_decay=config.weight_decay, beta1=config.beta1, beta2=config.beta2)


@cache
def _probe_rows():
    return heldout_rows(TrainConfig())


def probe(device="cuda"):
    """The fixed probe: the 64 held-out training-source sequences of Problem 3
    (sharpness_logging.heldout_rows), which no run in this assignment trains on.
    Every measurement uses all 64 rows."""
    return _probe_rows().to(device)


def validation_rows(config, device="cuda", rows=VALIDATION_ROWS):
    """The first `rows` validation sequences, for quick loss checks during continuations."""
    _, val = train_module.load_train_and_val_datasets(config)
    return torch.from_numpy(np.asarray(val["val"].tokens[:rows], dtype=np.int64)).to(device)
