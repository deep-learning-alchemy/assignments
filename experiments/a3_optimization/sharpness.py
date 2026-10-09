"""Read a run's sharpness measurements back from wandb and summarize them.

    from experiments.a3_optimization.p3_edge_of_stability import FULL_BATCH_RUNS, MINIBATCH_RUNS
    from experiments.a3_optimization.sharpness import history, summarize, window_median

    config = FULL_BATCH_RUNS[0]
    rows = history(config)
    summarize(rows)                             # gradient descent: threshold 2
    summarize(rows, kind="pre")                 # AdamW: threshold 38
    window_median(history(MINIBATCH_RUNS[0]), MINIBATCH_RUNS[0], 100e6, 180e6)  # minibatch

history() returns one dict per measured update with keys step, train_loss, lr,
lambda_max, eta_lambda (and lambda_max_pre, eta_lambda_pre for AdamW), plus
the training loss at every update under history(...).losses.
"""

import statistics

from train import checked_train_config, training_run_name

THRESHOLDS = {"raw": 2.0, "pre": 38.0}


class History(list):
    losses: list  # (update, training loss) at every update


def history(config):
    import wandb

    config = checked_train_config(config)
    name = training_run_name(config)
    runs = wandb.Api().runs(f"{config.wandb_entity}/{config.wandb_project}", filters={"display_name": name})
    if not runs:
        raise LookupError(f"No wandb run named {name}")
    run = max(runs, key=lambda r: r.created_at)
    rows, losses = {}, {}
    for r in run.scan_history():
        step = r.get("optimizer_step")
        if step is None:
            continue
        if r.get("train_loss") is not None:
            losses[step] = r["train_loss"]
        sharpness = {k.removeprefix("logging/sharpness/"): v for k, v in r.items()
                     if k.startswith("logging/sharpness/") and v is not None}
        if "lambda_max" in sharpness or "lambda_max_pre" in sharpness:
            rows[step] = {"step": step, "lr": r.get("learning_rate"), **sharpness}
    out = History(rows[s] for s in sorted(rows))
    for row in out:
        row["train_loss"] = losses.get(row["step"])
    out.losses = sorted(losses.items())
    return out


def summarize(rows, kind="raw", loss_floor=1.0, settle=100, start=0):
    """For full-batch runs.

    crossing: first measured update at or after `start` with eta * lambda above
    the threshold. plateau: median eta * lambda from `settle` updates after the
    crossing while the training loss is above `loss_floor`. exit (gradient
    descent only): first update after which every measurement is below 2.
    loss_up_before / loss_up_after: fraction of updates on which the training
    loss rose, before the crossing and after it (while above `loss_floor`)."""
    key = "eta_lambda" if kind == "raw" else "eta_lambda_pre"
    threshold = THRESHOLDS[kind]
    m = [r for r in rows if r.get(key) is not None and r["step"] >= start]
    crossing = next((r["step"] for r in m if r[key] > threshold), None)
    out = {"initial": m[0][key] if m else None, "crossing": crossing, "plateau": None}
    if crossing is not None:
        plateau = [r[key] for r in m if r["step"] >= crossing + settle and (r["train_loss"] or 0) > loss_floor]
        out["plateau"] = statistics.median(plateau) if plateau else None
    if kind == "raw":
        out["exit"] = next((r["step"] for i, r in enumerate(m) if crossing is not None and r["step"] > crossing
                            and all(x[key] <= threshold for x in m[i:])), None)
    losses = getattr(rows, "losses", [])
    if crossing is not None and losses:
        ups = [(u, b > a) for (_, a), (u, b) in zip(losses, losses[1:])]
        before = [up for u, up in ups if u <= crossing]
        after = [up for u, up in ups if u > crossing and dict(losses)[u] > loss_floor]
        out["loss_up_before"] = sum(before) / len(before) if before else None
        out["loss_up_after"] = sum(after) / len(after) if after else None
    return out


def window_median(rows, config, lo_tokens, hi_tokens, key="eta_lambda_pre"):
    """For minibatch runs: median of `key` over measurements taken between
    lo_tokens and hi_tokens training tokens (e.g. 100e6, 200e6). Start lo_tokens
    after warmup (about 6.2M tokens in minibatch())."""
    per_update = config.batch_size * config.train_dataset.context_length
    values = [r[key] for r in rows if r.get(key) is not None and lo_tokens <= r["step"] * per_update < hi_tokens]
    return statistics.median(values) if values else None
