"""Problem 1: forked runs, weight averaging, and ensembles.

Run the stages in order, waiting for each to finish:

    uv run python -m experiments.a3_optimization.p1_mode_connectivity prefix
    uv run python -m experiments.a3_optimization.p1_mode_connectivity branches
    uv run python -m experiments.a3_optimization.p1_mode_connectivity measure
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from train import TrainConfig, training_run_name


EXPERIMENT_KEY = "a3-mode-connectivity"
PREFIX_STEPS = (0, 50, 100, 250, 500, 1000)  # x
BRANCH_STEPS = (25, 50, 100, 250, 500)  # y
BRANCH_ORDER_SEEDS = (1, 2)
RESULTS_PATH = Path(__file__).resolve().parent / "results" / f"{EXPERIMENT_KEY}.json"

# Constant LR after warmup. The schedule (and so the warmup) is set by the full
# 600M-token run length; stop_at_step only ends training early.
BASE = TrainConfig(lr_schedule="wsd0.0", wandb_tags=(EXPERIMENT_KEY,))



def prefix_for(base):
    """The shared prefix run, keeping model and optimizer state at every x > 0."""
    return replace(
        base,
        keep_checkpoint_steps=tuple(x for x in PREFIX_STEPS if x > 0),
        stop_at_step=max(PREFIX_STEPS),
    )


PREFIX = prefix_for(BASE)


def branch(x, order_seed, base=BASE):
    """One branch: x shared updates, then max(BRANCH_STEPS) updates in its own data order.

    x = 0 starts from the same initialization (same model_seed) without a fork.
    """
    fork = {}
    if x > 0:
        fork = dict(fork_from_run=training_run_name(prefix_for(base)), fork_from_step=x)
    return replace(
        base,
        batch_order_seed=order_seed,
        stop_at_step=x + max(BRANCH_STEPS),
        keep_checkpoint_steps=tuple(x + y for y in BRANCH_STEPS),
        **fork,
    )


BRANCHES = {
    x: [branch(x, seed) for seed in BRANCH_ORDER_SEEDS] for x in PREFIX_STEPS
}


def pairs():
    """((run_a, step), (run_b, step)) for every (x, y) cell."""
    cells = []
    for x, (a, b) in BRANCHES.items():
        for y in BRANCH_STEPS:
            cells.append(((training_run_name(a), x + y), (training_run_name(b), x + y)))
    return cells


def check_prefix_checkpoints(prefix=PREFIX):
    """Fail before launching if the prefix's kept checkpoints are not on the volume yet."""
    from pathlib import PurePosixPath

    from modal.exception import NotFoundError

    from modal_train import _model_volume_path
    from modal_utils import user_volume

    try:
        entries = user_volume.listdir(_model_volume_path(training_run_name(prefix)))
    except NotFoundError:
        entries = []
    names = {PurePosixPath(entry.path).name for entry in entries}
    missing = [step for step in prefix.keep_checkpoint_steps if f"step_{step}.pt" not in names]
    if missing:
        raise SystemExit(
            f"Prefix {training_run_name(prefix)!r} has no checkpoints for steps {missing}. "
            "Run the prefix stage and wait for it to finish."
        )


# TODO: For part (d), vary the training choices by building prefix_for(base)
# and branch(x, seed, base=base) from other base configs.
# TODO: Plot endpoint, ensemble, and weight-averaged losses and the barrier
# B(x, y) from RESULTS_PATH.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prefix", "branches", "measure"))
    args = parser.parse_args()
    if args.stage == "prefix":
        from modal_train import launch_training_jobs

        launch_training_jobs([PREFIX])
    elif args.stage == "branches":
        from modal_train import launch_training_jobs

        check_prefix_checkpoints()
        launch_training_jobs([run for runs in BRANCHES.values() for run in runs])
    else:
        from experiments.a3_optimization.connectivity import measure_pairs_on_modal

        rows = measure_pairs_on_modal(pairs())
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"Wrote {len(rows)} cells to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
