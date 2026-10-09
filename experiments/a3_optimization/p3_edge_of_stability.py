"""Problem 3: sharpness and the edge of stability.

    uv run python -m experiments.a3_optimization.p3_edge_of_stability

Both helpers attach sharpness_logging.SharpnessLogger, which logs to W&B under
logging/sharpness/*. Read the results back with experiments.a3_optimization.sharpness.
"""

from metric_logging import AFTER_BACKWARD, MetricLogger
from modal_train import launch_training_jobs
from sharpness_logging import SharpnessLogger
from train import TrainConfig


EXPERIMENT_KEY = "a3-edge-of-stability"
FRACTION = 0.32  # of an epoch: 3,000 updates, 197M tokens at batch size 64


def full_batch(num_sequences, updates, every=25, **changes):
    """Full-batch gradient descent on the first `num_sequences` training sequences.

    Every update uses all of them, so one epoch is one update. lambda_max is
    measured on the same sequences every `every` updates. Keyword arguments
    override any setting, e.g. learning_rate=0.02 or optimizer_name="adamw".
    """
    optimizer_name = changes.get("optimizer_name", "sgd")
    settings = dict(
        num_train_sequences=num_sequences,
        batch_size=num_sequences,
        num_epochs=float(updates),
        optimizer_name=optimizer_name,
        # plain gradient descent for sgd (its default beta1 is momentum); AdamW keeps 0.9
        beta1=0.0 if optimizer_name == "sgd" else TrainConfig.beta1,
        lr_schedule="constant",
        weight_decay=0.0,
        grad_norm=None,
        embedding_init_scale=16.0,
        num_evals=10,
        metric_loggers=TrainConfig.metric_loggers
        + (MetricLogger(AFTER_BACKWARD, SharpnessLogger(every=every)),),
        wandb_tags=(EXPERIMENT_KEY,),
    )
    settings.update(changes)
    return TrainConfig(**settings)


def measurement_steps(batch_size):
    """Every 16 updates to update 208, then every 100 to 1,500, then every 300, at
    batch size 64; other batch sizes measure at the same token counts."""
    at_64 = [*range(0, 209, 16), *range(300, 1501, 100), *range(1800, 3001, 300)]
    total = int(FRACTION * (600_000 // batch_size))
    return sorted({round(u * 64 / batch_size) for u in at_64} & set(range(total)))


def minibatch(batch_size=64, **changes):
    """The course recipe with a constant learning rate after warmup, cut at 0.32
    epochs, measuring the preconditioned sharpness on 64 held-out sequences.

    Warmup is 3.125% of the run: 93 updates at batch size 64, and the same
    6M tokens at every batch size.
    """
    settings = dict(
        num_epochs=FRACTION,
        warmup_percent=0.03125,
        lr_schedule="wsd0.0",
        batch_size=batch_size,
        num_micro_batches=max(1, batch_size // 64),
        metric_loggers=TrainConfig.metric_loggers
        + (MetricLogger(AFTER_BACKWARD, SharpnessLogger(
            steps=measurement_steps(batch_size), probe="heldout", kinds=("pre",))),),
        wandb_tags=(EXPERIMENT_KEY,),
    )
    settings.update(changes)
    return TrainConfig(**settings)


# Part (a): full-batch training.
FULL_BATCH_RUNS = [
    full_batch(32, 1500, learning_rate=0.01),  # (a) i: the clean case
]

# Part (b): minibatch training.
MINIBATCH_RUNS = [
    minibatch(),  # (b): the base recipe
]

RUNS = FULL_BATCH_RUNS + MINIBATCH_RUNS


# TODO: Part (a) ii: halve and double the learning rate mid-run
# (lr_schedule="step750x0.5" and "step750x2").
# TODO: Part (c): your own hyperparameter exploration. Both helpers take any TrainConfig
# override, e.g. minibatch(batch_size=256) or minibatch(learning_rate=1e-3).


def main():
    launch_training_jobs(RUNS)


if __name__ == "__main__":
    main()
