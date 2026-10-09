"""Problem 2: schedules, parameter averaging, rewarming, and horizon-free schedules.

    uv run python -m experiments.a3_optimization.p2_river_valley
"""

from modal_train import launch_training_jobs
from train import TrainConfig


EXPERIMENT_KEY = "a3-river-valley"
# Each run logs val_ema{decay}_loss next to val_loss at every evaluation.
EMA_DECAYS = (0.99, 0.999)

# Parts (a) and (b): one run per schedule, with evaluation-only EMAs.
SCHEDULE_RUNS = [
    TrainConfig(lr_schedule=lr_schedule, ema_decays=EMA_DECAYS, wandb_tags=(EXPERIMENT_KEY,))
    for lr_schedule in (
        "wsd0.0",  # constant after warmup
        "linear",
        "cos",
        "wsd0.2",  # plateau, then decay over the last 20% of training
    )
]

# Part (c): the LR falls to 0.1x between 40% and 45% of training and is back at
# full LR by 50%, before the same 20% cooldown as the wsd0.2 control above.
REWARM_RUNS = [
    TrainConfig(lr_schedule="wsd0.2", lr_dip=(0.4, 0.45, 0.5, 0.1), wandb_tags=(EXPERIMENT_KEY,)),
]

# Part (d): horizon-free schedules. warmup_steps fixes the warmup in updates,
# so it does not depend on the budget; sgdrT restarts every T updates.
HORIZON_FREE_RUNS = [
    TrainConfig(lr_schedule="invsqrt", warmup_steps=94, wandb_tags=(EXPERIMENT_KEY,)),
    TrainConfig(lr_schedule="sgdr2000", warmup_steps=94, min_lr_ratio=0.1,
                wandb_tags=(EXPERIMENT_KEY,)),
]

RUNS = SCHEDULE_RUNS + REWARM_RUNS + HORIZON_FREE_RUNS


# TODO: Part (b): try other EMA timescales.
# TODO: Part (c): vary the depth and duration of lr_dip.
# TODO: Part (d): run the horizon-free schedules to your largest budget, read
# them off at several smaller budgets, and train budget-tuned schedules
# (e.g. linear with a smaller num_train_sequences) at each budget.


def main():
    launch_training_jobs(RUNS)


if __name__ == "__main__":
    main()
