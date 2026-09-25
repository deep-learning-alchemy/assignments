from modal_train import launch_training_jobs
from train import TrainConfig


EXPERIMENT_KEY = "a1-basics-p1"


LEARNING_RATES = (0.001, 0.003, 0.009)
BATCH_SIZES = (16, 64, 256)
WEIGHT_DECAYS = (0.03, 0.1, 0.3)
WARMUP_PERCENTS = (0.0, 0.03, 0.1)


RUNS = [
    TrainConfig(run_name_suffix=EXPERIMENT_KEY, wandb_tags=(EXPERIMENT_KEY,)),
    *(
        TrainConfig(
            run_name_suffix=EXPERIMENT_KEY,
            wandb_tags=(EXPERIMENT_KEY,),
            learning_rate=lr,
        )
        for lr in LEARNING_RATES
        if lr != TrainConfig.learning_rate
    ),
    *(
        TrainConfig(
            run_name_suffix=EXPERIMENT_KEY,
            wandb_tags=(EXPERIMENT_KEY,),
            batch_size=batch_size,
        )
        for batch_size in BATCH_SIZES
        if batch_size != TrainConfig.batch_size
    ),
    *(
        TrainConfig(
            run_name_suffix=EXPERIMENT_KEY,
            wandb_tags=(EXPERIMENT_KEY,),
            weight_decay=weight_decay,
        )
        for weight_decay in WEIGHT_DECAYS
        if weight_decay != TrainConfig.weight_decay
    ),
    *(
        TrainConfig(
            run_name_suffix=EXPERIMENT_KEY,
            wandb_tags=(EXPERIMENT_KEY,),
            warmup_percent=warmup_percent,
        )
        for warmup_percent in WARMUP_PERCENTS
        if warmup_percent != TrainConfig.warmup_percent
    ),
]


# TODO: Add paired sweeps for part (b), then replace RUNS with the run budget
# you actually commit to for the 3/5/10-run protocols in part (c).


def main():
    launch_training_jobs(RUNS)


if __name__ == "__main__":
    main()
