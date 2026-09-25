from modal_train import launch_training_jobs
from train import TrainConfig


EXPERIMENT_KEY = "a1-basics-p3"


RUNS = [
    TrainConfig(
        run_name_suffix=f"{EXPERIMENT_KEY}-deterministic-reference-1",
        wandb_tags=(EXPERIMENT_KEY,),
        deterministic=True,
    ),
    TrainConfig(
        run_name_suffix=f"{EXPERIMENT_KEY}-deterministic-reference-2",
        wandb_tags=(EXPERIMENT_KEY,),
        deterministic=True,
    ),
    *(
        TrainConfig(
            run_name_suffix=EXPERIMENT_KEY,
            wandb_tags=(EXPERIMENT_KEY,),
            model_seed=seed,
            data_seed=seed,
        )
        for seed in range(4, 14)
    ),
    *(
        TrainConfig(
            run_name_suffix=EXPERIMENT_KEY,
            wandb_tags=(EXPERIMENT_KEY,),
            model_seed=seed,
        )
        for seed in range(4, 9)
    ),
    *(
        TrainConfig(
            run_name_suffix=EXPERIMENT_KEY,
            wandb_tags=(EXPERIMENT_KEY,),
            data_seed=seed,
        )
        for seed in range(4, 9)
    ),
]


# TODO: For hardware nondeterminism, run the same config on at least two GPU
# types by passing `gpu=...` to launch_training_jobs from a local experiment.


def main():
    launch_training_jobs(RUNS)


if __name__ == "__main__":
    main()
