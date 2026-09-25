from modal_train import launch_training_jobs
from train import TrainConfig


EXPERIMENT_KEY = "a1-basics-p5"


RUNS = [
    TrainConfig(
        run_name_suffix=EXPERIMENT_KEY,
        wandb_tags=(EXPERIMENT_KEY,),
        deterministic=True,
    ),
    TrainConfig(
        run_name_suffix=EXPERIMENT_KEY,
        wandb_tags=(EXPERIMENT_KEY,),
        deterministic=True,
        perturb_one_token=True,
    ),
]


def main():
    launch_training_jobs(RUNS)


if __name__ == "__main__":
    main()
