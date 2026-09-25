from modal_train import launch_training_jobs
from train import TrainConfig


EXPERIMENT_KEY = "a1-basics-p5"


RUNS = [
    TrainConfig(run_name_suffix=EXPERIMENT_KEY, wandb_tags=(EXPERIMENT_KEY,)),
    TrainConfig(
        run_name_suffix=EXPERIMENT_KEY,
        wandb_tags=(EXPERIMENT_KEY,),
        batch_size=256,
    ),
    TrainConfig(
        run_name_suffix=EXPERIMENT_KEY,
        wandb_tags=(EXPERIMENT_KEY,),
        learning_rate=0.0003,
    ),
    TrainConfig(
        run_name_suffix=EXPERIMENT_KEY,
        wandb_tags=(EXPERIMENT_KEY,),
        learning_rate=0.009,
    ),
    TrainConfig(
        run_name_suffix=EXPERIMENT_KEY,
        wandb_tags=(EXPERIMENT_KEY,),
        lr_schedule="constant",
    ),
    TrainConfig(
        run_name_suffix=EXPERIMENT_KEY,
        wandb_tags=(EXPERIMENT_KEY,),
        beta1=0.5,
    ),
]


# TODO: Add additional runs that test the loss-curve factors you want to study.


def main():
    launch_training_jobs(RUNS)


if __name__ == "__main__":
    main()
