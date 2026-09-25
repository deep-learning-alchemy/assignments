from modal_train import launch_training_jobs
from train import TrainConfig


EXPERIMENT_KEY = "a1-basics-p7"


RUNS = [
    TrainConfig(run_name_suffix=EXPERIMENT_KEY, wandb_tags=(EXPERIMENT_KEY,)),
    TrainConfig(
        run_name_suffix=EXPERIMENT_KEY,
        wandb_tags=(EXPERIMENT_KEY,),
        learning_rate=0.027,
        warmup_percent=0.0,
    ),
]


# Parameter, gradient, and activation statistics are logged by the default
# metric loggers. To probe other statistics, extend TrainConfig.metric_loggers
# with a module-level MetricLogger function that installs temporary forward
# hooks on ctx.model.


def main():
    launch_training_jobs(RUNS)


if __name__ == "__main__":
    main()
