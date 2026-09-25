from modal_train import launch_training_jobs
from model_config import depth_model_config
from train import TrainConfig


EXPERIMENT_KEY = "a1-basics-p2"


RUNS = []
seen_configs = set()
for depth in range(4, 10):
    for learning_rate, lr_schedule, dropout in [
        (0.003, "linear", 0.0),
        (0.003, "constant", 0.0),
        (0.003, "linear", 0.2),
        (0.03, "linear", 0.0),
    ]:
        config_key = (depth, learning_rate, lr_schedule, dropout)
        if config_key in seen_configs:
            continue
        seen_configs.add(config_key)
        RUNS.append(
            TrainConfig(
                run_name_suffix=EXPERIMENT_KEY,
                wandb_tags=(EXPERIMENT_KEY,),
                model_config=depth_model_config(depth),
                learning_rate=learning_rate,
                lr_schedule=lr_schedule,
                dropout=dropout,
            )
        )


# TODO: Add your own slope-bending and scaling-law-breaking interventions for
# parts (b) and (c). The generated run name will encode the config changes.


def main():
    launch_training_jobs(RUNS)


if __name__ == "__main__":
    main()
