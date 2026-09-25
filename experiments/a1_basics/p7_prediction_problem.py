"""Blank scaffold for the optional student-designed prediction problem.

Keep your proposed diff under 100 lines, make the answer mechanically
checkable, and avoid changing shared training code unless the question is
explicitly about such a change.
"""

from modal_train import launch_training_jobs
from model_config import LMConfig
from train import TrainConfig


EXPERIMENT_KEY = "a1-basics-p7"


def d8w64_config():
    return LMConfig(
        name="d8w64",
        vocab_size=4096,
        context_length=1024,
        hidden_size=64,
        intermediate_size=224,
        num_hidden_layers=8,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=64,
    )


RUNS = [
    TrainConfig(run_name_suffix=EXPERIMENT_KEY, wandb_tags=(EXPERIMENT_KEY,)),
]


# Model mutations such as zeroing lm_head do not need to be TrainConfig fields.
# Use a clearly scoped code diff against model construction or initialization if
# your prediction problem needs such a model intervention.


def main():
    launch_training_jobs(RUNS)


if __name__ == "__main__":
    main()
