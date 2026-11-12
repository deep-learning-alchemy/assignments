# Non-Modal GPU Guide

This repo is still usable without Modal. Modal is only the default course launch
path. The core training entrypoint is the normal Python function
`train(TrainConfig(...))`, and it runs on local CUDA when `torch.cuda.is_available()`
is true.

## Viability

The non-Modal path is viable:

- training does not require Modal;
- training reads raw `.bin` token files through a NumPy memmap;
- checkpoints and final models save to a local `model_dir`;
- W&B can use your normal local `wandb login`;
- one local CUDA GPU uses `torch.compile(mode="reduce-overhead")`;
- multiple visible CUDA GPUs run through `torch.nn.DataParallel`, with compile
  disabled.

The main thing a non-Modal user must do is download the data locally and choose
local storage directories.

## Setup

Install `uv` using the official guide: <https://docs.astral.sh/uv/getting-started/installation/>.

From the repo root:

```bash
uv python install 3.10
uv sync
```

Check CUDA:

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

If this prints `False`, the default run will fall back to CPU and will be too
slow for normal use.

## Configure Local Storage

Open [../utils.py](../utils.py) and edit the student-facing block near the top.
For a non-Modal GPU user, the important fields are:

```python
CONFIG_WANDB_ENTITY = "YOUR_WANDB_USERNAME_OR_TEAM"
CONFIG_WANDB_PROJECT = "assignments"
CONFIG_SCRATCH_ROOT = "/path/with/space/dl_alchemy"
```

By default, local users use `~/dl_alchemy`. Set `CONFIG_SCRATCH_ROOT` explicitly if your
home directory is small or you want to use an external disk or cluster scratch
filesystem.

Derived defaults:

```text
MODEL_DIR = CONFIG_SCRATCH_ROOT/ckpts
DATA_DIR = CONFIG_SCRATCH_ROOT/data
```

You can override those separately with `CONFIG_MODEL_DIR` and `CONFIG_DATA_DIR`.

## Download Data

For Assignment 6, follow the [A6 local instructions](../experiments/a6_generalization/README.md#use-your-own-gpu)
to download the DNA data, pretrained checkpoints, and references from Hugging Face.

Download the raw binary dataset:

```bash
uv run python -m download_data
```

This downloads plain files from `kothasuhas/dl_alchemy_seq9p6m_context1024` into
`DATA_DIR`. The expected layout is:

```text
DATA_DIR/dclm_9p6m_ctx1024/train/tokens.bin
DATA_DIR/dclm_9p6m_ctx1024/train/metadata.json
DATA_DIR/dclm_9p6m_ctx1024/val/tokens.bin
DATA_DIR/dclm_9p6m_ctx1024/val/metadata.json
```

The default `TrainConfig` uses `data_seed=42`. If a preshuffled seed-42 local
cache is not present, training loads the base dataset above and materializes the
globally shuffled prefix once before training starts. This costs extra startup
time and temporary disk roughly proportional to:

```text
num_train_sequences * context_length * token_dtype_size
```

For the default `600,000 x 1024` run, this is around 1.2 GiB when tokens are
`uint16`, or 2.4 GiB if a future dataset uses `uint32`.

For quick debugging only, `TrainConfig(data_seed=None)` skips shuffling and reads
the base data in order. That changes the data order relative to the default
course recipe.

## Run Locally

The example launcher is [local_default_train.py](local_default_train.py):

```python
import torch

from train import TrainConfig, train


CONFIG = TrainConfig(
    num_train_sequences=600_000,
    run_name_suffix="local-gpu",
    wandb_online=False,
)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected.")
    train(CONFIG)
```

Run it from the repo root:

```bash
uv run python -m gpu.local_default_train
```

## W&B

Modal secrets are not used outside Modal. For local GPU runs:

```bash
uv run wandb login
```

Then set `wandb_online=True` in your `TrainConfig`. The W&B entity/project come
from `utils.py`, unless you pass `wandb_entity=` or `wandb_project=` directly in
`TrainConfig`.

## Custom Data

For the default dataset, prefer setting `CONFIG_DATA_DIR` in `utils.py` and
running `download_data`.

For a custom preprocessed token dataset, point the config at a split directory:

```python
from data import token_dataset_at
from train import TrainConfig, train

train(
    TrainConfig(
        num_train_sequences=100_000,
        train_dataset=token_dataset_at("/path/to/my_data/train", tag="my-data"),
        val_dataset=token_dataset_at("/path/to/my_data/val", tag="my-data"),
        data_seed=None,
        wandb_online=False,
    )
)
```

Each split directory must contain `tokens.bin` and `metadata.json` with the same
shape/metadata convention as the default data.

## Slurm

On a generic Slurm cluster, the most robust first test is to write a normal
Python launcher like `gpu/local_default_train.py`, then submit it with an
ordinary `sbatch` script:

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd /path/to/assignments
uv run python -m gpu.local_default_train
```

After that works, you can use the shared launcher
[../slurm_launch.py](../slurm_launch.py) for experiment grids. For a new cluster,
open that file and edit the `Slurm customization block` near the top.

The most likely fields to change are:

```python
QUEUE_CONFIGS = {
    "gpu": {"account": None, "partition": "gpu", "gpu_request": "gres"},
}

ENV_SETUP_COMMANDS = ""  # Example: "module load cuda/12.1"

LOCAL_SLURM_CONFIG = {
    "repo_dir": "/path/to/assignments",
    "slurm_log_dir": "/path/to/slurm_logs",
    "temp_script_dir": "/path/to/temp_scripts",
    "uv_path": "uv",
    "uv_project_environment": "/path/to/.venv",
    "uv_cache_dir": "/path/to/uv_cache",
    "slurm_exclude": "",
}
```

Use your cluster's real partition/account names. If your cluster does not use
Slurm accounts, leave the queue's `account` as `None` and the launcher will omit
the `#SBATCH --account=...` line. If your cluster supports
`#SBATCH --gpus-per-task=...`, omit `gpu_request` or set it to
`"gpus-per-task"`. If it uses `#SBATCH --gres=gpu:...`, set `gpu_request` to
`"gres"`.
If your cluster requires environment setup before running Python, put those shell
commands in `ENV_SETUP_COMMANDS`; otherwise leave it as the empty string.

You can also leave individual `LOCAL_SLURM_CONFIG` entries as `None`; those
values then come from [../utils.py](../utils.py) and its environment-variable
overrides. This is usually enough if you already configured `CONFIG_SCRATCH_ROOT`
and run from the repo checkout you want Slurm jobs to use.

Then create a small Python launch file that imports the objects used in the
function call string:

```python
from slurm_launch import launch_job
from train import TrainConfig, train

launch_job(
    "train(TrainConfig(num_train_sequences=600_000, run_name_suffix='slurm-gpu', wandb_online=False))",
    queue="gpu",
    gpus=1,
    mem=64,
    cpus=16,
)
```

The launcher stages a short Python script and submits it with `sbatch`. The
function call string is evaluated in a script that imports your launch file, so
keep imports like `from train import TrainConfig, train` at module scope.
