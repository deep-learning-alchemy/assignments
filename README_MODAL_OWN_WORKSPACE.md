# Modal In Your Own Workspace

Use this guide if you are following the course independently and do not have
access to the course's `cs312-shared-data` environment. Your data and training
jobs can live in the same Modal environment. No instructor scripts or access to
the course workspace are required.

You need a checkout of this repository, a Modal account with GPU access, and a
W&B account for the default online logging. Run every command from the repository
root. These commands assume a macOS/Linux shell (or WSL on Windows).

## 1. Install Dependencies And Authenticate

Install `uv` using its [installation guide](https://docs.astral.sh/uv/getting-started/installation/),
then run:

```bash
uv sync
uv run modal setup
uv run modal environment list
```

Authenticate to **your own Modal workspace**. Pick an existing environment, such
as `main`, or create one in that workspace:

```bash
uv run modal environment create my-environment
```

Throughout this guide, replace `my-environment` with the environment you chose.
Select it for subsequent Modal CLI commands:

```bash
uv run modal config set-environment my-environment
```

## 2. Configure The Repository

In [utils.py](utils.py), edit the three existing student-facing settings and add
the fourth line immediately below them:

```python
CONFIG_MODAL_ENVIRONMENT = "my-environment"
CONFIG_WANDB_ENTITY = "YOUR_WANDB_USERNAME_OR_TEAM"
CONFIG_WANDB_PROJECT = "assignments"
MODAL_SHARED_DATA_ENVIRONMENT = CONFIG_MODAL_ENVIRONMENT
```

The last line is important: it makes the launcher look for data in **your own
environment**, instead of `cs312-shared-data`. Despite the variable's name, it
does not share your data with other workspaces or change any permissions.
These settings persist in the file; you do not need to export environment
variables in each terminal session. Leave the data Volume name unchanged.

Create a [W&B account](https://wandb.ai/) if needed, obtain your API key, and save
it as a Modal Secret in the same environment:

```bash
uv run modal secret create --env my-environment --force dl-alchemy-wandb WANDB_API_KEY=YOUR_WANDB_API_KEY
```

Replace the placeholder with your actual key. Do not put the key in source files
or commit it. The key must have access to the W&B entity/project you configured.
Modal secrets are environment-specific.

## 3. Download And Upload The Data Once

Allow at least 25 GB of free local disk space. The prepared dataset is about
19.7 GB (18.3 GiB): 9,600,000 training sequences and 1,000 validation sequences,
each 1,024 tokens long.

Download the public raw-file release from
[`kothasuhas/dl_alchemy_seq9p6m_context1024`](https://huggingface.co/datasets/kothasuhas/dl_alchemy_seq9p6m_context1024):

```bash
uv run python -m download_data --output-dir ./data/dclm_9p6m_ctx1024
```

No Hugging Face account or key is needed. The script downloads both splits and
their metadata and verifies file sizes and SHA-256 checksums. Keep all of these
files together, not just the `.bin` files.

Create a persistent Volume in your training environment and upload the directory:

```bash
uv run modal volume create --env my-environment --version 2 hard-dl-dclm-v1
uv run modal volume put --env my-environment hard-dl-dclm-v1 ./data/dclm_9p6m_ctx1024 /datasets/dclm_9p6m_ctx1024
```

Only create the Volume once. Keep the destination path exactly as shown, without
a trailing slash. Check the uploaded layout:

```bash
uv run modal volume ls --env my-environment hard-dl-dclm-v1 /datasets/dclm_9p6m_ctx1024
```

It should contain `manifest.json`, `train/`, and `val/`. Each split directory
contains `metadata.json` and `tokens.bin`. See the
[Modal Volume CLI reference](https://modal.com/docs/cli/latest/volume) for upload
and inspection commands.

## 4. Launch A Default Run

```bash
uv run python -m experiments.smoke.modal_smoke_train
```

This launches a detached d8 run with 600,000 training sequences on an H100. Open
the printed Modal log link to follow it, and use W&B to inspect the loss curves.
The normal training run is approximately ten minutes; initial image building,
GPU allocation, data materialization, and compilation can add startup time.

The launcher mounts your data Volume read-only. It automatically creates a
separate writable `volume-dl_alchemy` Volume in the same environment for models
and checkpoints. Neither requires a local GPU.

The downloaded data is the **base dataset**, not a pre-shuffled seed cache. With
the default data seed of 42, training automatically materializes the requested
subset of the globally shuffled training data on temporary storage before
training begins. Validation uses the base validation split. This preparation
repeats for new training containers; it does not download the dataset from
Hugging Face again or modify the persistent base files.

After a successful upload, you may remove your local downloaded copy if you only
train on Modal. Keep the Modal Volume for subsequent runs. Your workspace pays
for training and persistent storage according to
[Modal's pricing](https://modal.com/pricing); a personal Volume is not free
storage and does not use course credits.
