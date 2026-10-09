# CS312 Deep Learning Alchemy

This repo trains small autoregressive language models on a prepared tokenized
DCLM dataset using Modal. This README is the setup guide for a student starting
from scratch.

For the model, data, optimizer, loss, and checkpointing details, read
[MACHINE_LEARNING.md](MACHINE_LEARNING.md).

For running on your own CUDA GPU or a non-course Slurm cluster without Modal, read
[gpu/README.md](gpu/README.md).

For using Modal in your own workspace without access to the course's shared data,
read [README_MODAL_OWN_WORKSPACE.md](README_MODAL_OWN_WORKSPACE.md).

Course handouts are distributed separately.

The default-run and LR-tuning examples below cover setup and the experiment
workflow. Assignment 3 starters and measurement tools are in
[experiments/a3_optimization](experiments/a3_optimization/README.md). The
assignment handout is distributed separately; solutions are not bundled.


## 0. Setting up Modal

### Accepting the Invitation

Wait for the course staff to send a Modal invitation to your Stanford email address.

When you receive this invitation, click to sign into Modal with Google, and ensure that you are using the Google account associated with your Stanford email. Note that if you use another email address, your account will be kicked out of the course Modal workspace within a few minutes, and a new invitation email will be sent to you. This process will continue until you sign into Modal with your Stanford email to accept the invite.

When you accept the invitation successfully, a Modal environment will be created for you, named `cs312-<your SUNet ID>`. You may get an error that you don't have access to this environment, but this should go away after waiting a couple minutes and refreshing the page (as long as you are using the Modal account associated with your Stanford email). Once this Modal environment has been created, you may need to wait a minute for our automatic GPU manager to allocate your environment permissions for 2 concurrent GPUs.

### Modal Policy

* For every assignment, you have a budget of 48 GPU hours. If you go beyond this, our automatic GPU monitor will set your concurrent GPU limit to 0 and cancel jobs that you try to run.
* Do all work within your assigned environment, and do not create any other environments. These unauthorized environments will have their GPU allocation set to 0, and the course staff will be notified.
* Do not try to change your concurrent GPU usage from 2. Our GPU monitor will cancel your jobs, reset your concurrent GPU usage, and notify course staff.
* We strongly recommend that you use H100s unless explicitly instructed otherwise.

## 1. Prerequisites

You need:

- a terminal
- `git`
- a Modal account with access to the course workspace and your assigned Modal
  environment
- a W&B account if you want online experiment logging

The commands below assume a macOS or Linux shell. On Windows, use WSL or
translate the shell commands to PowerShell.

## 2. Install `uv`

This project uses `uv` to install Python and manage dependencies.

Follow the official uv installation guide and choose the method for your
operating system: <https://docs.astral.sh/uv/getting-started/installation/>.

Check that it works:

```bash
uv --version
```

## 3. Get The Repo

Clone the course repo, then enter it:

```bash
git clone https://github.com/deep-learning-alchemy/assignments.git
cd assignments
```

If your instructor already gave you a checkout, just `cd` into that checkout
instead.

## 4. Fill In `utils.py`

Open [utils.py](utils.py) and edit the student-facing configuration block near
the top of the file:

```python
CONFIG_MODAL_ENVIRONMENT = "YOUR_MODAL_ENVIRONMENT"
CONFIG_WANDB_ENTITY = "YOUR_WANDB_USERNAME_OR_TEAM"
CONFIG_WANDB_PROJECT = "assignments"
```

Replace the placeholders with your assigned Modal environment and your W&B
username or team. `CONFIG_WANDB_PROJECT` can stay `assignments`.

Non-Modal users only: local GPU runs use `CONFIG_SCRATCH_ROOT`,
`CONFIG_MODEL_DIR`, and `CONFIG_DATA_DIR`; see [gpu/README.md](gpu/README.md).
Most Modal students should not need to change the path settings in `utils.py`.

## 5. Install Python Dependencies

The repo pins Python `3.10` in `.python-version`. Install Python and the
repo dependencies:

```bash
uv python install 3.10
uv sync
```

This creates a repo-local `.venv/`. You do not need to activate it; use
`uv run ...` for commands in this repo.

Check the environment:

```bash
uv run python --version
uv run python -c "import torch, modal; print('torch', torch.__version__); print('modal ok')"
```

Run repo scripts from the repo root using module syntax, such as
`uv run python -m experiments.smoke.modal_smoke_train`. This keeps Python's
import path pointed at the whole repo.

## 6. Configure Modal

Run Modal setup once:

```bash
uv run modal setup
```

This opens a browser flow and writes Modal credentials to your machine.

Check that Modal can see your account and environment:

```bash
uv run modal config set-environment YOUR_MODAL_ENVIRONMENT
uv run modal token info
uv run modal environment list
```

Set the same environment name as `CONFIG_MODAL_ENVIRONMENT` in `utils.py`. The
`modal config` command is for the Modal CLI; `utils.py` is what the Python
launchers read.

If your assigned environment is missing from the environment list, ask your
instructor to add you. That is usually a permissions issue, not a code issue.

## 7. Configure W&B

W&B is used for online experiment tracking. Create a W&B account first if you do
not already have one.

Create a Modal Secret named `dl-alchemy-wandb` in your assigned Modal
environment. Modal secrets are environment-specific, so use the same environment
name you put in `CONFIG_MODAL_ENVIRONMENT`:

```bash
uv run modal secret create --env YOUR_MODAL_ENVIRONMENT --force dl-alchemy-wandb WANDB_API_KEY=YOUR_WANDB_API_KEY
```

Replace `YOUR_MODAL_ENVIRONMENT` with your assigned Modal environment and
`YOUR_WANDB_API_KEY` with the key from your W&B account. Do not paste real keys
into Python files, experiment launchers, or README edits.

You only need the W&B secret for runs with `wandb_online=True`. Offline runs do
not require W&B setup.

## 8. Launch One Default Run

Start by launching one default training run:

```text
experiments/smoke/modal_smoke_train.py
```

Launch it:

```bash
uv run python -m experiments.smoke.modal_smoke_train
```

This submits a detached Modal job with the default recipe:

```text
model: d8
train sequences: 600,000
context length: 1024
batch size: 64
optimizer: AdamW
learning rate: 0.003
schedule: linear decay
data seed: 42
default GPU: H100
```

This one command checks the whole normal path: Modal launch, shared data access,
W&B logging, training, checkpoint writing, and final model saving. The launcher
prints the timestamped Modal app name plus a `modal_call_id`, then exits after
submission. The training job continues on Modal.

If everything is working normally, expect this default run to take about ten
minutes. Open the printed Modal app link to read the live run logs.

The run writes checkpoints and final model artifacts to your Modal Volume under
`/root/data/ckpts`.

## 9. Change Training Configs

The main thing to inspect is `TrainConfig` in [train.py](train.py). It is the
default recipe for a run and the main interface for experiments.

Common fields to change:

```python
TrainConfig(
    learning_rate=3e-3,
    lr_schedule="linear",
    num_train_sequences=600_000,
    batch_size=64,
    model_name="d8",
    data_seed=42,
    run_name_suffix="my-experiment",
    wandb_tags=("my-experiment",),
)
```

Experiment launchers usually create one or more `TrainConfig` objects and pass
them to `launch_training_jobs`. Prefer changing configs in Python over adding
command-line flags for every experiment.

## 10. Walk Through LR Tuning

The learning-rate tuning example is a complete experiment template:

```text
experiments/lr_tuning/launch_lr_tuning.py
experiments/lr_tuning/plot_lr_tuning.py
```

The launcher declares one experiment key and a list of learning rates:

```python
EXPERIMENT_KEY = "lr-tuning-v1"
LEARNING_RATES = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)
```

Each run starts from the default training recipe and overrides only the
learning rate:

```python
TrainConfig(
    learning_rate=learning_rate,
    run_name_suffix=EXPERIMENT_KEY,
    wandb_tags=(EXPERIMENT_KEY,),
)
```

This is the intended pattern for new experiments: keep the default recipe in
`TrainConfig`, override the field you are studying, and tag all runs with one
shared experiment key. Use a new `EXPERIMENT_KEY` when you rerun a changed
experiment so the plotting script does not mix old and new runs.

Launch the sweep:

```bash
uv run python -m experiments.lr_tuning.launch_lr_tuning
```

The launcher submits detached Modal jobs in your configured environment. Each
launcher invocation creates one timestamped Modal app, and each config in the
sweep becomes one function call inside that app. By default, all learning rates
can run concurrently on the default H100 GPU. To throttle a larger sweep, pass
`max_parallel_runs=...` to `launch_training_jobs`. To request a different GPU,
add `gpu=...` to the `launch_training_jobs` call in the launcher.

After the jobs have logged to W&B, make the plot:

```bash
uv run python -m experiments.lr_tuning.plot_lr_tuning
```

The plotting script queries the W&B project configured in `utils.py`, selects
runs tagged with `EXPERIMENT_KEY`, and writes:

```text
experiments/lr_tuning/plots/lr-tuning-v1_loss_summary.png
```

The left subplot shows the validation-loss trajectories. The right subplot
shows final validation loss as a function of learning rate. When you make your
own experiment, copy this structure: one launch script that creates a list of
`TrainConfig` objects and one plotting script that loads runs by a shared W&B
tag.

## 11. Check Your Modal Usage

Print usage for your configured Modal environment:

```bash
uv run python -m scripts.modal_usage
```

For Modal's raw CLI summary:

```bash
uv run modal environment billing summary --for "this month"
```

## 12. Useful Commands

List Modal secrets:

```bash
uv run modal secret list --env YOUR_MODAL_ENVIRONMENT
```

Update your W&B key:

```bash
uv run modal secret create --env YOUR_MODAL_ENVIRONMENT --force dl-alchemy-wandb WANDB_API_KEY=YOUR_NEW_WANDB_API_KEY
```

Open the Modal dashboard:

```bash
uv run modal dashboard
```

Inspect the current git branch:

```bash
git branch --show-current
```

Inspect one tokenized data row:

```bash
uv run python -m scripts.inspect_data_row
```

This starts a CPU-only Modal job in your configured environment. It reads one
row from the shared binary data, downloads the small course SentencePiece
tokenizer, and prints token IDs plus decoded text.

## 13. Understand Data And Outputs

Normal Modal training uses two Modal Volumes:

- shared read-only course data at `/root/shared_data`
- your writable per-environment storage at `/root/data`

The default training data is already prepared by the instructors in the shared
data Volume.

Note: Students should NOT modify any aspect of the shared environment by 
modifying the data or uploading their own volumes. This data is shared between
all students and should remain consistent. Students do not need to download
or preprocess it for normal runs.



Training outputs, checkpoints, and custom datasets live in your writable
`/root/data` Volume. They persist after a Modal job finishes.

For online W&B runs, Modal uses `/tmp/wandb` as W&B's local staging directory.
That directory is ephemeral, but W&B metrics that have already synced remain in
W&B. Checkpoint-based resume does not depend on `/tmp/wandb`; checkpoints and
the W&B run id are stored under `/root/data/ckpts`.

The default shared data files are mounted at:

```text
/root/shared_data/datasets/dclm_9p6m_ctx1024/train/tokens.bin
/root/shared_data/datasets/dclm_9p6m_ctx1024/val/tokens.bin
/root/shared_data/datasets/dclm_9p6m_ctx1024_ds42/train/tokens.bin
/root/shared_data/datasets/dclm_9p6m_ctx1024_ds42/val/tokens.bin
```

The default data seed is `42`, so normal training reads from the preshuffled
`dclm_9p6m_ctx1024_ds42` dataset. If you request a seed without a preshuffled
shared cache, training loads the base dataset and materializes the shuffled
prefix once before training starts.

Non-Modal users only: direct local training runs do not use Modal Volumes. They
read data from `DATA_DIR` and write checkpoints to `MODEL_DIR`, both configured
through `utils.py`. If local data is missing, download the raw binary release
from `kothasuhas/dl_alchemy_seq9p6m_context1024`:

```bash
uv run python -m download_data
```

See [gpu/README.md](gpu/README.md) for the full non-Modal setup.
