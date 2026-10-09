# Assignment 3 Starters

Complete the repository's [setup guide](../../README.md) first. These launchers
use your environment and W&B project from `utils.py`; training still uses the
default recipe unless a starter explicitly overrides it. Run commands from
the repository root. Review each file before launching its sweep.

## Problems

- [p1_mode_connectivity.py](p1_mode_connectivity.py): train a shared prefix,
  fork model and optimizer state with different batch orders, then measure
  interpolation and ensemble losses.
- [p2_river_valley.py](p2_river_valley.py): compare schedules, evaluation-only
  parameter EMAs, temporary LR dips, and horizon-free schedules.
- [p3_edge_of_stability.py](p3_edge_of_stability.py): full-batch and minibatch
  starters with raw or Adam-preconditioned sharpness logging.
- [p4_inside_the_hessian.py](p4_inside_the_hessian.py): measure the supplied
  checkpoints, rescale parameters, and continue with projected gradients.

For Problem 1, run the stages in order and wait for each to finish before
starting the next:

```bash
uv run python -m experiments.a3_optimization.p1_mode_connectivity prefix
uv run python -m experiments.a3_optimization.p1_mode_connectivity branches
uv run python -m experiments.a3_optimization.p1_mode_connectivity measure
```

Problems 2 and 3 submit detached training jobs:

```bash
uv run python -m experiments.a3_optimization.p2_river_valley
uv run python -m experiments.a3_optimization.p3_edge_of_stability
```

Problem 4 has separate measurement/training stages, listed at the top of its
starter. For example, `a` measures the provided AdamW checkpoints; after those
jobs finish, `fetch` downloads the results:

```bash
uv run python -m experiments.a3_optimization.p4_inside_the_hessian a
uv run python -m experiments.a3_optimization.p4_inside_the_hessian fetch
```

Hessian probes are substantially more expensive than ordinary evaluation. The
starters document approximate costs. Problem 4 writes results to your writable
Modal Volume and fetches them into `results/` in this directory. Those outputs
and plots are ignored by Git and excluded from Modal source uploads.

## Training Controls

- `fork_from_run` and `fork_from_step`: load another run's kept checkpoint,
  including optimizer and RNG state, while keeping the new config's optimizer
  hyperparameters. Keep the optimizer type and architecture compatible.
- `batch_order_seed`: permute the order of training microbatches independently
  of the dataset seed.
- `stop_at_step`: stop early without shortening the LR schedule's horizon.
- `keep_checkpoint_steps`: retain the numbered checkpoints needed by later
  branches or measurements. `init_checkpoint_path` still loads weights only.
- `ema_decays`: track parameter EMAs for evaluation, logging
  `val_ema{decay}_loss`. The ordinary training weights are unchanged, and EMA
  state is saved in training checkpoints for resume.
- `warmup_steps`: specify warmup in optimizer updates instead of a fraction.
- `lr_schedule`: also accepts `invsqrt`, `sgdr2000` (a 2,000-update restart
  cycle), and `step750x0.5` (halve the LR after 750 updates, without warmup).
  `min_lr_ratio` sets the SGDR floor; `lr_dip` adds a temporary decay/rewarmup.
- `embedding_init_scale`: multiply the initialized input embedding weights.

Sharpness logging uses FP32 model weights (`precision="mp"` or `"fp32"`) and
requires `wandb_online=True`, like other custom loggers. The batch probe requires
one microbatch. The held-out probe uses 64 rows outside the default 600,000-row
training subset and rejects a training subset that overlaps it.

## Provided Checkpoints

Problem 4's `adamw` and `sgdm` aliases read from the existing shared data Volume
`hard-dl-dclm-v1` in `cs312-shared-data`, mounted read-only at:

```text
/root/shared_data/a3_hessian/runs/
```

Each run contains metadata and model/optimizer checkpoints at steps
`3, 15, 61, 153, 305, 763, 1526, 3052, 4578, 6104, 7630, 9375`.
Step 0 is reconstructed from the recorded model seed. You do not need to copy
these files to your personal Volume or modify the shared environment.

Non-Modal users only: set `A3_HESSIAN_RUNS` to a local directory containing the
two provided run directories, then add `--local` to Problem 4 measurement
commands. Training configs also work with `train(config)` on a local GPU; see
[gpu/README.md](../../gpu/README.md).

The [personal-workspace setup](../../README_MODAL_OWN_WORKSPACE.md) downloads
the dataset, not these course checkpoints. Outside the course workspace,
Problem 4 requires checkpoint copies from the staff or checkpoints you train
yourself. The loader also accepts your own run names or paths.

## Checks

The numerical and small training tests use synthetic data on CPU, without
launching Modal jobs or contacting W&B:

```bash
uv run python -m unittest discover -s tests -p 'test_a3_*.py' -v
```
