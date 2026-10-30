# Assignment 5: capacity

Start with the repository [setup guide](../../README.md). Run commands from the
repository root, using your configured data, W&B project, and Modal environment.

| Problem | Implementation |
|---|---|
| 1: depth and width | [p1_depth_width.py](p1_depth_width.py): shapes, parameter/FLOP estimates, reference loaders |
| 2: readout bottlenecks | [p2_readout.py](p2_readout.py): factorized heads, MoS, Sigsoftmax, CPLM, SpamLang |
| 3: mixture of experts | [p3_moe.py](p3_moe.py): four SwiGLU experts and editable routing |
| 4: looping | [p4_looping.py](p4_looping.py): prelude, shared core, coda, optional boundary map, mid-run sharing switch |
| 5: prediction question | Write a compact experiment using the model and launch helpers above |

These classes reuse the native transformer and trainer. The algorithm equations
are in the problem files; [sweeps.py](sweeps.py) defines editable configurations
and launch helpers. [diagnostics.py](diagnostics.py) contains the measurements.

## Supplied depth/width runs

Completed depth/width runs and retained checkpoints are published at the
locations below. The release is updated as additional trials finish; its
manifest records the current counts and publication time.

- Public W&B project: [hashimoto-group/public-alchemy-a5](https://wandb.ai/hashimoto-group/public-alchemy-a5/overview), tag `a5-capacity-v1`.
- Modal environment: `cs312-shared-data`, volume: `hard-dl-dclm-v1`.
- Read-only mount: `/root/shared_data/a5_capacity/v1`; its `manifest.json` lists
  public run links, architectures, learning rates, losses, metric histories,
  and retained checkpoint paths/checksums.

The ratio grid is depths 4–10 with widths `W = ratio * D`, for
`ratio in (16, 32, 64, 128, 256)`: all 35 architectures are represented.
Additional shapes include the two parameter-matched extremes for Problem 1(b):

| Shape | Total parameters | Reference LR |
|---|---:|---:|
| Default: 8 layers × width 512 | 34,612,736 | .003 |
| Deep and skinny: 62 layers × width 192 | 34,745,536 | .003 |
| Shallow and wide: 1 layer × width 1280 | 34,246,528 | .003 |

All use 600,000 sequences (614.4M tokens), with embeddings and the untied head
included in parameter counts. The extremes each have one LR and seed42.
Use the supplied runs for Problem 1; no new training is expected. The release
includes all completed LR trials, **not a complete five-LR sweep at every
shape**. We are still training the full grid for all of the width/depth
combinations. Compare tuning effort and fixed-LR controls when interpreting small gaps.

Histories include LR trials whose checkpoints were not retained; check
`row['checkpoint']` before attempting to load weights. `best()` selects the
best finite **retained** checkpoint, which can differ from the best loss over
all LR histories. Use `runs()` when comparing all trials.

```python
from experiments.a5_capacity import p1_depth_width as refs

rows = refs.runs(depth=8, width=512, seed=42)
row = refs.best(8, 512)
history = refs.load_history(row)
model = refs.load_model(row)

# Parameter-matched extremes and the d8 trial at the same LR.
skinny = refs.best(*refs.EXTREMES['deep_skinny'])
wide = refs.best(*refs.EXTREMES['shallow_wide'])
control = next(r for r in rows if r['lr'] == .003)
skinny_history = refs.load_history(skinny)
wide_model = refs.load_model(wide)

release = refs.manifest()
print(release['run_count'], release['checkpoint_count'], release['published_at'])
```

These calls read the mounted manifest and files without a W&B API request.
For local copies, pass `root=...` to the loaders or set `A5_REFERENCE_ROOT`
before importing. The loaders report a missing manifest explicitly.
The Modal files require access to the course workspace; W&B curves are publicly
readable without that access. Record the manifest publication time and run
cell IDs in your report. Checkpoint paths remain available across updates;
`snapshots/` contains earlier manifest versions.
Start a fresh Modal job to see a newer snapshot, or call
`modal_utils.shared_data_volume.reload()` after closing open reference files.
History `optimizer_step` is zero-based; `(optimizer_step + 1) * 64 * 1024`
gives the number of training positions processed.
If a logging outage required recovery, the run's `history_recovery` entry
identifies recorded stdout losses (rounded to six decimals) and exact saved
validation measurements. No missing losses are interpolated.

## Problem 1(a): plot loss versus parameters

Run this explicit plotting example from the repository root:

```bash
uv run python -m experiments.a5_capacity.p1_depth_width
```

It fetches only `manifest.json` from the shared Modal volume using a CPU worker,
then saves **`plots/a5_capacity/loss_vs_parameters.png` locally**. It needs your
configured Modal environment and course-workspace access; no GPU, W&B API call,
checkpoint download, or new training is needed.

The plot has one color per width/depth ratio `W/D` and a logarithmic parameter
axis. Faint points show all available LR trials; connected circles select the
lowest final validation loss at each of the 35 shapes, using seed 42 and the
same 614.4M-position training budget. Counts include embeddings and the untied
output head. Selection includes trials whose checkpoints were pruned, unlike
`best()`, which selects retained weights. The number of LR trials differs
between shapes; these are the best **available** losses.

With a local copy or mounted reference directory, use:

```bash
A5_REFERENCE_ROOT=/path/to/a5_capacity/v1 uv run python -m experiments.a5_capacity.p1_depth_width --local
```

The editable plotting function is in `p1_depth_width.py`; its core comparison is
`row['parameters'] / 1e6` against `row['val_loss']`. You can also call it directly:

```python
from experiments.a5_capacity import p1_depth_width as refs

release = refs.manifest(root="/path/to/a5_capacity/v1")
refs.plot_loss_vs_parameters(release, output="plots/my_depth_width.png")
```

For the worksheet's compute comparison, `training_flops()` supplies the FLOP
estimate for each run's architecture. Parameter count and compute are separate
axes; this example shows the parameter comparison.

## Run a new experiment

Edit the call at the bottom of a problem file, then run its module, for example:

```bash
uv run python -m experiments.a5_capacity.p3_moe
```

Or select a variant directly:

```python
from experiments.a5_capacity.sweeps import config, launch_one

launch_one('moe-learned', .003, dry_run=True)
launch_one('moe-learned', .003)              # Modal H100.
launch_one('cplm-r64', .003, local=True)     # Local GPU.
cfg = config('loop-r6', model_builder_kwargs=dict(
    passes=6, boundary_norm=True, recall=.5), run_name='a5-my-loop')
```

`VARIANTS` lists the available names. MoE interventions have separate presets;
compose them by replacing `model_builder_kwargs` with a complete options dict.
Rank presets are `softmax-r{32,64,128,256}`, and the same suffixes on `mos`,
`sigsoftmax`, and `cplm`. Unsuffixed methods use full-width vocabulary heads.
Use distinct run names when changing the architecture, recipe, or seed.
New runs use your own configured W&B project.

The recipe inherits `TrainConfig`: d8 width 512, vocabulary 4096, context 1024,
600,000 sequences (614.4M input tokens), batch 64, AdamW, linear LR decay,
1% warmup, weight decay .1, clipping at 1, and model seed 42. A5 uses four
microbatches for memory; the effective batch stays 64. With one microbatch,
the native trainer compiles the loss using mode `default`. MoE also compiles
its small expert computation on CUDA. Run `sweeps.train(cfg)` for local custom
configs: it updates quantile biases once per optimizer step, after accumulating
all microbatches. No shared trainer changes are required.

For SpamLang, use the same launch helper:

```python
from experiments.a5_capacity.p2_readout import training_config
from experiments.a5_capacity.sweeps import launch, train

cfg = training_config(vocab=4096, width=64, lr=.001)
launch(cfg)  # Or train(cfg) locally.
```

It draws independent uniform symbols for each sequence, repeats each 64 times,
and uses one layer, four heads, 102,400 training sequences, 2,048 independent
validation sequences, and 800 updates at batch 128. Modal regenerates these
small datasets on the worker.

## Measure checkpoints

This example loads the supplied default d8 and
measures held-out loss, projected gradient energy, and signals per block:

```bash
uv run python -m experiments.a5_capacity.measure_default
```

Edit its `DEPTH`/`WIDTH` constants for another reference. For your saved models,
use the native `model_io.load_model` and `diagnostics.measure_model(model, tokens)`.
Detailed probes run separately from training on matched held-out positions.

- Readout: `gradient_energy` measures the fraction of `p - one_hot(target)`
  energy outside the effective head's column space. This linear-head diagnostic
  applies to ordinary softmax, including factorized heads.
- MoE: selected-load CV, normalized probability entropy, and mean largest
  routing probability. `router_layers` retains each layer; `routers` reports
  maximum layer CV and mean layer entropy/confidence. Quantile routing uses
  unbiased probabilities. Random routing has load statistics only.
- Looping: activation RMS, state-gradient RMS, and relative update per core
  pass, with separate prelude/coda entries. Internal matrices and norm gains
  are shared. The optional boundary RMSNorm has no learned parameters and is
  followed by `recall * prelude_output`, including before the coda by default.

`loop-r{1,2,3,6}` keeps eight block applications. `loop-1block-rR` fixes three
distinct blocks and executes `R+2` applications. Set `model.eval_passes` only
after `model.eval()` to test another inference loop count. Untied models can only
use the independent blocks they contain.

For Problem 4(e), `loop-to-unique` and `unique-to-loop` switch after update
4,688 of 9,375; `loop-reset` and `unique-reset` are matched Adam-reset controls.
All four execute eight blocks. Untying clones values into six independent core
blocks; tying averages their parameters (set `merge="first"` to retain the first).
The prelude, coda and boundary map stay in place. Adam moments reset across the
whole model, while LR, data order and token count continue. Use `sweeps.train` or
`launch_one`, one visible GPU, the default four microbatches, and no EMA. Model
and optimizer checkpoints preserve sharing, the update counter and the switch
stage when resuming. Final native model checkpoints also restore their sharing.
The launch helper measures native validation loss immediately before/after the
switch, logs `switch/before_val_loss` and `switch/after_val_loss` to W&B, and saves
both values in `switch_metrics.json` alongside your checkpoint.

```python
launch_one("loop-to-unique", .003)
launch_one("unique-to-loop", .003)
launch_one("loop-reset", .003)
launch_one("unique-reset", .003)
```

For a shorter training budget, override `model_builder_kwargs` with the complete
options dictionary, including a midpoint `switch_step` for that budget.
`model.set_sharing(False)` or `model.set_sharing(True, merge="mean")` also performs
the conversion explicitly; rebuild the optimizer after manual conversion.

Synthetic CPU checks require no downloaded data or cloud connection:

```bash
uv run python -m unittest discover -s tests -p 'test_a5_*.py' -v
```
