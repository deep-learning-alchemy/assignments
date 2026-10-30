# Assignment 5: capacity

Start with the repository [setup guide](../../README.md). Run commands from the
repository root, using your configured data, W&B project, and Modal environment.

| Problem | Implementation |
|---|---|
| 1: depth and width | [p1_depth_width.py](p1_depth_width.py): shapes, parameter/FLOP estimates, reference loaders |
| 2: readout bottlenecks | [p2_readout.py](p2_readout.py): factorized heads, MoS, Sigsoftmax, CPLM, SpamLang |
| 3: mixture of experts | [p3_moe.py](p3_moe.py): four SwiGLU experts and editable routing |
| 4: looping | [p4_looping.py](p4_looping.py): prelude, shared core, coda, optional boundary map |

These classes reuse the native transformer and trainer. The algorithm equations
are in the problem files; [sweeps.py](sweeps.py) defines editable configurations
and launch helpers. [diagnostics.py](diagnostics.py) contains the measurements.

## Supplied depth/width runs

**TODO: publish the completed depth/width grid and finalize its manifest.** The
locations below are the planned release destinations. Any existing preliminary
contents may be incomplete; they are not the final grid.

- Public W&B project: [hashimoto-group/public-alchemy-a5](https://wandb.ai/hashimoto-group/public-alchemy-a5/overview).
- Modal environment: `cs312-shared-data`, volume: `hard-dl-dclm-v1`.
- Read-only mount: `/root/shared_data/a5_capacity/v1`; its `manifest.json` will
  list public run links, architectures, learning rates, losses, metric histories,
  and retained checkpoint paths/checksums.

The intended grid is depths 4–10 with widths `W = ratio * D`, for
`ratio in (16, 32, 64, 128, 256)`: 35 architectures. Use the supplied LR trials
for Problem 1 once published. New training is not expected for that problem.
Histories will include LR trials whose checkpoints were not retained; check
`row['checkpoint']` before attempting to load weights. `best()` selects the
best finite **retained** checkpoint, which can differ from the best loss over
all LR histories. Use `runs()` when comparing all trials.

```python
from experiments.a5_capacity import p1_depth_width as refs

rows = refs.runs(depth=8, width=512, seed=42)
row = refs.best(8, 512)
history = refs.load_history(row)
model = refs.load_model(row)
```

These calls read the mounted manifest and files without a W&B API request.
For local copies, pass `root=...` to the loaders or set `A5_REFERENCE_ROOT`
before importing. The loaders report a missing manifest explicitly.

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

After reference publication, this example loads the supplied default d8 and
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
after `model.eval()` to test another inference loop count.

Synthetic CPU checks require no downloaded data or cloud connection:

```bash
uv run python -m unittest discover -s tests -p 'test_a5_*.py' -v
```
