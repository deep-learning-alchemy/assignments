# Assignment 6: generalization

Use the repository setup guide, then run commands from the repository root.
The handout is distributed separately. The report is due **November 19, 2026, at 1:00pm Pacific**, per the
[course schedule](https://deep-learning-alchemy.github.io/). Submit it on Gradescope.

Reference histories are in [hashimoto-group/public-alchemy-a6](https://wandb.ai/hashimoto-group/public-alchemy-a6/overview).
The shared Modal environment is `cs312-shared-data`, volume `hard-dl-dclm-v1`.

The setting is 30,000 DNA chunks of length 1,024 (30.72M tokens including BOS),
with a separate 1,000-chunk validation set. The vocabulary has 19 tokens.
The baseline is d8, width 512, QK normalization, AdamW (.9,.95), LR .003, WD 3.2,
batch 64, microbatch 8, clipping 1, 32 epochs, and 4,800 sequence exposures of warmup
followed by linear LR decay. Only Problem 3(a) compares constant LR.
**The default baseline is the sweep-selected 32-epoch D8.**
The supplied LR×WD recall grid fixes 16 epochs. `references/wd_tuning.json` supplies separate
32-epoch AdamW controls: WD 0.1 without WD tuning and WD 3.2 with WD tuning. Fine-tuning separately
uses 1/4/16 passes on 30k chunks in 4(a), and 16 passes across sample counts in 4(b). Use validation NLL for both selection and reporting; there is no student test split.

| Problem | Starter | Supplied work / student experiment |
|---|---|---|
| 1: architecture | `p1_architecture.py` | Analyze the supplied tuned d4–d10 suite. Train the three-LR A5 MoE/loop comparisons and explore a fixed parameter budget. |
| 2: regularization | `p2_regularization.py`, `quadratic_fit.py` | Fit the supplied LR×WD grid. Tune Hyperball LR. Train attention dropout and its ensembles; reuse the supplied no-dropout member predictions. |
| 3: optimization | `p3_optimization.py` | Reuse supplied schedule/batch references, test selected horizons, compare SAM with AdamW, and explore another optimizer. |
| 4: fine-tuning | `p4_finetuning.py` | Start from released weights: 4(a) uses 30k at 1/4/16 passes; 4(b) uses 1k/3k/10k/30k at 16 passes; study adaptation in 4(c). |

Read [RELEASE_PLAN.md](RELEASE_PLAN.md) for exact supplied files and the experiments students train. Small validation-only reference JSONs are included
in `references/`. Large datasets/checkpoints/predictions are intentionally not
checked into Git. Set `A6_REFERENCE_ROOT` for a local copy; on Modal the existing course setup
mounts the references read-only at `/root/shared_data/a6_generalization/v1`.
`common.manifest()` reads the published file/checksum manifest. Original
pretraining datasets, architecture runs and dropout banks are not supplied.

## Analyze supplied data without a GPU

```bash
uv run python -m experiments.a6_generalization.p1_architecture
uv run python -m experiments.a6_generalization.quadratic_fit --output plots/a6_generalization/lr_wd.png
```

`analysis.power_law(x,y)` fits `C+A*x**(-alpha)`. Inspect the measured points and
fit sensitivity before interpreting an extrapolated floor. `ensemble_losses`
averages probabilities over every member subset; K1 is mean single-model NLL.
`p2_regularization.ensemble_curve()` reads the supplied no-dropout per-K chunk
losses directly from Git without loading logits or contacting Modal.
`chunk_bootstrap` resamples the same chunks across methods/K; it does not measure
uncertainty from additional training seeds.

## Train one experiment

Use the editable dataclass, like the A4/A5 experiment configurations:

```python
from experiments.a6_generalization.common import Config
from experiments.a6_generalization.training import train, launch

cfg = Config(name='my-baseline')
print(cfg)  # Review before allocating a GPU.
launch(cfg)  # Detached Modal H100, configured course environment.
# train(cfg)           # Your local CUDA GPU instead.
```

Nothing launches on import. Running a problem module prints example configs;
P1 also plots its small supplied reference bundle. Start with one run and edit
configurations rather than launching every cross product at once. Use a distinct
`name` per recipe/seed. Local paths use `A6_REFERENCE_ROOT`; the Modal
mount is `/root/shared_data/a6_generalization/v1`.

The small A6 loop in `training.py` uses the course transformer and A5 model
classes. It shows microbatch accumulation, causal NLL, validation, the warmup and
linear/constant schedules, and SAM's perturb/update directly. Histories and final
weights are written under your model directory (`a6/<name>`). It is a single-GPU
starter with final checkpoints; use the repository's native trainer if extending
to distributed/resumable infrastructure.

Attention dropout changes attention probabilities only and is disabled during
validation. Hyperball projects each Linear weight tensor using its Frobenius norm
and uses plain Adam for embeddings/norms, at the course fallback-LR ratio. WD 0
for Hyperball is deliberate. Other AdamW runs use the existing course decay mask.
For sharpness, call `p3_optimization.directional_sharpness(model, train_rows[:4])`
on the same four training chunks for every model.

## Fine-tuning

`models.build` loads the DNA/noise checkpoints in full. For DCLM it loads the
transformer body into newly initialized DNA embeddings/head. Both parts 4(a–b) update all parameters, starting independently from the original
pretrain. Part 4(a) compares 1/4/16 passes on 30k chunks and tunes fine-tuning LR.
Part 4(b) fixes 16 passes and compares 1k/3k/10k/30k chunks on one plot;
`p4_finetuning.sample_scaling()` constructs those four configurations.

Part 4(c) focuses on related DNA. `adaptation` accepts `full`,
`toward_initialization` (decoupled decay toward pretrained weights), `frozen_head`
(output head fixed, embedding/body trainable), or `frozen_body` (embedding/head
trainable). `full` and `toward_initialization` update all parameters. No pretraining
run is required: the three 10B-token initializations are supplied; their original pretraining datasets are not shared yet.

Synthetic CPU checks, with no Modal or W&B access:

```bash
uv run python -m unittest discover -s tests -p 'test_a6_*.py' -v
```
