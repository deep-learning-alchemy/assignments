# A6 supplied files and student experiments

Public reference project: [hashimoto-group/public-alchemy-a6](https://wandb.ai/hashimoto-group/public-alchemy-a6/overview), tags `a6-reference` and `a6-generalization-v1`.
The release contains 48 distinct training runs after reusing shared controls.
Their W&B histories are imports of the original recorded validation callbacks.

Modal uses environment `cs312-shared-data`, volume `hard-dl-dclm-v1`.
The read-only student mount is `/root/shared_data/a6_generalization/v1`.
`manifest.json` lists source IDs, W&B URLs, file sizes and SHA256 checksums.
Small validation-only reference bundles are also included here in Git.

| Supplied files | Contents | Work students can skip |
|---|---|---|
| `references/parameter_scaling.json`, `checkpoints/parameter_suite/d4.pt`–`d10.pt` | Seven selected dense recipes, histories and weights. | The parameter/LR/WD/epoch search for 1(a). |
| `references/epoch_scaling.json` | Four duration references from the same frozen parameter-sweep registry. | The motivating epoch sweep. |
| `references/lr_wd_runs.json` | The completed 16-epoch LR×WD grid. Its historical WD values are preserved; logarithmic fits use positive WD. | All training for 2(a)(i). |
| `references/wd_tuning.json` | AdamW LR .003, 32 epochs, WD 0.1 without tuning versus WD 3.2 with tuning. | Retraining the Hyperball comparison's AdamW controls. |
| `references/batch_scaling.json`, `checkpoints/batch_e32/b*.pt` | B16/64/256/1024/4096 under sqrt LR and WD scaling, 32 epochs. | The batch reference sweep for 3(b). |
| `references/ensemble_bank.json`, `references/ensemble_curves.json`, `references/ensemble_chunks.npz`, `ensemble_logits/<id>.npy` | Five no-dropout members, seeds 42–46, at the 32-epoch baseline; validation logits and per-K chunk losses. | Training the no-dropout bank for 2(b)(ii). |
| `references/schedules.json` | Five linear horizons and five endpoints of one continuous constant-LR trajectory. | Repeating the supplied schedule references for 3(a). |
| `splits/target_30k.npy`, `splits/validation.npy` | 30k training and 1k validation chunks. Fine-tuning pools are nested prefixes of the training pool. | Downloading and preparing target DNA. |
| `checkpoints/baseline_d8_e32_s42.pt` | D8 LR .003, WD 3.2, B64, seed 42, 32 epochs. | Retraining the standard control. |
| `checkpoints/{related,dclm,noise}.pt` | Three 10B-token pretrained initializations. | All pretraining for Problem 4. |
| `histories/<id>.json.gz` | Original recorded validation histories, also mirrored to W&B. | Querying W&B to reconstruct supplied curves. |

## Not supplied

- MoE/looping comparison runs, histories or checkpoints. Starter architecture
  code is supplied; students train the three-LR comparisons for 1(b–c).
- Dropout models, logits, ensemble banks or composition curves. Students train
  dropout and extra members for 2(b)(i,iii), then use `analysis.ensemble_losses`.
- Original DNA/DCLM pretraining corpora and related-DNA selection indices.
  Only the target/validation arrays and pretrained checkpoints are shared.
- Hyperball/SAM/fine-tuning results or instructor held-out metrics.

Students also train the exploratory architecture/optimizer, Hyperball LR search,
SAM comparison, and fine-tuning adaptations. Part 4(a) uses 30k at 1/4/16 passes;
part 4(b) uses 1k/3k/10k/30k at 16 passes only. Every scratch baseline otherwise
uses 32 epochs, except the fixed 16-epoch recall grid and varying schedule horizons.
