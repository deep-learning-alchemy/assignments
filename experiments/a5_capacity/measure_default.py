"""Measure a supplied dense checkpoint on held-out tokens, without training."""
import json
from pathlib import Path
import torch
from . import p1_depth_width as references
from .diagnostics import measure_model, signal_profile

DEPTH, WIDTH = 8, 512
OUTPUT = Path('results/a5_capacity/reference_d8.json')


def measure(depth=DEPTH, width=WIDTH, root=references.SHARED_ROOT, device='cuda'):
    from data import load_token_dataset, dclm_val_dataset
    run = references.best(depth, width, root=root)
    model = references.load_model(run, root=root, device=device)
    validation = load_token_dataset(dclm_val_dataset(), role='val', data_seed=42)
    tokens = torch.as_tensor(validation[:8]['input_ids'].astype('int64'), device=device)
    result = measure_model(model, tokens[:, :128])
    result.update(reference=run['url'], cell_id=run['cell_id'])
    result['signals'] = signal_profile(model, tokens[:2, :128])
    return result


if __name__ == '__main__':
    import modal
    from modal_utils import app, build_image, VOLUME_MOUNTS, MODAL_ENVIRONMENT
    remote = app.function(image=build_image(), volumes=VOLUME_MOUNTS, gpu='H100',
                          cpu=8, memory=32768, timeout=3600)(measure)
    with modal.enable_output(), app.run(environment_name=MODAL_ENVIRONMENT):
        result = remote.remote()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(OUTPUT)
