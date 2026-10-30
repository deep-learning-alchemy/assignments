"""Problem 1: transformer shapes, resource accounting, and public references."""
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
from model_config import LMConfig

DEPTHS = tuple(range(4, 11))
RATIOS = (16, 32, 64, 128, 256)
PUBLIC_PROJECT = 'hashimoto-group/public-alchemy-a5'
PUBLIC_URL = 'https://wandb.ai/' + PUBLIC_PROJECT
RELEASE_TAG = 'a5-capacity-v1'
EXTREMES = {'deep_skinny': (62, 192), 'shallow_wide': (1, 1280)}
SHARED_ROOT = Path(os.environ.get('A5_REFERENCE_ROOT', '/root/shared_data/a5_capacity/v1'))


def architecture(depth=8, width=512, vocab=4096, context=1024):
    head = math.gcd(width, 64)
    return LMConfig(f'd{depth}-w{width}', vocab, context, width, int(3.5*width),
                    depth, width//head, width//head, head_dim=head)


def parameters(config):
    d, w, i, v = config.num_hidden_layers, config.hidden_size, config.intermediate_size, config.vocab_size
    return 2*v*w + d*(4*w*w + 3*w*i + 2*w + 2*config.head_dim) + w


def training_flops(config, tokens=614_400_000, *, depth=None):
    """Analytic estimate including attention products, rather than wall time."""
    d, w, i, t = config.num_hidden_layers, config.hidden_size, config.intermediate_size, config.context_length
    if depth is not None:
        d = depth  # Looped models execute more blocks than they store.
    return tokens*(6*(d*(4*w*w + 3*w*i) + config.vocab_size*w) + 12*d*t*w)


def grid():
    return [architecture(d, ratio*d) for ratio in RATIOS for d in DEPTHS]


def manifest(root=SHARED_ROOT):
    path = Path(root)/'manifest.json'
    if not path.exists():
        raise FileNotFoundError(
            f'No A5 reference manifest at {path}. Mount the course shared-data volume '
            'or set A5_REFERENCE_ROOT to a local copy; see README.md for the release location.')
    return json.loads(path.read_text())


def runs(*, depth=None, width=None, seed=None, retained=False, root=SHARED_ROOT):
    return [r for r in manifest(root)['runs']
            if (depth is None or r['depth'] == depth) and (width is None or r['width'] == width)
            and (seed is None or r['seed'] == seed) and (not retained or r['checkpoint'] is not None)]


def best(depth=8, width=512, seed=42, *, root=SHARED_ROOT):
    candidates = [r for r in runs(depth=depth, width=width, seed=seed, retained=True, root=root)
                  if r['val_loss'] is not None and math.isfinite(r['val_loss'])]
    if not candidates:
        raise ValueError('No finite retained checkpoint for this architecture/seed.')
    return min(candidates, key=lambda r: r['val_loss'])


def load_history(run=None, root=SHARED_ROOT):
    """Unsampled archived metrics, without a W&B API request."""
    run = best(root=root) if run is None else run
    with gzip.open(Path(root)/run['history'], 'rt') as f:
        return json.load(f)


def load_model(run=None, root=SHARED_ROOT, device='cuda', verify=True):
    """Load a retained raw state dictionary using its manifest architecture."""
    import torch
    from modeling import AutoregressiveLM, LlamaRotaryEmbedding
    run = best(root=root) if run is None else run
    if isinstance(run, str):
        run = next(r for r in manifest(root)['runs'] if r['cell_id'] == run)
    if run['checkpoint'] is None:
        raise ValueError('This LR trial has curves but its weights were not retained.')
    path = Path(root)/run['checkpoint']
    if verify:
        digest = hashlib.sha256()
        with path.open('rb') as f:
            while block := f.read(8*1024**2):
                digest.update(block)
        if digest.hexdigest() != run['sha256']:
            raise ValueError(f'Checkpoint checksum failed: {path}')
    config = LMConfig.from_dict(run['architecture'])
    with torch.device('meta'):
        model = AutoregressiveLM(config, qk_norm=True)
    state = torch.load(path, map_location=device, weights_only=True, mmap=True)
    model.load_state_dict(state, strict=True, assign=True)
    # RoPE frequencies are nonpersistent, so the checkpoint cannot fill them.
    model.model.rotary_emb = LlamaRotaryEmbedding(config).to(device)
    return model.eval()
