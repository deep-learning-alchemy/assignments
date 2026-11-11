"""Editable recipe and read-only course files; no downloads on import."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import numpy as np

REFERENCE_ROOT = Path(os.environ.get('A6_REFERENCE_ROOT', '/root/shared_data/a6_generalization/v1'))
SAMPLE_COUNTS = (1000, 3000, 10000, 30000)
PASSES = (1, 4, 16)
VOCAB = (
    'A',
    'C',
    'G',
    'T',
    'N',
    'R',
    'Y',
    'S',
    'W',
    'K',
    'M',
    'B',
    'D',
    'H',
    'V',
    '<bos>',
    '<eos>',
    'U',
    'X',
)


@dataclass(frozen=True)
class Config:
    name: str = 'baseline'
    architecture: str = 'dense'
    lr: float = 0.003
    wd: float = 3.2
    epochs: int = 32
    ntrain: int = 30000
    batch: int = 64
    micro_batch: int = 8
    seed: int = 42
    data_seed: int = 42
    schedule: str = 'linear'
    attention_dropout: float = 0.0
    optimizer: str = 'adamw'
    rho: float = 0.05
    initialization: str = 'random'
    adaptation: str = 'full'
    depth: int = 8
    width: int = 512
    context: int = 1024


def reference(path, root=None):
    path = Path(root or REFERENCE_ROOT) / path
    if not path.is_file():
        raise FileNotFoundError(
            f'{path} is not staged. See RELEASE_PLAN.md; set A6_REFERENCE_ROOT for a local copy.'
        )
    return path


def json_file(path, root=None):
    return json.loads(reference(path, root).read_text())


def manifest(root=None):
    return json_file('manifest.json', root)


def data(config, root=None):
    train = np.load(reference('splits/target_30k.npy', root), mmap_mode='r')
    validation = np.load(reference('splits/validation.npy', root), mmap_mode='r')
    if not 0 < config.ntrain <= min(30000, len(train)):
        raise ValueError('Use at most the released 30k training chunks.')
    return train[: config.ntrain, : config.context], validation[:, : config.context]


def runs(name):
    """Small supplied run bundles are tracked here; large assets use the mount."""
    path = Path(__file__).with_name('references') / (name + '.json')
    bundle = (
        json.loads(path.read_text())
        if path.is_file()
        else json_file('references/' + name + '.json')
    )
    if bundle.get('ready') is False:
        raise RuntimeError(
            f'{name}: reference runs are incomplete; see references/READINESS.json.'
        )
    return bundle


def nucleotide_noise(indices, seed=42):
    """Exact pretraining control: per-sequence IID A/C/G/T, BOS at position0."""
    rows = []
    for index in indices:
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(index)]))
        row = rng.integers(0, 4, size=1024, dtype=np.uint8)
        row[0] = 15
        rows.append(row)
    return np.stack(rows) if rows else np.empty((0, 1024), dtype=np.uint8)
