"""P2: supplied LR/WD grid, LR-tuned Hyperball, dropout and ensembles."""

import numpy as np
from .common import Config, reference, runs


def supplied_wd_runs():
    """32-epoch controls: WD0.1 without tuning and WD3.2 with tuning."""
    return runs('wd_tuning')['runs']


def dropout(rate, seed=Config.seed):
    return Config(name=f'attention-p{rate:g}-s{seed}', attention_dropout=rate, seed=seed)


def hyperball(lr):
    return Config(name=f'hyperball-lr{lr:g}', optimizer='hyperball', lr=lr, wd=0.0)


def ensemble_bank(epochs=32, dropout=0.0, root=None):
    """Load the supplied no-dropout validation logits."""
    if dropout != 0:
        raise ValueError('Dropout banks are not supplied; train and evaluate your own members.')
    from .common import json_file, runs

    bank = (
        runs('ensemble_bank') if root is None else json_file('references/ensemble_bank.json', root)
    )
    members = bank['groups'][f'e{epochs}_p{dropout:g}']
    return [np.load(reference(m['validation_logits'], root), mmap_mode='r') for m in members]


def ensemble_curve(epochs=32, dropout=0.0, root=None):
    if dropout != 0:
        raise ValueError('Dropout curves are not supplied; analyze your own predictions.')
    from .common import json_file, runs

    data = (
        runs('ensemble_curves')
        if root is None
        else json_file('references/ensemble_curves.json', root)
    )
    row = dict(data['groups'][f'e{epochs}_p{dropout:g}'])
    from pathlib import Path

    path = (
        Path(__file__).with_name('references') / 'ensemble_chunks.npz'
        if root is None
        else reference(row['chunk_losses_file'], root)
    )
    with np.load(path) as arrays:
        row['chunk_losses'] = arrays[row['chunk_losses_key']].copy()
    return row


if __name__ == '__main__':
    print([dropout(p) for p in (0.05, 0.15, 0.25)])
    print([hyperball(lr) for lr in (0.003, 0.01, 0.03)])
    # P2(a)(i): python -m experiments.a6_generalization.quadratic_fit
    # P2(b): use ensemble_losses on small chunk slices of ensemble_bank().
