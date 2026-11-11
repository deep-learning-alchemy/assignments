"""Small NumPy helpers for parameter, ensemble, and transfer scaling laws."""

import itertools
import numpy as np


def power_law(x, y):
    """Descriptive L(x)=C+A*x**(-alpha); short curves may not identify C."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or x.shape != y.shape or np.any(x <= 0) or not np.isfinite(y).all():
        raise ValueError('Provide at least three finite losses and positive x values.')
    best = None
    for exponent in np.geomspace(0.02, 4.0, 400):
        basis = x**-exponent
        a, c = np.linalg.lstsq(np.column_stack((basis, np.ones_like(x))), y, rcond=None)[0]
        if a < 0 or c < 0 or c > y.min():
            continue
        error = np.mean((c + a * basis - y) ** 2)
        if best is None or error < best[0]:
            best = (error, float(a), float(exponent), float(c))
    if best is None:
        raise ValueError('No decreasing positive-offset fit; inspect the measured curve.')
    return dict(amplitude=best[1], exponent=best[2], offset=best[3])


def predict(fit, x):
    return fit['offset'] + fit['amplitude'] * np.asarray(x, dtype=float) ** -fit['exponent']


def ensemble_losses(logits, labels):
    """Mean chunk NLL over ALL subsets, one row per member count K.

    logits: list of (chunks, positions, vocabulary) arrays; labels: next tokens.
    Use small chunk slices of memory-mapped arrays to bound memory.
    """
    probabilities = []
    for z in logits:
        z = np.asarray(z, dtype=np.float64)
        z = z - z.max(-1, keepdims=True)
        p = np.exp(z)
        probabilities.append(p / p.sum(-1, keepdims=True))
    losses = []
    for k in range(1, len(logits) + 1):
        members = []
        for ids in itertools.combinations(range(len(logits)), k):
            p = np.mean([probabilities[i] for i in ids], axis=0)
            correct = np.take_along_axis(p, labels[..., None], axis=-1)[..., 0]
            members.append(-np.log(np.maximum(correct, 1e-30)).mean(-1))
        losses.append(np.mean(members, axis=0))
    return np.asarray(losses)


def chunk_bootstrap(chunks, repeats=1000, seed=42):
    """Resample columns together across K/methods; seed uncertainty is separate."""
    a = np.asarray(chunks)
    indices = np.random.default_rng(seed).integers(0, a.shape[-1], (repeats, a.shape[-1]))
    return a[..., indices].mean(-1)
