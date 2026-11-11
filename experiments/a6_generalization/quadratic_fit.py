"""Quadratic-fit helpers for Problem 2(a)(i).

Requires numpy and matplotlib. From this directory, run:
    python quadratic_fit.py --output lr_wd.png

The supplied JSON contains final validation losses, configurations, and
validation histories. Zero-WD controls are retained in the data; logarithmic
fits and the example plots use positive WD only. Interpret the measured
points and the fitted surface separately.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

COLORS = ('#5178D5', '#7030A0', '#2B8D6E')


def features(u, v):
    u, v = np.broadcast_arrays(u, v)
    return np.stack((np.ones_like(u), u, v, u * u, u * v, v * v), axis=-1)


@dataclass
class QuadraticFit:
    coefficients: np.ndarray
    log_centers: np.ndarray
    rmse: float

    def predict(self, lr, wd):
        """Evaluate a+b*u+c*v+d*u**2+e*u*v+f*v**2 in centered log space."""
        lr, wd = np.asarray(lr, dtype=float), np.asarray(wd, dtype=float)
        if np.any(lr <= 0) or np.any(wd <= 0):
            raise ValueError('Prediction requires positive LR and WD.')
        return (
            features(np.log(lr) - self.log_centers[0], np.log(wd) - self.log_centers[1])
            @ self.coefficients
        )


def fit_quadratic(lr, wd, loss):
    """Fit the six coefficients by least squares, excluding zero-WD controls."""
    lr, wd, loss = [np.asarray(x, dtype=float) for x in (lr, wd, loss)]
    if lr.ndim != 1 or lr.shape != wd.shape or lr.shape != loss.shape:
        raise ValueError('LR, WD and loss must be matching one-dimensional arrays.')
    if not all(np.isfinite(x).all() for x in (lr, wd, loss)) or np.any(lr <= 0) or np.any(wd < 0):
        raise ValueError('Use finite losses, positive LRs and nonnegative WDs.')
    mask = wd > 0
    lr, wd, loss = lr[mask], wd[mask], loss[mask]
    if len(loss) < 6:
        raise ValueError('At least six positive-WD measurements are required.')
    logs = np.column_stack((np.log(lr), np.log(wd)))
    centers = logs.mean(axis=0)
    design = features(*(logs - centers).T)
    coefficients, _, rank, _ = np.linalg.lstsq(design, loss, rcond=None)
    if rank < 6:
        raise ValueError('The LR/WD grid cannot identify all six quadratic coefficients.')
    return QuadraticFit(
        coefficients, centers, float(np.sqrt(np.mean((design @ coefficients - loss) ** 2)))
    )


def plot_example(runs, output):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import NullFormatter

    plt.rcParams.update(
        {
            'font.family': ['P052', 'DejaVu Serif'],
            'font.size': 11,
            'axes.spines.top': False,
            'axes.spines.right': False,
        }
    )
    positive = [r for r in runs if r['wd'] > 0]
    lr, wd, loss = [
        np.array([r[name] for r in positive]) for name in ('lr', 'wd', 'validation_nll')
    ]
    fit = fit_quadratic(lr, wd, loss)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    xx, yy = np.meshgrid(
        np.geomspace(lr.min(), lr.max(), 120), np.geomspace(wd.min(), wd.max(), 120)
    )
    cmap = LinearSegmentedColormap.from_list('course', ['#8CD9FF', '#7030A0'])
    image = axes[0].contourf(xx, yy, fit.predict(xx, yy), levels=16, cmap=cmap)
    axes[0].contour(xx, yy, fit.predict(xx, yy), levels=10, colors='white', linewidths=0.5)
    axes[0].scatter(lr, wd, color='black', s=22, zorder=5)
    axes[0].set(
        xscale='log',
        yscale='log',
        xlabel='Peak LR',
        ylabel='Weight decay',
        title='Quadratic fit in log LR and log WD',
    )
    axes[0].set_xticks(sorted(set(lr)), [f'{v:g}' for v in sorted(set(lr))])
    axes[0].set_yticks(sorted(set(wd)), [f'{v:g}' for v in sorted(set(wd))])
    axes[0].yaxis.set_minor_formatter(NullFormatter())
    fig.colorbar(image, ax=axes[0], label='Fitted validation NLL')
    for i, value in enumerate(sorted(set(lr))):
        rows = sorted([r for r in positive if r['lr'] == value], key=lambda r: r['wd'])
        axes[1].plot(
            [r['lr'] * r['wd'] for r in rows],
            [r['validation_nll'] for r in rows],
            '-o',
            color=COLORS[i % len(COLORS)],
            label=f'LR {value:g}',
            lw=1.7,
            ms=5,
        )
    axes[1].set(
        xscale='log',
        xlabel='Peak LR × weight decay',
        ylabel='Validation NLL',
        title='Measured loss versus LR × WD',
    )
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(True, which='major', linestyle=':', alpha=0.25)
        ax.set_axisbelow(True)
        ax.xaxis.set_minor_formatter(NullFormatter())
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'runs',
        nargs='?',
        type=Path,
        default=Path(__file__).with_name('references') / 'lr_wd_runs.json',
    )
    parser.add_argument('--output', type=Path, default=Path('lr_wd.png'))
    args = parser.parse_args()
    bundle = json.loads(args.runs.read_text())
    fit = plot_example(bundle['runs'], args.output)
    print(f'Loaded {len(bundle["runs"])} completed runs at E={bundle["setting"]["epochs"]}.')
    print('Coefficients [a,b,c,d,e,f]:', fit.coefficients)
    print(f'Quadratic-fit RMSE: {fit.rmse:.6f}; saved {args.output}')


if __name__ == '__main__':
    main()
