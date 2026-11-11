"""P1: analyze the supplied parameter suite; compare the A5 architectures."""

from .common import Config, runs
from .analysis import power_law, predict


def comparisons(architecture):
    return [
        Config(name=f'{architecture}-lr{lr:g}', architecture=architecture, lr=lr)
        for lr in (0.001, 0.003, 0.01)
    ]


def plot_parameters(output='plots/a6_generalization/parameters.png'):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path

    plt.rcParams.update(
        {
            'font.family': ['P052', 'DejaVu Serif'],
            'axes.spines.top': False,
            'axes.spines.right': False,
        }
    )
    bundle = runs('parameter_scaling')
    rows = bundle['runs']
    x = np.array([r['parameters'] / 1e6 for r in rows])
    y = np.array([r['validation_nll'] for r in rows])
    fit = power_law(x, y)
    xx = np.geomspace(x.min(), x.max(), 200)
    fig, ax = plt.subplots(figsize=(6, 4.7))
    ax.plot(x, y, 's', color='#7030A0', label='Supplied single-model runs')
    ax.plot(
        xx,
        predict(fit, xx),
        '--',
        color='#7030A0',
        label=f'{fit["offset"]:.4f}+{fit["amplitude"]:.4f} N^(-{fit["exponent"]:.2f})',
    )
    baseline = bundle.get('baseline')
    if baseline:
        ax.scatter(
            baseline['parameters'] / 1e6,
            baseline['validation_nll'],
            marker='*',
            color='#e74c3c',
            s=85,
            label='Default D8 · 32 epochs',
        )
    ax.set(xscale='log', xlabel='Parameters (millions)', ylabel='Validation NLL')
    ax.grid(True, ls=':', alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fit


def load_model(row, root=None):
    import torch
    from .common import reference
    from .models import build

    config = Config(depth=row['depth'], width=row['width'], seed=row['seed'])
    model = build(config)
    model.load_state_dict(
        torch.load(reference(row['checkpoint'], root), map_location='cpu', weights_only=False)[
            'model'
        ]
    )
    return model.eval()


if __name__ == '__main__':
    print(plot_parameters())
    # Edit a config, then call training.train(cfg) locally or training.launch(cfg).
    print(comparisons('moe'))
    print(comparisons('loop'))
