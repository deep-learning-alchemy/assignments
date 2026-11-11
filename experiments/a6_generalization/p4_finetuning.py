"""P4(a): 30k at 1/4/16 passes. P4(b): four target pools at 16 passes."""

from .common import Config, SAMPLE_COUNTS, PASSES

INITIALIZATIONS = ('random', 'related', 'dclm', 'noise')
ADAPTATIONS = ('full', 'toward_initialization', 'frozen_head', 'frozen_body')


def config(initialization='related', ntrain=30000, epochs=16, lr=1e-5, wd=0.0, adaptation='full'):
    if ntrain not in SAMPLE_COUNTS or epochs not in PASSES:
        raise ValueError('Use 1k/3k/10k/30k chunks and 1/4/16 passes.')
    if initialization not in INITIALIZATIONS or adaptation not in ADAPTATIONS:
        raise ValueError('Unknown initialization or adaptation.')
    return Config(
        name=f'ft-{initialization}-{ntrain}-e{epochs}-lr{lr:g}-wd{wd:g}-{adaptation}',
        initialization=initialization,
        ntrain=ntrain,
        epochs=epochs,
        lr=lr,
        wd=wd,
        adaptation=adaptation,
    )


def sample_scaling(initialization='related', lr=1e-5, wd=0.0):
    """Part 4(b): full fine-tuning at 16 passes for each sample count."""
    return [config(initialization, ntrain=n, epochs=16, lr=lr, wd=wd) for n in SAMPLE_COUNTS]


if __name__ == '__main__':
    # Start with one experiment; expand LRs/sample counts after checking it.
    print(config())
    print(config(adaptation='frozen_head'))
    print(config(adaptation='frozen_body'))
    # Part 4(b): fit the 16-pass loss-versus-samples curve using analysis.power_law.
    print(sample_scaling())
