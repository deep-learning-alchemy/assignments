"""P3: linear/constant horizons, sqrt batch scaling, and SAM."""

import math
from .common import Config


def horizons(schedule='linear', lr=Config.lr):
    return [
        Config(name=f'{schedule}-e{e}-lr{lr:g}', schedule=schedule, epochs=e, lr=lr)
        for e in (16, 32, 64, 128, 256)
    ]


def batch_size(batch):
    factor = math.sqrt(batch / Config.batch)
    return Config(name=f'batch{batch}', batch=batch, lr=Config.lr * factor, wd=Config.wd * factor)


def sam(lr=Config.lr, wd=Config.wd, rho=Config.rho):
    return Config(name=f'sam-lr{lr:g}-wd{wd:g}-rho{rho:g}', optimizer='sam', lr=lr, wd=wd, rho=rho)


def directional_sharpness(model, ids, radius=0.05):
    """max± L(theta±radius*g/||g||)-L(theta), with dropout disabled.

    This is a directional lower bound, not a Hessian eigenvalue or a
    parameterization-invariant generalization measure.
    """
    import torch
    from .training import nll, autocast

    was_training = model.training
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    saved = [p.detach().clone() for p in params]
    device = next(model.parameters()).device
    ids = torch.as_tensor(ids, dtype=torch.long, device=device)

    def objective():
        with autocast(device):
            return nll(model(ids), ids)

    try:
        with torch.enable_grad():
            base = objective()
            grads = torch.autograd.grad(base, params, allow_unused=True)
        grads = [torch.zeros_like(p) if g is None else g for p, g in zip(params, grads)]
        norm = torch.stack([g.float().norm() for g in grads]).norm().clamp_min(1e-12)
        losses = []
        with torch.no_grad():
            for sign in (-1, 1):
                for p, value, g in zip(params, saved, grads):
                    p.copy_(value + sign * radius * g / norm)
                losses.append(float(objective()))
        return dict(
            radius=radius,
            base_loss=float(base.detach()),
            max_loss_increase=max(losses) - float(base.detach()),
        )
    finally:
        with torch.no_grad():
            for p, value in zip(params, saved):
                p.copy_(value)
        model.train(was_training)


if __name__ == '__main__':
    print(horizons())
    print(horizons('constant', lr=0.001))
    print([batch_size(b) for b in (16, 64, 256, 1024, 4096)])
    print(sam())
    # Add an optimizer in optimizers.py for your exploratory comparison.
