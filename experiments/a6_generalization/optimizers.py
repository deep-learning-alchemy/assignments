"""AdamW and the course Hyperball update; SAM's two passes are in training.py."""

import torch
from optimizers import build_optimizer

# The same fallback-LR ratio used by the instructor's AdamH implementation.
ADAM_LR_RATIO = 0.000656 / 0.00630


class Hyperball(torch.optim.Optimizer):
    def __init__(self, model, lr):
        matrices = {id(m.weight) for m in model.modules() if isinstance(m, torch.nn.Linear)}
        groups = [
            dict(
                params=[
                    p
                    for p in model.parameters()
                    if p.requires_grad and (id(p) in matrices) == selected
                ],
                lr=lr if selected else lr * ADAM_LR_RATIO,
                hyperball=selected,
            )
            for selected in (True, False)
        ]
        super().__init__(groups, dict(betas=(0.9, 0.95), eps=1e-8))

    @torch.no_grad()
    def step(self, closure=None):
        # Form Adam directions explicitly so projection occurs on each tensor.
        for g in self.param_groups:
            b1, b2 = g['betas']
            for p in g['params']:
                if p.grad is None:
                    continue
                s = self.state[p]
                if not s:
                    s.update(step=0, mean=torch.zeros_like(p), variance=torch.zeros_like(p))
                s['step'] += 1
                s['mean'].lerp_(p.grad, 1 - b1)
                s['variance'].mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                direction = (s['mean'] / (1 - b1 ** s['step'])) / (
                    s['variance'].sqrt() / (1 - b2 ** s['step']) ** 0.5 + g['eps']
                )
                if g['hyperball']:
                    radius = p.norm()
                    candidate = p - g['lr'] * direction * radius / direction.norm().clamp_min(1e-10)
                    p.copy_(candidate * radius / candidate.norm().clamp_min(1e-10))
                else:
                    p.add_(direction, alpha=-g['lr'])


def build(model, config):
    if config.optimizer == 'hyperball':
        if config.wd != 0:
            raise ValueError('Hyperball uses WD0; tune its LR independently.')
        return Hyperball(model, config.lr)
    if config.optimizer not in ('adamw', 'sam'):
        raise ValueError('Add exploratory optimizers here.')
    wd = 0.0 if config.adaptation == 'toward_initialization' else config.wd
    return build_optimizer(model, 'adamw', config.lr, wd, beta1=0.9, beta2=0.95)
