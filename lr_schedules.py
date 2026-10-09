import math
import re

import torch
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0.0):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            return [base_lr * (step / self.warmup_steps) for base_lr in self.base_lrs]
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cosine_decay = 0.5 * (1 + torch.cos(torch.tensor(progress * torch.pi)).item())
        return [self.min_lr + (base_lr - self.min_lr) * cosine_decay for base_lr in self.base_lrs]


class WarmupLinearDecayScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0.0):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            return [base_lr * (step / self.warmup_steps) for base_lr in self.base_lrs]
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return [self.min_lr + (base_lr - self.min_lr) * (1 - progress) for base_lr in self.base_lrs]


class WarmupStableDecayScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, decay_fraction, min_lr=0.0):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.decay_steps = int(total_steps * decay_fraction)
        self.stable_end = total_steps - self.decay_steps
        self.min_lr = min_lr
        super().__init__(optimizer)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            return [base_lr * (step / self.warmup_steps) for base_lr in self.base_lrs]
        if step < self.stable_end:
            return list(self.base_lrs)
        progress = (step - self.stable_end) / max(1, self.decay_steps)
        progress = min(1.0, max(0.0, progress))
        return [self.min_lr + (base_lr - self.min_lr) * (1 - progress) for base_lr in self.base_lrs]


class ConstantScheduler(_LRScheduler):
    def get_lr(self):
        return list(self.base_lrs)


class StepScheduler(_LRScheduler):
    """Constant learning rate, multiplied by `factor` from step `at` on (no warmup)."""

    def __init__(self, optimizer, at, factor):
        self.at = at
        self.factor = factor
        super().__init__(optimizer)

    def get_lr(self):
        scale = self.factor if self.last_epoch >= self.at else 1.0
        return [base_lr * scale for base_lr in self.base_lrs]


class WarmupInverseSqrtScheduler(_LRScheduler):
    """Vaswani et al. (2017): min(t / w, sqrt(w / t)), independent of the horizon."""

    def __init__(self, optimizer, warmup_steps):
        if warmup_steps <= 0:
            raise ValueError("invsqrt needs a positive warmup length.")
        self.warmup_steps = warmup_steps
        super().__init__(optimizer)

    def get_lr(self):
        step = self.last_epoch
        multiplier = min(step / self.warmup_steps, math.sqrt(self.warmup_steps / max(step, 1)))
        return [base_lr * multiplier for base_lr in self.base_lrs]


class WarmupCosineRestartsScheduler(_LRScheduler):
    """SGDR (Loshchilov and Hutter, 2017): repeated cosine cycles of `cycle_steps` updates."""

    def __init__(self, optimizer, warmup_steps, cycle_steps, min_lr_ratio=0.0):
        if cycle_steps <= 0:
            raise ValueError("sgdr needs a positive cycle length.")
        self.warmup_steps = warmup_steps
        self.cycle_steps = cycle_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            return [base_lr * (step / self.warmup_steps) for base_lr in self.base_lrs]
        cycle_position = (step - self.warmup_steps) % self.cycle_steps
        cosine = 0.5 * (1 + math.cos(math.pi * cycle_position / self.cycle_steps))
        multiplier = self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine
        return [base_lr * multiplier for base_lr in self.base_lrs]


def dip_multiplier(progress, dip):
    """Linear decay from 1 to `scale` over [start, bottom], then back to 1 by `end`."""
    start, bottom, end, scale = dip
    if progress <= start or progress >= end:
        return 1.0
    if progress <= bottom:
        fraction = (progress - start) / max(bottom - start, 1e-12)
    else:
        fraction = (end - progress) / max(end - bottom, 1e-12)
    return 1.0 + (scale - 1.0) * fraction


class DipScheduler(_LRScheduler):
    """Multiply another schedule by a temporary dip, given as fractions of training."""

    def __init__(self, optimizer, base_scheduler, total_steps, dip):
        self.base_scheduler = base_scheduler
        self.total_steps = total_steps
        self.dip = dip
        super().__init__(optimizer)

    def get_lr(self):
        self.base_scheduler.last_epoch = self.last_epoch
        multiplier = dip_multiplier(self.last_epoch / self.total_steps, self.dip)
        return [lr * multiplier for lr in self.base_scheduler.get_lr()]


def build_scheduler(
    optimizer,
    lr_schedule,
    warmup_steps,
    total_steps,
    min_lr_ratio=0.0,
    lr_dip=None,
):
    scheduler = _build_base_scheduler(
        optimizer, lr_schedule, warmup_steps, total_steps, min_lr_ratio
    )
    if lr_dip is None:
        return scheduler
    return DipScheduler(optimizer, scheduler, total_steps, lr_dip)


def _build_base_scheduler(optimizer, lr_schedule, warmup_steps, total_steps, min_lr_ratio):
    if lr_schedule == "invsqrt":
        return WarmupInverseSqrtScheduler(optimizer, warmup_steps=warmup_steps)
    if lr_schedule.startswith("sgdr"):
        return WarmupCosineRestartsScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            cycle_steps=int(lr_schedule[4:]),
            min_lr_ratio=min_lr_ratio,
        )
    if lr_schedule == "cos":
        return WarmupCosineScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )
    if lr_schedule in ["linear", "lin"]:
        return WarmupLinearDecayScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )
    if lr_schedule in ["constant", "const", "flat"]:
        return ConstantScheduler(optimizer)
    if lr_schedule.startswith("step"):
        # e.g. "step750x0.5": halve the learning rate from step 750 on.
        match = re.fullmatch(r"step(\d+)x([0-9.]+(?:e-?\d+)?)", lr_schedule)
        if match is None:
            raise ValueError(f"Step schedules look like 'step750x0.5', got {lr_schedule!r}.")
        return StepScheduler(optimizer, at=int(match[1]), factor=float(match[2]))
    if lr_schedule.startswith("wsd"):
        decay_fraction = float(lr_schedule[3:])
        return WarmupStableDecayScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            decay_fraction=decay_fraction,
        )
    raise ValueError(f"Unknown lr_schedule: {lr_schedule}")


def set_scheduler_to_completed_steps(scheduler, completed_steps):
    scheduler.last_epoch = completed_steps
    if hasattr(scheduler, "_step_count"):
        scheduler._step_count = completed_steps + 1
    lrs = scheduler.get_lr()
    for param_group, lr in zip(scheduler.optimizer.param_groups, lrs):
        param_group["lr"] = lr
    scheduler._last_lr = list(lrs)
