"""Small hook system for optional training diagnostics."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import importlib
import sys
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from train import TrainConfig


AFTER_BACKWARD = "after_backward"
AFTER_TRAIN_STEP = "after_train_step"
AFTER_EVAL = "after_eval"
LOGGER_EVENTS = frozenset({AFTER_BACKWARD, AFTER_TRAIN_STEP, AFTER_EVAL})
MetricFn = Callable[["LoggerContext"], Mapping[str, Any]]
MetricFnSpec = MetricFn | str


@dataclass(frozen=True)
class LoggerContext:
    config: TrainConfig
    step: int
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: Any
    input_ids: torch.Tensor | None
    loss: torch.Tensor | None
    val_batches: Mapping[str, Any]


@dataclass(frozen=True)
class MetricLogger:
    event: str
    fn: MetricFnSpec


class LoggerManager:
    def __init__(
        self,
        loggers: Sequence[MetricLogger] = (),
        *,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.loggers = tuple(resolve_metric_logger(logger) for logger in loggers)
        self._setup_loggers: list[MetricLogger] = []
        self._validate_loggers()

    def _validate_loggers(self) -> None:
        for logger in self.loggers:
            if not isinstance(logger, MetricLogger):
                raise TypeError(
                    "metric_loggers must contain only MetricLogger objects, got "
                    f"{type(logger).__name__}."
                )
            if logger.event not in LOGGER_EVENTS:
                raise ValueError(
                    f"Unknown logger event {logger.event!r}. "
                    f"Expected one of {sorted(LOGGER_EVENTS)}."
                )
            if not callable(logger.fn):
                raise TypeError(f"Metric logger fn must be callable, got {logger.fn!r}.")

    def describe(self) -> list[dict[str, str]]:
        return [
            {
                "event": logger.event,
                "function": callable_name(logger.fn),
            }
            for logger in self.loggers
        ]

    def setup(self, context: LoggerContext) -> None:
        if not self.enabled:
            return
        for logger in self.loggers:
            setup = getattr(logger.fn, "setup", None)
            if setup is not None:
                setup(context)
                self._setup_loggers.append(logger)

    def close(self) -> None:
        for logger in reversed(self._setup_loggers):
            close = getattr(logger.fn, "close", None)
            if close is not None:
                close()
        self._setup_loggers.clear()

    def collect(self, event: str, context: LoggerContext) -> dict[str, Any]:
        if not self.enabled:
            return {}

        stats: dict[str, Any] = {}
        for logger in self.loggers:
            if logger.event != event:
                continue
            logger_stats = logger.fn(context)
            if not isinstance(logger_stats, Mapping):
                raise TypeError(
                    f"Metric logger {callable_name(logger.fn)} must return a dict, "
                    f"got {type(logger_stats).__name__}."
                )
            for key, value in logger_stats.items():
                if not isinstance(key, str):
                    raise TypeError(
                        f"Metric logger {callable_name(logger.fn)} returned "
                        f"non-string key {key!r}."
                    )
                metric_key = f"logging/{key}"
                if metric_key in stats:
                    raise ValueError(
                        f"Duplicate metric key returned by loggers: {metric_key}"
                    )
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    value = value.detach().item()
                stats[metric_key] = value
        return stats


def callable_name(fn: Callable[..., Any]) -> str:
    if hasattr(fn, "__name__"):
        return str(fn.__name__)
    return type(fn).__name__


def resolve_metric_logger(logger: MetricLogger) -> MetricLogger:
    if not isinstance(logger, MetricLogger):
        return logger
    return MetricLogger(event=logger.event, fn=resolve_metric_fn(logger.fn))


def resolve_metric_fn(fn: MetricFnSpec) -> MetricFn:
    if callable(fn):
        return fn
    if not isinstance(fn, str):
        raise TypeError(f"Metric logger fn must be callable or import path, got {fn!r}.")
    module_name, _, qualname = fn.partition(":")
    if not module_name or not qualname:
        raise ValueError(
            f"Metric logger import path must look like 'module:function', got {fn!r}."
        )
    resolved: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        resolved = getattr(resolved, part)
    if not callable(resolved):
        raise TypeError(f"Metric logger import path {fn!r} did not resolve to a callable.")
    return resolved


def importable_metric_loggers(loggers: Sequence[MetricLogger]) -> tuple[MetricLogger, ...]:
    return tuple(
        MetricLogger(event=logger.event, fn=importable_metric_fn(logger.fn))
        for logger in loggers
    )


def importable_metric_fn(fn: MetricFnSpec) -> MetricFnSpec:
    """Functions become import paths. A callable object whose class lives in an
    importable module (e.g. sharpness_logging.SharpnessLogger(every=25)) is kept
    as is, so its constructor arguments travel with the pickled config."""
    cls = type(fn)
    if (
        not isinstance(fn, str)
        and not hasattr(fn, "__qualname__")
        and cls.__module__ not in ("__main__", "builtins")
        and "<locals>" not in cls.__qualname__
    ):
        return fn
    return metric_fn_import_path(fn)


def metric_fn_import_path(fn: MetricFnSpec) -> str:
    if isinstance(fn, str):
        return fn
    module_name = getattr(fn, "__module__", "")
    qualname = getattr(fn, "__qualname__", "")
    if module_name == "__main__":
        main_spec = getattr(sys.modules.get("__main__"), "__spec__", None)
        module_name = getattr(main_spec, "name", "") or module_name
    if not module_name or not qualname or "<locals>" in qualname:
        raise ValueError(
            "Metric logger functions used with Modal must be module-level functions "
            "or explicit import paths like 'my_module:my_logger'."
        )
    return f"{module_name}:{qualname}"
