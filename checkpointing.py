import json
import os
import random
import signal
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from lr_schedules import set_scheduler_to_completed_steps
from model_io import TRAINING_CHECKPOINT_VERSION, load_model, save_model


CHECKPOINT_VERSION = TRAINING_CHECKPOINT_VERSION


def _to_cpu_snapshot(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {key: _to_cpu_snapshot(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu_snapshot(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu_snapshot(value) for value in obj)
    return obj


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        cuda_states = state["cuda"]
        for device_idx, device_state in enumerate(cuda_states[: torch.cuda.device_count()]):
            torch.cuda.set_rng_state(device_state, device=device_idx)


def _atomic_torch_save(payload, tmp_path, final_path):
    torch.save(payload, tmp_path)
    os.replace(tmp_path, final_path)


def _atomic_json_write(payload, tmp_path, final_path):
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, final_path)


class AsyncCheckpointManager:
    def __init__(self, run_dir, enabled=True):
        self.run_dir = Path(run_dir)
        self.enabled = enabled
        self.latest_path = self.run_dir / "latest.pt"
        self.run_state_path = self.run_dir / "run_state.json"
        self._executor = ThreadPoolExecutor(max_workers=1) if enabled else None
        self._pending_future = None

    def close(self):
        self.wait()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def wait(self):
        if self._pending_future is not None:
            self._pending_future.result()
            self._pending_future = None

    def has_pending_write(self):
        if self._pending_future is None:
            return False
        if self._pending_future.done():
            self.wait()
            return False
        return True

    def cleanup_stale_temp_files(self):
        if not self.run_dir.exists():
            return
        for path in self.run_dir.glob("*.tmp.*.pt"):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def load_latest(self, max_attempts=5, sleep_seconds=2.0):
        if not self.enabled or not self.latest_path.exists():
            return None
        print(f"Loading latest training checkpoint from {self.latest_path}")
        for attempt in range(1, max_attempts + 1):
            try:
                return torch.load(self.latest_path, map_location="cpu", weights_only=False)
            except (OSError, RuntimeError) as exc:
                if attempt == max_attempts:
                    raise
                print(
                    f"Checkpoint load failed on attempt {attempt}/{max_attempts}: "
                    f"{exc}. Retrying in {sleep_seconds}s."
                )
                time.sleep(sleep_seconds)

    def load_run_state(self):
        if not self.run_state_path.exists():
            return {}
        with open(self.run_state_path) as f:
            return json.load(f)

    def write_run_state(self, state):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.run_dir / f"run_state.tmp.{os.getpid()}.{uuid.uuid4().hex}.json"
        _atomic_json_write(state, tmp_path, self.run_state_path)

    def _save_async(self, payload, final_path, force=False):
        if not self.enabled:
            return False
        final_path = Path(final_path)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.has_pending_write():
            completed_steps = payload.get("completed_steps", "unknown")
            if not force:
                print(
                    f"Skipping async checkpoint for step {completed_steps} "
                    "because the previous checkpoint write is still pending."
                )
                return False
            self.wait()
        snapshot = _to_cpu_snapshot(payload)
        tmp_path = final_path.with_name(
            f"{final_path.stem}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
            f"{final_path.suffix}"
        )
        print(
            f"Saving async checkpoint for step {snapshot['completed_steps']} "
            f"to {final_path}"
        )
        self._pending_future = self._executor.submit(
            _atomic_torch_save,
            snapshot,
            tmp_path,
            final_path,
        )
        return True

    def save_latest_async(self, payload, force=False):
        return self._save_async(payload, self.latest_path, force=force)

    def step_checkpoint_path(self, completed_steps):
        return self.run_dir / f"step_{completed_steps}.pt"

    def save_step_async(self, payload, completed_steps):
        return self._save_async(
            payload,
            self.step_checkpoint_path(completed_steps),
            force=True,
        )


@dataclass
class InterruptState:
    stop_requested: bool = False
    received_signal: int | None = None


def normalize_checkpoint_steps(checkpoint_steps):
    normalized = []
    seen = set()
    for step in checkpoint_steps:
        if not isinstance(step, int) or isinstance(step, bool):
            raise TypeError(
                f"keep_checkpoint_steps must contain ints, got {step!r}."
            )
        if step <= 0:
            raise ValueError(f"keep_checkpoint_steps must be positive, got {step}.")
        if step in seen:
            raise ValueError(f"keep_checkpoint_steps contains duplicate step {step}.")
        seen.add(step)
        normalized.append(step)
    return tuple(sorted(normalized))


class TrainingCheckpointer:
    def __init__(
        self,
        run_dir,
        run_name,
        save_model=True,
        resume_from_checkpoint=True,
        latest_checkpoint_frequency=1000,
        keep_checkpoint_steps=(),
        metadata=None,
    ):
        self.run_dir = Path(run_dir)
        self.run_name = run_name
        self.save_model = save_model
        self.resume_from_checkpoint = resume_from_checkpoint
        self.latest_checkpoint_frequency = latest_checkpoint_frequency
        self.keep_checkpoint_steps = frozenset(
            normalize_checkpoint_steps(keep_checkpoint_steps)
        )
        self.latest_enabled = (
            save_model
            and latest_checkpoint_frequency is not None
            and latest_checkpoint_frequency > 0
        )
        self.step_checkpoints_enabled = save_model and bool(self.keep_checkpoint_steps)
        self.enabled = self.latest_enabled or self.step_checkpoints_enabled
        self.metadata = metadata or {}
        self.manager = AsyncCheckpointManager(self.run_dir, enabled=self.enabled)
        self.latest_checkpoint = None
        self.manager.cleanup_stale_temp_files()

    @property
    def mode(self):
        if not self.enabled:
            return "disabled"
        kinds = []
        if self.latest_enabled:
            kinds.append("latest")
        if self.step_checkpoints_enabled:
            kinds.append("kept_steps")
        return "async:" + "+".join(kinds)

    @property
    def latest_path(self):
        return self.manager.latest_path

    def __enter__(self):
        self.manager.cleanup_stale_temp_files()
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.close()

    def close(self):
        self.manager.close()

    def set_metadata(self, metadata):
        self.metadata = metadata

    def load_completed_model_if_available(self, device, precision, force_run=False):
        if force_run or not self.save_model:
            return None
        try:
            print(f"Trying to load model from {self.run_dir}")
            model = load_model(
                self.run_dir,
                device=device,
                precision=precision,
            )
            print(f"Model already exists at {self.run_dir}; returning checkpoint")
            return model
        except Exception as exc:
            print(f"Exception: {exc}")
            print(f"No model found at {self.run_dir}; training from scratch")
            return None

    def restore_if_available(self, model, optimizer, scheduler, total_steps):
        if not (self.resume_from_checkpoint and self.latest_enabled):
            return 0

        checkpoint = self.manager.load_latest()
        self.latest_checkpoint = checkpoint
        if checkpoint is None:
            return 0

        checkpoint_run_name = checkpoint.get("metadata", {}).get("run_name")
        if checkpoint_run_name is not None and checkpoint_run_name != self.run_name:
            raise ValueError(
                f"Checkpoint run_name mismatch: {checkpoint_run_name} != {self.run_name}"
            )

        start_step = self._restore_training_state(
            model,
            optimizer,
            scheduler,
            checkpoint,
        )
        if start_step > total_steps:
            raise ValueError(
                f"Checkpoint completed_steps={start_step} exceeds submitted "
                f"total_steps={total_steps}"
            )
        print(f"Resuming training from completed step {start_step}/{total_steps}")
        return start_step

    def _restore_training_state(self, model, optimizer, scheduler, checkpoint):
        version = checkpoint.get("version")
        if version != CHECKPOINT_VERSION:
            raise ValueError(f"Unsupported checkpoint version: {version}")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        completed_steps = int(checkpoint["completed_steps"])
        set_scheduler_to_completed_steps(scheduler, completed_steps)
        restore_rng_state(checkpoint.get("rng_state"))
        return completed_steps

    def wandb_run_id(self):
        if self.latest_checkpoint is not None:
            wandb_run_id = self.latest_checkpoint.get("wandb_run_id")
            if wandb_run_id is not None:
                return wandb_run_id
        if self.latest_enabled:
            return self.manager.load_run_state().get("wandb_run_id")
        return None

    def write_wandb_run_state(self, wandb_run_id, wandb_entity, wandb_project):
        if not self.latest_enabled:
            return
        self.manager.write_run_state(
            {
                "wandb_run_id": wandb_run_id,
                "run_name": self.run_name,
                "wandb_entity": wandb_entity,
                "wandb_project": wandb_project,
                "config": self.metadata,
            }
        )

    def maybe_save_training_checkpoint(
        self,
        completed_steps,
        total_steps,
        model,
        optimizer,
        wandb_run_id=None,
        force=False,
        extra_state=None,
    ):
        if not self.enabled:
            return False
        completed_training = completed_steps >= total_steps
        should_save_latest = self.latest_enabled and (
            completed_steps % self.latest_checkpoint_frequency == 0
            or completed_training
            or force
        )
        should_save_step = completed_steps in self.keep_checkpoint_steps
        if not (should_save_latest or should_save_step):
            return False

        payload = {
            "version": CHECKPOINT_VERSION,
            "completed_steps": completed_steps,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "rng_state": capture_rng_state(),
            "wandb_run_id": wandb_run_id,
            "metadata": self.metadata,
            **(extra_state or {}),
        }
        saved = False
        if should_save_latest:
            saved = self.manager.save_latest_async(
                payload,
                force=completed_training or force,
            )
        if should_save_step:
            saved = self.manager.save_step_async(payload, completed_steps) or saved
        return saved

    def save_final_model(self, model, completed_training, last_completed_step, total_steps):
        if self.save_model and completed_training:
            print(f"Saving model to {self.run_dir}")
            save_model(model, self.run_dir, metadata=self.metadata)
        elif self.save_model:
            checkpoint_message = "no training checkpoint is configured"
            if self.latest_enabled:
                checkpoint_message = (
                    f"latest training checkpoint is saved at {self.latest_path}"
                )
            elif self.step_checkpoints_enabled:
                checkpoint_message = (
                    f"kept step checkpoints are saved under {self.run_dir}"
                )
            print(
                f"Not saving final model because training stopped at step "
                f"{last_completed_step}/{total_steps}; {checkpoint_message}"
            )
        else:
            print("Not saving model since save_model is False")

    def report_interrupted(
        self,
        interrupt_state,
        completed_steps,
        total_steps,
        checkpoint_saved,
    ):
        if checkpoint_saved:
            print(
                f"Received signal {interrupt_state.received_signal}; saved checkpoint "
                f"at completed step {completed_steps}/{total_steps} and stopping."
            )
        else:
            print(
                f"Received signal {interrupt_state.received_signal}; stopping at "
                f"completed step {completed_steps}/{total_steps} with no checkpoint "
                "written for this step."
            )

    @contextmanager
    def capture_interrupts(self):
        state = InterruptState()

        def request_stop(signum, _frame):
            state.stop_requested = True
            state.received_signal = signum

        old_signal_handlers = {
            signal.SIGTERM: signal.getsignal(signal.SIGTERM),
            signal.SIGINT: signal.getsignal(signal.SIGINT),
        }
        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        try:
            yield state
        finally:
            for sig, handler in old_signal_handlers.items():
                signal.signal(sig, handler)
