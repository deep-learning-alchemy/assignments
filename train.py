import hashlib
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import torch
from torch.nn import DataParallel
from torch.nn import functional as F

from checkpointing import (
    TrainingCheckpointer,
    normalize_checkpoint_steps,
    restore_rng_state,
)
from data import (
    DEFAULT_DATA_SEED,
    TokenDatasetConfig,
    dclm_train_dataset,
    dclm_val_dataset,
    load_token_dataset,
)
from lr_schedules import build_scheduler, set_scheduler_to_completed_steps
from metric_logging import (
    AFTER_BACKWARD,
    AFTER_EVAL,
    AFTER_TRAIN_STEP,
    LoggerContext,
    LoggerManager,
    MetricLogger,
)
from model_config import (
    LMConfig,
    model_dtype_for_precision,
    resolve_model_config,
    validate_precision,
)
from model_io import load_model, load_model_config, resolve_model_builder
from modeling import AutoregressiveLM, initialize_model
from optimizers import build_optimizer
from module_rms_logging import log_module_rms
from utils import (
    MODEL_DIR,
    WANDB_ENTITY,
    WANDB_PROJECT,
    autocast_context,
    configure_deterministic_training,
    create_batches,
    format_token_count,
    parameter_count,
    precision_config,
    seed_everything,
)


@dataclass(frozen=True)
class TrainConfig:
    model_name: str = "d8"
    model_config: LMConfig | None = None
    model_builder: str | None = None
    model_builder_kwargs: dict = field(default_factory=dict)
    embedding_init_scale: float = 1.0
    run_name_suffix: str | None = None
    init_checkpoint_path: str | Path | None = None
    learning_rate: float = 3e-3
    num_epochs: float = 1.0
    batch_size: int = 64
    num_micro_batches: int = 1
    num_evals: int = 100
    warmup_percent: float = 0.01
    weight_decay: float = 0.1
    grad_norm: float | None = 1.0
    dropout: float = 0.0
    qk_norm: bool = True
    tie_word_embeddings: bool = False
    deterministic: bool = False
    perturb_one_token: bool = False
    wandb_tags: tuple[str, ...] = field(default_factory=tuple)
    wandb_online: bool = True
    force_run: bool = False
    data_seed: int | None = DEFAULT_DATA_SEED
    model_seed: int | None = 42
    train_dataset: TokenDatasetConfig = field(default_factory=dclm_train_dataset)
    val_dataset: TokenDatasetConfig = field(default_factory=dclm_val_dataset)
    optimizer_name: str = "adamw"
    beta1: float = 0.9
    beta2: float = 0.95
    save_model: bool = True
    lr_schedule: str = "linear"
    # Absolute warmup length in updates; overrides warmup_percent when set.
    warmup_steps: int | None = None
    # Floor of the "sgdrT" schedule, as a fraction of learning_rate.
    min_lr_ratio: float = 0.0
    # (start, bottom, end, scale) as fractions of training: the LR multiplier
    # falls linearly from 1 to scale over [start, bottom], then returns to 1 by end.
    lr_dip: tuple[float, float, float, float] | None = None
    # Evaluation-only exponential moving averages of the parameters.
    ema_decays: tuple[float, ...] = field(default_factory=tuple)
    # Seeded permutation of the training micro-batch order.
    batch_order_seed: int | None = None
    # Continue another run from its kept step checkpoint, with optimizer state.
    fork_from_run: str | None = None
    fork_from_step: int | None = None
    # Stop after this many updates; the schedule still uses the full run length.
    stop_at_step: int | None = None
    precision: str = "mp"
    num_train_sequences: int = 600_000
    resume_from_checkpoint: bool = True
    latest_checkpoint_frequency: int | None = 1000
    keep_checkpoint_steps: tuple[int, ...] = field(default_factory=tuple)
    torch_compile_mode: str = "reduce-overhead"
    model_dir: str | Path = MODEL_DIR
    wandb_entity: str = WANDB_ENTITY
    wandb_project: str = WANDB_PROJECT
    metric_loggers: tuple[MetricLogger, ...] = (
        MetricLogger(event=AFTER_BACKWARD, fn=log_module_rms),
    )


def training_model_name(config):
    if config.init_checkpoint_path is not None:
        return load_model_config(config.init_checkpoint_path).name
    if config.model_config is not None:
        return config.model_config.name
    return config.model_name


def format_duration(seconds):
    seconds = int(max(0.0, seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def dataset_run_tags(config):
    if (
        config.train_dataset == dclm_train_dataset()
        and config.val_dataset == dclm_val_dataset()
    ):
        return []

    train_tag = config.train_dataset.tag
    val_tag = config.val_dataset.tag
    if train_tag == val_tag:
        return [train_tag]
    return [f"train-{train_tag}", f"val-{val_tag}"]


def configured_train_tokens(config):
    return config.num_train_sequences * config.train_dataset.context_length


def training_run_name(config):
    num_epochs = float(config.num_epochs)
    train_tokens = configured_train_tokens(config)
    run_name = f"model-{training_model_name(config)}-lr{config.learning_rate}"
    if config.batch_size != TrainConfig.batch_size:
        run_name += f"-bs{config.batch_size}"
    if num_epochs != TrainConfig.num_epochs:
        run_name += f"-epochs{num_epochs}"
    run_name += f"-tok{format_token_count(train_tokens)}"
    if config.weight_decay != TrainConfig.weight_decay:
        run_name += f"-wd{config.weight_decay}"
    if config.embedding_init_scale != TrainConfig.embedding_init_scale:
        run_name += f"-emb{config.embedding_init_scale:g}"
    if config.warmup_percent != TrainConfig.warmup_percent:
        run_name += f"-warmup{config.warmup_percent}"
    if config.grad_norm != TrainConfig.grad_norm:
        if config.grad_norm is None:
            run_name += "-nogradclip"
        else:
            run_name += f"-gn{config.grad_norm}"
    if config.dropout != TrainConfig.dropout:
        run_name += f"-dropout{config.dropout}"
    if config.qk_norm != TrainConfig.qk_norm:
        run_name += "-noqknorm"
    if config.tie_word_embeddings != TrainConfig.tie_word_embeddings:
        run_name += "-tiedemb"
    if config.deterministic != TrainConfig.deterministic:
        run_name += "-deterministic"
    if config.perturb_one_token != TrainConfig.perturb_one_token:
        run_name += "-perturb1tok"
    if config.data_seed != TrainConfig.data_seed:
        if config.data_seed is None:
            run_name += "-dsnone"
        else:
            run_name += f"-ds{config.data_seed}"
    if config.model_seed != TrainConfig.model_seed:
        if config.model_seed is None:
            run_name += "-msnone"
        else:
            run_name += f"-ms{config.model_seed}"
    if config.beta1 != TrainConfig.beta1:
        run_name += f"-b1{config.beta1}"
    if config.beta2 != TrainConfig.beta2:
        run_name += f"-b{config.beta2}"
    for dataset_tag in dataset_run_tags(config):
        run_name += f"-{dataset_tag}"
    if config.optimizer_name != TrainConfig.optimizer_name:
        run_name += f"-{config.optimizer_name}"
    if config.lr_schedule != TrainConfig.lr_schedule:
        run_name += f"-{config.lr_schedule}"
    if config.precision != TrainConfig.precision:
        run_name += f"-{config.precision}"
    if config.warmup_steps is not None:
        run_name += f"-warmupsteps{config.warmup_steps}"
    if config.min_lr_ratio != TrainConfig.min_lr_ratio:
        run_name += f"-minlr{config.min_lr_ratio:g}"
    if config.lr_dip is not None:
        start, bottom, end, scale = config.lr_dip
        run_name += f"-dip{start:g}-{bottom:g}-{end:g}x{scale:g}"
    if config.ema_decays:
        run_name += "-ema" + "-".join(f"{decay:g}" for decay in config.ema_decays)
    if config.batch_order_seed is not None:
        run_name += f"-order{config.batch_order_seed}"
    if config.fork_from_run is not None:
        source = hashlib.sha1(config.fork_from_run.encode()).hexdigest()[:6]
        run_name += f"-fork{config.fork_from_step}-{source}"
    if config.stop_at_step is not None:
        run_name += f"-stop{config.stop_at_step}"
    if config.run_name_suffix is not None:
        run_name += f"-{config.run_name_suffix}"
    return run_name


def build_model(config, device):
    if config.init_checkpoint_path is not None:
        print(f"Loading initial model from {config.init_checkpoint_path}")
        return load_model(
            config.init_checkpoint_path,
            device=device,
            precision=config.precision,
            qk_norm=config.qk_norm,
            tie_word_embeddings=config.tie_word_embeddings,
            dropout=config.dropout,
        )

    dtype = model_dtype_for_precision(config.precision)
    model_config = resolve_model_config(
        model_name=config.model_name,
        model_config=config.model_config,
    )
    print(f"Initializing native {model_config.name} model")
    seed_everything(config.model_seed)
    builder = resolve_model_builder(config.model_builder)
    if builder is None:
        model = AutoregressiveLM(
            model_config,
            dtype=dtype,
            qk_norm=config.qk_norm,
            tie_word_embeddings=config.tie_word_embeddings,
            dropout=config.dropout,
        )
    else:
        print(f"Using custom model builder {config.model_builder}")
        model = builder(
            model_config,
            dtype=dtype,
            qk_norm=config.qk_norm,
            tie_word_embeddings=config.tie_word_embeddings,
            dropout=config.dropout,
            **config.model_builder_kwargs,
        )
    seed_everything(config.model_seed)
    initializer = getattr(model, "initialize_parameters", None)
    if initializer is None:
        initialize_model(model)
    else:
        initializer()
    post_initialize = getattr(model, "post_initialize", None)
    if post_initialize is not None:
        post_initialize()
    if config.embedding_init_scale != 1.0:
        with torch.no_grad():
            model.get_input_embeddings().weight.mul_(config.embedding_init_scale)
    return model.to(device=device, dtype=dtype)


def causal_lm_loss(logits, input_ids):
    """Mean next-token negative log-likelihood for causal language modeling."""
    shift_logits = logits[..., :-1, :].contiguous().float()
    shift_labels = input_ids[..., 1:].contiguous()
    log_probs = F.log_softmax(shift_logits, dim=-1)
    target_log_probs = log_probs.gather(
        dim=-1,
        index=shift_labels.unsqueeze(-1),
    ).squeeze(-1)
    return -target_log_probs.mean()


class CausalLMTrainingLoss(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        logits = self.model(input_ids=input_ids)
        loss = causal_lm_loss(logits, input_ids)
        aux_source = self.model.module if isinstance(self.model, DataParallel) else self.model
        auxiliary_loss = getattr(aux_source, "auxiliary_loss", None)
        if auxiliary_loss is not None:
            loss = loss + auxiliary_loss()
        return loss


def evaluate(model, val_batches, config, device):
    was_training = model.training
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for input_ids in val_batches:
            input_ids = input_ids.to(device)
            with autocast_context(config.precision, device=device):
                logits = model(input_ids=input_ids)
                loss = causal_lm_loss(logits, input_ids)
                num_next_tokens = input_ids[..., 1:].numel()
                total_nll += loss.item() * num_next_tokens
                total_tokens += num_next_tokens
    if was_training:
        model.train()
    if total_tokens == 0:
        raise ValueError("Cannot evaluate loss on an empty batch iterator.")
    return torch.tensor(total_nll / total_tokens)


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def validate_a3_fields(config):
    if config.warmup_steps is not None and (
        not _is_int(config.warmup_steps) or config.warmup_steps < 0
    ):
        raise ValueError(f"warmup_steps must be a nonnegative int, got {config.warmup_steps!r}.")
    if not 0.0 <= config.min_lr_ratio <= 1.0:
        raise ValueError(f"min_lr_ratio must be in [0, 1], got {config.min_lr_ratio}.")
    if config.min_lr_ratio and not config.lr_schedule.startswith("sgdr"):
        raise ValueError("min_lr_ratio is only used by the sgdr schedule.")
    if config.lr_dip is not None:
        if len(config.lr_dip) != 4:
            raise ValueError("lr_dip must be (start, bottom, end, scale).")
        start, bottom, end, scale = config.lr_dip
        if not 0.0 <= start <= bottom <= end <= 1.0 or start == end or scale < 0:
            raise ValueError(
                "lr_dip needs 0 <= start <= bottom <= end <= 1, start < end, "
                f"and scale >= 0; got {config.lr_dip}."
            )
    if not isinstance(config.ema_decays, tuple):
        raise TypeError("ema_decays must be a tuple.")
    if len(set(config.ema_decays)) != len(config.ema_decays) or not all(
        0.0 < decay < 1.0 for decay in config.ema_decays
    ):
        raise ValueError(f"ema_decays must be distinct values in (0, 1), got {config.ema_decays}.")
    if config.batch_order_seed is not None and not _is_int(config.batch_order_seed):
        raise TypeError("batch_order_seed must be an int or None.")
    if (config.fork_from_run is None) != (config.fork_from_step is None):
        raise ValueError("Set fork_from_run and fork_from_step together.")
    if config.fork_from_step is not None and (
        not _is_int(config.fork_from_step) or config.fork_from_step <= 0
    ):
        raise ValueError("fork_from_step must be a positive int; for step 0, start a fresh run.")
    if config.stop_at_step is not None:
        if not _is_int(config.stop_at_step) or config.stop_at_step <= 0:
            raise ValueError("stop_at_step must be a positive int.")
        if config.fork_from_step is not None and config.stop_at_step <= config.fork_from_step:
            raise ValueError("stop_at_step must come after fork_from_step.")
        late = [step for step in config.keep_checkpoint_steps if step > config.stop_at_step]
        if late:
            raise ValueError(f"keep_checkpoint_steps after stop_at_step are never saved: {late}")


class PermutedBatches:
    """Training batches in a fixed seeded order; micro-batches are permuted."""

    def __init__(self, batches, seed):
        self.batches = batches
        generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(len(batches), generator=generator).tolist()

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self.batches[self.order[idx]]


def fork_checkpoint_path(config):
    return Path(config.model_dir) / config.fork_from_run / f"step_{config.fork_from_step}.pt"


def restore_fork(config, model, optimizer, scheduler):
    """Load model and optimizer state from another run, keeping this config's hyperparameters."""
    path = fork_checkpoint_path(config)
    if not path.is_file():
        raise FileNotFoundError(
            f"Fork checkpoint {path} does not exist. Train {config.fork_from_run!r} "
            f"with keep_checkpoint_steps including {config.fork_from_step} first."
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    completed_steps = int(checkpoint["completed_steps"])
    assert completed_steps == config.fork_from_step
    model.load_state_dict(checkpoint["model_state"])
    hyperparameters = [
        {key: value for key, value in group.items() if key != "params"}
        for group in optimizer.param_groups
    ]
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    for group, values in zip(optimizer.param_groups, hyperparameters):
        group.update(values)
    set_scheduler_to_completed_steps(scheduler, completed_steps)
    restore_rng_state(checkpoint.get("rng_state"))
    print(f"Forked from {path} at completed step {completed_steps}")
    return completed_steps, checkpoint


class ParameterEMA:
    """FP32 parameter averages, used only for evaluation."""

    def __init__(self, params, decays, state=None):
        self.params = params
        self.decays = decays
        device = params[0].device
        if state is None:
            self.averages = {
                decay: [p.detach().float().clone() for p in params] for decay in decays
            }
        else:
            self.averages = {
                decay: [value.to(device) for value in state[f"{decay:g}"]] for decay in decays
            }

    @torch.no_grad()
    def update(self):
        current = [p.detach().float() for p in self.params]
        for decay, values in self.averages.items():
            torch._foreach_lerp_(values, current, 1.0 - decay)

    def state_dict(self):
        return {f"{decay:g}": values for decay, values in self.averages.items()}

    @torch.no_grad()
    def evaluate(self, model, batches, config, device):
        original = [p.detach().clone() for p in self.params]
        losses = {}
        try:
            for decay, values in self.averages.items():
                torch._foreach_copy_(self.params, values)
                losses[decay] = evaluate(model, batches, config, device).item()
        finally:
            torch._foreach_copy_(self.params, original)
        return losses


def checked_train_config(config):
    if not isinstance(config, TrainConfig):
        raise TypeError(f"config must be a TrainConfig, got {type(config).__name__}.")
    validate_precision(config.precision)
    if config.optimizer_name not in {"adamw", "sgd"}:
        raise ValueError(
            f"optimizer_name must be 'adamw' or 'sgd', got {config.optimizer_name!r}."
        )
    for beta_name, beta_value in [("beta1", config.beta1), ("beta2", config.beta2)]:
        if beta_value < 0.0 or beta_value >= 1.0:
            raise ValueError(f"{beta_name} must be in [0, 1), got {beta_value}.")
    if not isinstance(config.qk_norm, bool):
        raise TypeError(f"qk_norm must be bool, got {type(config.qk_norm).__name__}.")
    if not isinstance(config.tie_word_embeddings, bool):
        raise TypeError(
            "tie_word_embeddings must be bool, got "
            f"{type(config.tie_word_embeddings).__name__}."
        )
    if not isinstance(config.dropout, (int, float)) or isinstance(config.dropout, bool):
        raise TypeError(f"dropout must be a float, got {config.dropout!r}.")
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError(f"dropout must be in [0, 1), got {config.dropout}.")
    if not isinstance(config.deterministic, bool):
        raise TypeError(
            f"deterministic must be bool, got {type(config.deterministic).__name__}."
        )
    if not isinstance(config.perturb_one_token, bool):
        raise TypeError(
            "perturb_one_token must be bool, got "
            f"{type(config.perturb_one_token).__name__}."
        )
    if config.deterministic and config.model_seed is None:
        raise ValueError("deterministic=True requires model_seed to be set.")
    if config.model_builder_kwargs is None:
        raise TypeError("model_builder_kwargs must be a dict, got None.")
    if not isinstance(config.model_builder_kwargs, dict):
        raise TypeError(
            "model_builder_kwargs must be a dict, got "
            f"{type(config.model_builder_kwargs).__name__}."
        )
    if not config.embedding_init_scale > 0:
        raise ValueError(
            f"embedding_init_scale must be positive, got {config.embedding_init_scale!r}."
        )
    resolve_model_builder(config.model_builder)
    LoggerManager(config.metric_loggers)
    if config.batch_size % config.num_micro_batches != 0:
        raise ValueError(
            f"batch_size={config.batch_size} must be divisible by "
            f"num_micro_batches={config.num_micro_batches}."
        )
    if config.num_evals <= 0:
        raise ValueError(f"num_evals must be positive, got {config.num_evals}.")
    if not isinstance(config.num_train_sequences, int) or isinstance(
        config.num_train_sequences,
        bool,
    ):
        raise TypeError(
            "num_train_sequences must be a positive integer, got "
            f"{config.num_train_sequences!r}."
        )
    if config.num_train_sequences <= 0:
        raise ValueError(
            f"num_train_sequences must be positive, got {config.num_train_sequences}."
        )
    normalize_checkpoint_steps(config.keep_checkpoint_steps)
    validate_a3_fields(config)
    if not isinstance(config.train_dataset, TokenDatasetConfig):
        raise TypeError(
            f"train_dataset must be a TokenDatasetConfig, got "
            f"{type(config.train_dataset).__name__}."
        )
    if not isinstance(config.val_dataset, TokenDatasetConfig):
        raise TypeError(
            f"val_dataset must be a TokenDatasetConfig, got "
            f"{type(config.val_dataset).__name__}."
        )
    return config


def load_train_and_val_datasets(config):
    train_dataset = load_token_dataset(
        config.train_dataset,
        role="train",
        data_seed=config.data_seed,
    )
    val_dataset = load_token_dataset(
        config.val_dataset,
        role="val",
        data_seed=config.data_seed,
    )
    return train_dataset, {"val": val_dataset}


def select_train_sequences(train_dataset, config):
    available_train_sequences = len(train_dataset)
    if available_train_sequences <= 0:
        raise ValueError("Cannot train on an empty dataset.")

    num_train_sequences = config.num_train_sequences
    if num_train_sequences <= 0 or num_train_sequences > available_train_sequences:
        raise ValueError(
            f"num_train_sequences={num_train_sequences} is invalid for "
            f"dataset of length {available_train_sequences}."
        )
    return train_dataset.select(range(num_train_sequences))


def train(config):
    train_start_time = time.perf_counter()
    config = checked_train_config(config)
    configure_deterministic_training(config.deterministic)
    metric_loggers = LoggerManager(
        config.metric_loggers,
        enabled=config.wandb_online,
    )

    train_dataset, val_dataset = load_train_and_val_datasets(config)
    train_dataset = select_train_sequences(train_dataset, config)
    if config.data_seed is not None:
        train_dataset = train_dataset.shuffle(seed=config.data_seed)
    if config.perturb_one_token:
        first_input_ids = train_dataset[0]["input_ids"]
        assert int(first_input_ids[100]) != 17
        first_input_ids[100] = 17
    effective_train_sequences = len(train_dataset)
    seq_len = len(train_dataset[0]["input_ids"])
    train_tokens = effective_train_sequences * seq_len
    print(
        f"Training on {effective_train_sequences:,} sequences "
        f"({train_tokens:,} tokens)."
    )
    run_name = training_run_name(config)
    model_save_path = Path(config.model_dir) / run_name

    checkpointer = TrainingCheckpointer(
        run_dir=model_save_path,
        run_name=run_name,
        save_model=config.save_model,
        resume_from_checkpoint=config.resume_from_checkpoint,
        latest_checkpoint_frequency=config.latest_checkpoint_frequency,
        keep_checkpoint_steps=config.keep_checkpoint_steps,
    )

    num_cuda_devices = torch.cuda.device_count()
    torch_compile_enabled = num_cuda_devices == 1 and config.num_micro_batches == 1
    torch_compile_mode = config.torch_compile_mode if torch_compile_enabled else None
    device = "cuda:0" if num_cuda_devices > 0 else "cpu"

    completed_model = checkpointer.load_completed_model_if_available(
        device=device,
        precision=config.precision,
        force_run=config.force_run,
    )
    if completed_model is not None:
        checkpointer.close()
        return completed_model

    print(f"Found {num_cuda_devices} CUDA devices; using {device}")
    if config.num_micro_batches > 1:
        assert not torch_compile_enabled
        print("Skipping torch.compile because num_micro_batches > 1.")
    base_model = build_model(config, device=device)
    base_model.train()
    model = base_model
    if num_cuda_devices > 1:
        print(f"Using DataParallel across {num_cuda_devices} CUDA devices")
        model = DataParallel(base_model, device_ids=list(range(num_cuda_devices)))

    optimizer = build_optimizer(
        base_model,
        optimizer_name=config.optimizer_name,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        beta1=config.beta1,
        beta2=config.beta2,
    )

    micro_batch_size = config.batch_size // config.num_micro_batches
    train_batches = create_batches(train_dataset, micro_batch_size)
    if config.batch_order_seed is not None:
        train_batches = PermutedBatches(train_batches, config.batch_order_seed)

    val_batches = {
        split: create_batches(val_dataset[split], micro_batch_size)
        for split in val_dataset.keys()
    }
    steps_per_epoch = len(train_batches) // config.num_micro_batches
    total_steps = int(config.num_epochs * steps_per_epoch)
    keep_checkpoint_steps = normalize_checkpoint_steps(config.keep_checkpoint_steps)
    out_of_range_steps = [step for step in keep_checkpoint_steps if step > total_steps]
    if out_of_range_steps:
        raise ValueError(
            "keep_checkpoint_steps cannot exceed total_steps="
            f"{total_steps}: {out_of_range_steps}"
        )
    end_step = total_steps
    if config.stop_at_step is not None:
        end_step = min(total_steps, config.stop_at_step)
    if config.fork_from_step is not None and config.fork_from_step >= end_step:
        raise ValueError(f"fork_from_step must be below the final step {end_step}.")
    if config.warmup_steps is not None:
        warmup_steps = config.warmup_steps
    else:
        warmup_steps = int(total_steps * config.warmup_percent)
    eval_steps = (total_steps // config.num_evals) + 1

    scheduler = build_scheduler(
        optimizer,
        lr_schedule=config.lr_schedule,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        **({"min_lr_ratio": config.min_lr_ratio} if config.min_lr_ratio else {}),
        **({"lr_dip": config.lr_dip} if config.lr_dip is not None else {}),
    )
    config_metadata = asdict(replace(config, metric_loggers=()))
    config_metadata["metric_loggers"] = metric_loggers.describe()
    run_config = {
        **config_metadata,
        "model_name": base_model.config.name,
        "model_config": base_model.config.to_dict(),
        "batch_size": config.batch_size,
        "num_epochs": float(config.num_epochs),
        "optim_lr": config.learning_rate,
        "total_steps": total_steps,
        "end_step": end_step,
        "warmup_steps_applied": warmup_steps,
        "num_train_sequences": effective_train_sequences,
        "train_tokens": train_tokens,
        "parameter_count": parameter_count(base_model),
        "run_name": run_name,
        "qk_norm": config.qk_norm,
        "tie_word_embeddings": config.tie_word_embeddings,
        "deterministic": config.deterministic,
        "torch_compile": torch_compile_enabled,
        "torch_compile_mode": torch_compile_mode,
        "checkpointing": checkpointer.mode,
        **precision_config(config.precision),
    }
    run_config["model_dir"] = str(run_config["model_dir"])
    if run_config["init_checkpoint_path"] is not None:
        run_config["init_checkpoint_path"] = str(run_config["init_checkpoint_path"])
    run_config["wandb_tags"] = list(config.wandb_tags)
    checkpointer.set_metadata(run_config)

    start_step = checkpointer.restore_if_available(
        base_model,
        optimizer,
        scheduler,
        total_steps,
    )
    resumed = start_step > 0
    source_checkpoint = checkpointer.latest_checkpoint if resumed else None
    if not resumed and config.fork_from_run is not None:
        start_step, source_checkpoint = restore_fork(config, base_model, optimizer, scheduler)

    ema = None
    if config.ema_decays:
        ema_state = None
        if source_checkpoint is not None:
            ema_state = source_checkpoint.get("ema_state")
            if ema_state is not None and not all(
                f"{decay:g}" in ema_state for decay in config.ema_decays
            ):
                ema_state = None
            if ema_state is None and resumed:
                raise ValueError("Cannot resume: the checkpoint has no matching EMA state.")
        ema = ParameterEMA(list(base_model.parameters()), config.ema_decays, ema_state)

    training_loss = CausalLMTrainingLoss(model)
    if torch_compile_enabled:
        assert num_cuda_devices == 1
        print(f"Compiling training loss with torch.compile(mode={torch_compile_mode!r})")
        training_loss = torch.compile(training_loss, mode=torch_compile_mode)

    wandb_run_id = checkpointer.wandb_run_id()

    if config.wandb_online:
        import wandb

        if checkpointer.enabled and wandb_run_id is None:
            wandb_run_id = wandb.util.generate_id()
            checkpointer.write_wandb_run_state(
                wandb_run_id,
                wandb_entity=config.wandb_entity,
                wandb_project=config.wandb_project,
            )
        wandb_kwargs = {
            "entity": config.wandb_entity,
            "project": config.wandb_project,
            "name": run_name,
            "config": run_config,
            "tags": list(config.wandb_tags),
        }
        if wandb_run_id is not None:
            wandb_kwargs["id"] = wandb_run_id
            wandb_kwargs["resume"] = "allow"
        wandb_run = wandb.init(**wandb_kwargs)
        wandb_run_id = wandb_run.id
        if checkpointer.enabled:
            checkpointer.write_wandb_run_state(
                wandb_run_id,
                wandb_entity=config.wandb_entity,
                wandb_project=config.wandb_project,
            )
        wandb.define_metric("optimizer_step")
        wandb.define_metric("*", step_metric="optimizer_step")
        print(f"WANDB_RUN_URL={wandb_run.url}")

    metric_loggers.setup(
        LoggerContext(
            config=config,
            step=start_step,
            model=base_model,
            optimizer=optimizer,
            scheduler=scheduler,
            input_ids=None,
            loss=None,
            val_batches=val_batches,
        )
    )

    last_completed_step = start_step
    completed_training = start_step >= end_step
    interrupted = False
    training_loop_start_time = time.perf_counter()
    first_step_seconds = None
    try:
        with checkpointer.capture_interrupts() as interrupt_state:
            for step in range(start_step, end_step):
                step_start_time = time.perf_counter()
                current_loss_tensor = None
                for micro_batch in range(config.num_micro_batches):
                    batch_idx = (step * config.num_micro_batches + micro_batch) % len(
                        train_batches
                    )
                    input_ids = train_batches[batch_idx].to(device)
                    if torch_compile_enabled and hasattr(
                        torch.compiler,
                        "cudagraph_mark_step_begin",
                    ):
                        torch.compiler.cudagraph_mark_step_begin()
                    with autocast_context(config.precision, device=device):
                        loss = training_loss(input_ids)
                    loss = loss / config.num_micro_batches
                    detached_loss = loss.detach().clone()
                    if current_loss_tensor is None:
                        current_loss_tensor = detached_loss
                    else:
                        current_loss_tensor = current_loss_tensor + detached_loss
                    loss.backward()

                train_logger_stats = metric_loggers.collect(
                    AFTER_BACKWARD,
                    LoggerContext(
                        config=config,
                        step=step,
                        model=base_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        input_ids=input_ids,
                        loss=current_loss_tensor,
                        val_batches=val_batches,
                    ),
                )

                if config.grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        base_model.parameters(), max_norm=config.grad_norm
                    )

                optimizer.step()
                scheduler.step()
                if ema is not None:
                    ema.update()

                completed_steps = step + 1
                last_completed_step = completed_steps
                completed_training = completed_steps >= end_step

                is_eval_step = (step + 1) % eval_steps == 0 or step == end_step - 1
                current_loss = None
                if config.wandb_online or is_eval_step:
                    current_loss = current_loss_tensor.item()

                if config.wandb_online:
                    train_logger_stats.update(
                        metric_loggers.collect(
                            AFTER_TRAIN_STEP,
                            LoggerContext(
                                config=config,
                                step=step,
                                model=base_model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                input_ids=input_ids,
                                loss=current_loss_tensor,
                                val_batches=val_batches,
                            ),
                        )
                    )
                    wandb.log(
                        {
                            "optimizer_step": step,
                            "train_loss": current_loss,
                            "learning_rate": scheduler.get_last_lr()[0],
                            "step": step,
                            "progress": completed_steps / total_steps,
                            **train_logger_stats,
                        }
                    )

                optimizer.zero_grad()

                if is_eval_step:
                    stats = {}
                    elapsed_seconds = time.perf_counter() - training_loop_start_time
                    steps_this_run = completed_steps - start_step
                    steps_per_second = steps_this_run / max(elapsed_seconds, 1e-12)
                    eta_seconds = (end_step - completed_steps) / max(
                        steps_per_second,
                        1e-12,
                    )
                    print_str = (
                        f"Step {step + 1}/{end_step}, "
                        f"Progress: {(step + 1) / end_step:.2%}, "
                        f"Loss: {current_loss:.6f}, "
                        f"Elapsed: {format_duration(elapsed_seconds)}, "
                        f"ETA: {format_duration(eta_seconds)}"
                    )
                    for split in val_dataset.keys():
                        val_loss = evaluate(
                            model,
                            val_batches[split],
                            config=config,
                            device=device,
                        )
                        stats[f"{split}_loss"] = val_loss.item()
                        print_str += f", {split} Loss: {val_loss.item():.6f}"
                        if ema is not None:
                            ema_losses = ema.evaluate(
                                model, val_batches[split], config=config, device=device
                            )
                            for decay, ema_loss in ema_losses.items():
                                stats[f"{split}_ema{decay:g}_loss"] = ema_loss
                                print_str += f", {split} EMA{decay:g}: {ema_loss:.6f}"
                    stats.update(
                        metric_loggers.collect(
                            AFTER_EVAL,
                            LoggerContext(
                                config=config,
                                step=step,
                                model=base_model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                input_ids=input_ids,
                                loss=current_loss_tensor,
                                val_batches=val_batches,
                            ),
                        )
                    )
                    print(print_str)
                    if config.wandb_online:
                        wandb.log({"optimizer_step": step, **stats})

                checkpoint_saved = checkpointer.maybe_save_training_checkpoint(
                    completed_steps=completed_steps,
                    total_steps=end_step,
                    model=base_model,
                    optimizer=optimizer,
                    wandb_run_id=wandb_run_id,
                    force=interrupt_state.stop_requested,
                    extra_state=None if ema is None else {"ema_state": ema.state_dict()},
                )
                if first_step_seconds is None:
                    first_step_seconds = time.perf_counter() - step_start_time

                if interrupt_state.stop_requested:
                    interrupted = True
                    checkpointer.report_interrupted(
                        interrupt_state,
                        completed_steps=completed_steps,
                        total_steps=end_step,
                        checkpoint_saved=checkpoint_saved,
                    )
                    break
    finally:
        metric_loggers.close()
        checkpointer.close()

    checkpointer.save_final_model(
        base_model,
        completed_training=completed_training,
        last_completed_step=last_completed_step,
        total_steps=end_step,
    )

    train_end_time = time.perf_counter()
    first_step_seconds = 0.0 if first_step_seconds is None else first_step_seconds
    training_seconds = train_end_time - training_loop_start_time
    timing_stats = {
        "timing/startup_seconds": training_loop_start_time - train_start_time,
        "timing/first_step_seconds": first_step_seconds,
        "timing/training_seconds": training_seconds,
        "timing/steady_training_seconds": max(
            0.0,
            training_seconds - first_step_seconds,
        ),
        "timing/total_seconds": train_end_time - train_start_time,
        "timing/completed_steps": last_completed_step - start_step,
    }
    print(
        "Timing summary: "
        f"startup={timing_stats['timing/startup_seconds']:.2f}s, "
        f"first_step={timing_stats['timing/first_step_seconds']:.2f}s, "
        f"training={timing_stats['timing/training_seconds']:.2f}s, "
        f"total={timing_stats['timing/total_seconds']:.2f}s"
    )
    if config.wandb_online:
        wandb.run.summary.update(timing_stats)

    if config.wandb_online:
        wandb.finish(exit_code=255 if interrupted else 0)

    return base_model
