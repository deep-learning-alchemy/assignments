from pathlib import Path, PurePosixPath

import modal
from modal.config import config as modal_config

from utils import (
    REPO_ROOT,
    config_str,
    timestamped_modal_app_name as _timestamped_modal_app_name,
)


DEFAULT_SHARED_DATA_ENVIRONMENT = "cs312-shared-data"
DEFAULT_SHARED_DATA_VOLUME_NAME = "hard-dl-dclm-v1"
DEFAULT_WANDB_SECRET_NAME = "dl-alchemy-wandb"


def _configured_modal_environment():
    environment_name = (
        config_str(
            "CONFIG_MODAL_ENVIRONMENT",
            "DL_ALCHEMY_MODAL_ENVIRONMENT",
            "MODAL_ENVIRONMENT",
        )
        or modal_config.get("environment")
    )
    if environment_name:
        return environment_name
    raise RuntimeError(
        "Set CONFIG_MODAL_ENVIRONMENT near the top of utils.py or configure a Modal default "
        "environment with `uv run modal config set-environment ENVIRONMENT_NAME`."
    )


MODAL_ENVIRONMENT = _configured_modal_environment()
MODAL_SHARED_DATA_ENVIRONMENT = config_str(
    "MODAL_SHARED_DATA_ENVIRONMENT",
    "DL_ALCHEMY_MODAL_SHARED_DATA_ENVIRONMENT",
    default=DEFAULT_SHARED_DATA_ENVIRONMENT,
)
APP_NAME = config_str(
    "CONFIG_MODAL_APP_NAME",
    "DL_ALCHEMY_MODAL_APP_NAME",
    default="dl_alchemy",
)


def timestamped_modal_app_name(base_name: str = APP_NAME) -> str:
    return _timestamped_modal_app_name(base_name)


VOLUME_NAME = config_str(
    "CONFIG_MODAL_VOLUME_NAME",
    "DL_ALCHEMY_MODAL_VOLUME_NAME",
    default=f"volume-{APP_NAME}",
)
SHARED_DATA_VOLUME_NAME = config_str(
    "MODAL_SHARED_DATA_VOLUME_NAME",
    "DL_ALCHEMY_MODAL_SHARED_DATA_VOLUME_NAME",
    default=DEFAULT_SHARED_DATA_VOLUME_NAME,
)
WANDB_SECRET_NAME = config_str(
    "CONFIG_MODAL_WANDB_SECRET",
    "DL_ALCHEMY_MODAL_WANDB_SECRET",
    default=DEFAULT_WANDB_SECRET_NAME,
)
EXTRA_SECRET_NAMES = config_str(
    "MODAL_SECRETS",
    "DL_ALCHEMY_MODAL_SECRETS",
    default="",
)
WANDB_ENTITY = config_str("CONFIG_WANDB_ENTITY", "WANDB_ENTITY", default="")
WANDB_PROJECT = config_str("CONFIG_WANDB_PROJECT", "WANDB_PROJECT", default="assignments")
MODAL_REPO_DIR = PurePosixPath("/root/assignments")
MODAL_DATA_DIR = PurePosixPath("/root/data")
MODAL_USER_DATASETS_DIR = MODAL_DATA_DIR / "datasets"
MODAL_MODEL_DIR = MODAL_DATA_DIR / "ckpts"
MODAL_SHARED_DATA_DIR = PurePosixPath("/root/shared_data")
MODAL_SHARED_DATASETS_DIR = MODAL_SHARED_DATA_DIR / "datasets"

IGNORED_SOURCE_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "cache",
    "ckpts",
    "data",
    "plots",
    "results",
    "slurmjobs",
    "temp_scripts",
    "wandb",
}
IGNORED_SOURCE_NAMES = {
    ".env",
    ".wandb.secret.env",
}

app = modal.App(APP_NAME)
user_volume = modal.Volume.from_name(
    VOLUME_NAME,
    create_if_missing=True,
    version=2,
)
shared_data_volume = modal.Volume.from_name(
    SHARED_DATA_VOLUME_NAME,
    create_if_missing=False,
    environment_name=MODAL_SHARED_DATA_ENVIRONMENT,
    version=2,
)


def ignore_source_path(path: Path) -> bool:
    if path.name in IGNORED_SOURCE_NAMES:
        return True
    if any(part in IGNORED_SOURCE_PARTS for part in path.parts):
        return True
    return path.suffix in {".pyc", ".pyo"}


def build_image(*, include_tests: bool = False) -> modal.Image:
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("wget", "gzip")
        .uv_sync(uv_project_dir=str(REPO_ROOT))
        .workdir(str(MODAL_REPO_DIR))
        .env(
            {
                "DL_ALCHEMY_DATA_DIR": str(MODAL_SHARED_DATASETS_DIR),
                "DL_ALCHEMY_SHARED_DATA_DIR": str(MODAL_SHARED_DATASETS_DIR),
                "DL_ALCHEMY_USER_DATA_DIR": str(MODAL_USER_DATASETS_DIR),
                "DL_ALCHEMY_MODAL_REPO_DIR": str(MODAL_REPO_DIR),
                "DL_ALCHEMY_MODAL_MODEL_DIR": str(MODAL_MODEL_DIR),
                "DL_ALCHEMY_MODAL_DATA_DIR": str(MODAL_DATA_DIR),
                "DL_ALCHEMY_MODAL_SHARED_DATA_DIR": str(MODAL_SHARED_DATA_DIR),
                "DL_ALCHEMY_MODAL_ENVIRONMENT": MODAL_ENVIRONMENT,
                "DL_ALCHEMY_MODAL_SHARED_DATA_ENVIRONMENT": MODAL_SHARED_DATA_ENVIRONMENT,
                "DL_ALCHEMY_MODAL_APP_NAME": APP_NAME,
                "DL_ALCHEMY_MODAL_VOLUME_NAME": VOLUME_NAME,
                "DL_ALCHEMY_MODAL_SHARED_DATA_VOLUME_NAME": SHARED_DATA_VOLUME_NAME,
                "DL_ALCHEMY_MODAL_WANDB_SECRET": WANDB_SECRET_NAME,
                "DL_ALCHEMY_MODAL_SECRETS": EXTRA_SECRET_NAMES,
                "WANDB_ENTITY": WANDB_ENTITY,
                "WANDB_PROJECT": WANDB_PROJECT,
                "TMPDIR": "/tmp",
                "TEMP": "/tmp",
                "TMP": "/tmp",
                "TORCHINDUCTOR_CACHE_DIR": "/tmp/torchinductor",
                "TRITON_CACHE_DIR": "/tmp/triton",
                "WANDB_DIR": "/tmp/wandb",
                "WANDB__SERVICE_WAIT": "300",
                "WANDB_SERVICE_WAIT": "300",
            }
        )
        .add_local_dir(
            REPO_ROOT,
            remote_path=str(MODAL_REPO_DIR),
            ignore=ignore_source_path,
        )
    )
    if include_tests and Path("tests").is_dir():
        image = image.add_local_dir("tests", remote_path="/root/tests")
    return image


USER_VOLUME_MOUNTS: dict[str | PurePosixPath, object] = {
    MODAL_DATA_DIR: user_volume,
}
VOLUME_MOUNTS: dict[str | PurePosixPath, object] = {
    MODAL_DATA_DIR: user_volume,
    MODAL_SHARED_DATA_DIR: shared_data_volume.read_only(),
}


def _secret_names(*, include_wandb: bool) -> tuple[str, ...]:
    secret_names = []
    if include_wandb and WANDB_SECRET_NAME.strip():
        secret_names.append(WANDB_SECRET_NAME.strip())
    secret_names.extend(
        name.strip()
        for name in EXTRA_SECRET_NAMES.split(",")
        if name.strip()
    )
    return tuple(dict.fromkeys(secret_names))


def secrets(*, include_wandb: bool = False) -> list[modal.Secret]:
    return [
        modal.Secret.from_name(name)
        for name in _secret_names(include_wandb=include_wandb)
    ]
