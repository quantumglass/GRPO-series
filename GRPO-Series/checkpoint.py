"""LoRA checkpoint save / resume helpers shared by training entrypoints."""

from __future__ import annotations

import shutil
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn

from lora import get_lora_state_dict, load_lora_state_dict

RUN_TIMESTAMP_FMT = "%Y%m%d-%H%M%S"


def format_run_timestamp(when: datetime | None = None) -> str:
    return (when or datetime.now()).strftime(RUN_TIMESTAMP_FMT)


def run_configs_dir(run_log_dir: str | Path) -> Path:
    return Path(run_log_dir) / "configs"


def save_run_config_snapshot(
    config_path: str | Path,
    run_log_dir: str | Path,
    session_timestamp: str | None = None,
    *,
    resume_from_ckpt: str | Path | None = None,
) -> Path:
    """Save a timestamped config snapshot under ``run_log_dir/configs/``.

    Each training invocation (including resume) writes a new file so prior
  configs are preserved, e.g. ``configs/config_r1_thinking_20260623-224822.yaml``.
    """
    config_path = Path(config_path)
    run_log_dir = Path(run_log_dir)
    session_timestamp = session_timestamp or format_run_timestamp()
    configs_dir = run_configs_dir(run_log_dir)
    configs_dir.mkdir(parents=True, exist_ok=True)

    snapshot_name = f"{config_path.stem}_{session_timestamp}{config_path.suffix}"
    snapshot_path = configs_dir / snapshot_name
    shutil.copy2(config_path, snapshot_path)

    if resume_from_ckpt is not None:
        with open(snapshot_path, "a", encoding="utf-8") as f:
            f.write(
                f"\n# [run] session={session_timestamp} "
                f"resume_from={Path(resume_from_ckpt).resolve()}\n"
            )
    else:
        with open(snapshot_path, "a", encoding="utf-8") as f:
            f.write(f"\n# [run] session={session_timestamp} fresh_run=true\n")

    latest_link = configs_dir / f"{config_path.stem}_latest{config_path.suffix}"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    try:
        latest_link.symlink_to(snapshot_path.name)
    except OSError:
        shutil.copy2(snapshot_path, latest_link)

    return snapshot_path


def checkpoint_filename(step: int, timestamp: str | None = None) -> str:
    ts = timestamp or format_run_timestamp()
    return f"ckpt_{step:06d}_{ts}.pt"


def checkpoint_path(
    ckpt_dir: str | Path,
    step: int,
    timestamp: str | None = None,
) -> Path:
    return Path(ckpt_dir) / checkpoint_filename(step, timestamp)


def save_training_checkpoint(
    ckpt_dir: str | Path,
    step: int,
    *,
    use_lora: bool,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    base_model_path: str | Path | None = None,
    lora_config: dict[str, Any] | None = None,
    saved_at: str | None = None,
    exgrpo_manager: Any | None = None,
    exgrpo_compact: bool = True,
) -> Path:
    """Persist a timestamped checkpoint and return its path."""
    saved_at = saved_at or format_run_timestamp()
    output_file = checkpoint_path(ckpt_dir, step, saved_at)
    exgrpo_state = None
    if exgrpo_manager is not None and getattr(exgrpo_manager, "config", None):
        if bool(getattr(exgrpo_manager.config, "enabled", False)):
            exgrpo_state = exgrpo_manager.state_dict(compact=exgrpo_compact)
    if use_lora:
        if base_model_path is None or lora_config is None:
            raise ValueError("base_model_path and lora_config required for LoRA save")
        checkpoint = build_lora_checkpoint_payload(
            step,
            model,
            optimizer,
            base_model_path=base_model_path,
            lora_config=lora_config,
            saved_at=saved_at,
            exgrpo_state=exgrpo_state,
        )
    else:
        checkpoint = {
            "checkpoint_type": "full_model",
            "step": step,
            "saved_at": saved_at,
            "state_dict": model.state_dict(),
        }
        if exgrpo_state is not None:
            checkpoint["exgrpo_state"] = exgrpo_state
    torch.save(checkpoint, output_file)
    return output_file


def add_resume_arguments(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--resume_lora_ckpt",
        type=str,
        default=None,
        help="Path to LoRA checkpoint (.pt). Overrides config training.resume_lora_ckpt.",
    )
    parser.add_argument(
        "--resume_log_dir",
        type=str,
        default=None,
        help=(
            "Log directory (TensorBoard + ckpt) to continue. "
            "Default: start a new timestamped run directory."
        ),
    )


def resolve_resume_paths(
    training_cfg: dict[str, Any],
    *,
    cli_resume_lora_ckpt: str | None = None,
    cli_resume_log_dir: str | None = None,
) -> tuple[Path | None, Path | None]:
    """
    Resolve resume checkpoint and log directory.

    Returns (checkpoint_path, log_dir_override). Both are None when starting a
    fresh run. ``log_dir_override`` is only set when user explicitly requests
    in-place logging via ``resume_log_dir``.
    """
    ckpt_raw = cli_resume_lora_ckpt or training_cfg.get("resume_lora_ckpt")
    if not ckpt_raw:
        return None, None

    ckpt_path = Path(ckpt_raw).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {ckpt_path}")

    log_raw = cli_resume_log_dir or training_cfg.get("resume_log_dir")
    if log_raw:
        log_dir = Path(log_raw).expanduser().resolve()
    else:
        log_dir = None
    return ckpt_path, log_dir


def load_lora_training_checkpoint(
    ckpt_path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    use_lora: bool,
    load_optimizer: bool = True,
    exgrpo_manager: Any | None = None,
    exgrpo_strict: bool = True,
) -> int:
    """Load LoRA weights and optionally optimizer / ExGRPO state. Returns global step."""
    if not use_lora:
        raise ValueError("resume_lora_ckpt is set but training.mode is not lora.")

    ckpt_path = Path(ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    lora_state_dict = checkpoint.get("lora_state_dict")
    if lora_state_dict is None:
        raise KeyError(f"No 'lora_state_dict' found in checkpoint: {ckpt_path}")

    load_lora_state_dict(model, lora_state_dict, strict=True)

    optimizer_loaded = False
    if load_optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        optimizer_loaded = True

    exgrpo_loaded = False
    if exgrpo_manager is not None:
        exgrpo_state = checkpoint.get("exgrpo_state")
        if exgrpo_state is not None:
            exgrpo_manager.load_state_dict(exgrpo_state, strict=exgrpo_strict)
            exgrpo_loaded = True

    resumed_step = int(checkpoint.get("step", 0))
    suffix_parts: list[str] = []
    if not optimizer_loaded:
        suffix_parts.append("optimizer state not loaded")
    if exgrpo_manager is not None:
        if exgrpo_loaded:
            suffix_parts.append("ExGRPO buffer restored")
        else:
            suffix_parts.append("ExGRPO buffer missing in checkpoint")
    suffix = f" ({', '.join(suffix_parts)})" if suffix_parts else ""
    print(f"Resumed LoRA checkpoint from {ckpt_path}, step={resumed_step}{suffix}")
    return resumed_step


def build_lora_checkpoint_payload(
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    base_model_path: str | Path,
    lora_config: dict[str, Any],
    saved_at: str | None = None,
    exgrpo_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "checkpoint_type": "lora_adapter",
        "step": step,
        "saved_at": saved_at or format_run_timestamp(),
        "base_model_path": str(base_model_path),
        "lora_config": lora_config,
        "lora_state_dict": get_lora_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if exgrpo_state is not None:
        payload["exgrpo_state"] = exgrpo_state
    return payload
