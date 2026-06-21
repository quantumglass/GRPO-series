"""LoRA checkpoint save / resume helpers shared by training entrypoints."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import torch
from torch import nn

from lora import get_lora_state_dict, load_lora_state_dict


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
            "Default: parent directory of resume_lora_ckpt."
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

    Returns (checkpoint_path, log_dir). Both are None when starting a fresh run.
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
        log_dir = ckpt_path.parent
    return ckpt_path, log_dir


def load_lora_training_checkpoint(
    ckpt_path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    use_lora: bool,
    load_optimizer: bool = True,
) -> int:
    """Load LoRA weights and optionally optimizer state. Returns global step."""
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

    resumed_step = int(checkpoint.get("step", 0))
    suffix = "" if optimizer_loaded else " (optimizer state not loaded)"
    print(f"Resumed LoRA checkpoint from {ckpt_path}, step={resumed_step}{suffix}")
    return resumed_step


def build_lora_checkpoint_payload(
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    base_model_path: str | Path,
    lora_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "checkpoint_type": "lora_adapter",
        "step": step,
        "base_model_path": str(base_model_path),
        "lora_config": lora_config,
        "lora_state_dict": get_lora_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
    }
