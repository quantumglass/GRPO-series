"""Model preset registry and path resolution for training pipelines."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# preset_name -> path relative to PROJECT_ROOT
MODEL_PRESETS: dict[str, str] = {
    "qwen2.5-3b-instruct": "models/Qwen2.5-3B-Instruct",
    "qwen2.5-math-7b": "models/Qwen2.5-Math-7B",
    "qwen2.5-math-7b-instruct": "models/Qwen2.5-Math-7B-Instruct",
}

MATH_BASE_MODEL_PRESETS: frozenset[str] = frozenset({"qwen2.5-math-7b"})

# preset_name -> log directory tag (used by train_exgrpo.py)
MODEL_RUN_TAGS: dict[str, str] = {
    "qwen2.5-3b-instruct": "exgrpo-3b",
    "qwen2.5-math-7b": "exgrpo-math7b-base",
    "qwen2.5-math-7b-instruct": "exgrpo-math7b",
}


@dataclass(frozen=True)
class ResolvedModelConfig:
    preset: str | None
    path: Path
    run_tag: str
    max_position_embeddings: int


def _infer_preset_from_path(path: Path) -> str | None:
    normalized = path.name.lower().replace("_", "-")
    for preset, rel_path in MODEL_PRESETS.items():
        if Path(rel_path).name.lower().replace("_", "-") == normalized:
            return preset
    return None


def _run_tag_for_preset(preset: str | None, path: Path) -> str:
    if preset is not None and preset in MODEL_RUN_TAGS:
        return MODEL_RUN_TAGS[preset]
    name = path.name.lower()
    if "math" in name and "7b" in name:
        return "exgrpo-math7b"
    if "7b" in name:
        return "exgrpo-7b"
    if "3b" in name:
        return "exgrpo-3b"
    return "exgrpo"


def _resolve_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    project_candidate = (PROJECT_ROOT / path).resolve()
    if project_candidate.exists():
        return project_candidate
    return path.resolve()


def load_model_arch_config(model_path: Path) -> dict:
    config_file = model_path / "config.json"
    if not config_file.is_file():
        raise FileNotFoundError(f"Missing model config: {config_file}")
    with open(config_file, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_model_config(model_cfg: dict) -> ResolvedModelConfig:
    preset = model_cfg.get("preset")
    if preset is not None:
        preset = str(preset).lower()
        if preset not in MODEL_PRESETS:
            known = ", ".join(sorted(MODEL_PRESETS))
            raise ValueError(f"Unknown model.preset '{preset}'. Known presets: {known}")
        path = _resolve_path(MODEL_PRESETS[preset])
    else:
        raw_path = model_cfg.get("pretrained_model_path")
        if not raw_path:
            raise ValueError("model.pretrained_model_path is required when model.preset is unset")
        path = _resolve_path(str(raw_path))
        preset = _infer_preset_from_path(path)

    if not path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {path}")

    arch_cfg = load_model_arch_config(path)
    max_pos = int(arch_cfg.get("max_position_embeddings", 32768))
    return ResolvedModelConfig(
        preset=preset,
        path=path,
        run_tag=_run_tag_for_preset(preset, path),
        max_position_embeddings=max_pos,
    )


def validate_training_seq_limits(
    model_cfg: ResolvedModelConfig,
    *,
    max_prompt_len: int,
    max_gen_len: int,
) -> None:
    """Warn when configured sequence budget may exceed model context."""
    budget = max_prompt_len + max_gen_len
    if budget > model_cfg.max_position_embeddings:
        raise ValueError(
            "max_prompt_len + max_gen_len exceeds model max_position_embeddings: "
            f"{max_prompt_len} + {max_gen_len} = {budget} > "
            f"{model_cfg.max_position_embeddings} ({model_cfg.path.name})"
        )
