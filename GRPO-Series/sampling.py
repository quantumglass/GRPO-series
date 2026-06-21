from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class SamplingConfig:
    """Token sampling settings for rollout / evaluation generation."""

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    do_sample: bool = True

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if not 0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {self.top_k}")


def sampling_config_from_dict(cfg: dict[str, Any] | None) -> SamplingConfig:
    if not cfg:
        return SamplingConfig()
    return SamplingConfig(
        temperature=float(cfg.get("temperature", 1.0)),
        top_p=float(cfg.get("top_p", 1.0)),
        top_k=int(cfg.get("top_k", 0)),
        do_sample=bool(cfg.get("do_sample", True)),
    )


def _apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0:
        return logits
    top_k = min(top_k, logits.size(-1))
    threshold = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def _apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_mask = cumulative_probs > top_p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False
    mask = sorted_mask.scatter(1, sorted_indices, sorted_mask)
    return logits.masked_fill(mask, float("-inf"))


def sample_next_token(logits: torch.Tensor, sampling: SamplingConfig) -> torch.Tensor:
    """
    Sample one token per row from logits of shape (batch, vocab_size).
    """
    logits = logits.float()

    if not sampling.do_sample or sampling.temperature == 0:
        return torch.argmax(logits, dim=-1)

    if sampling.temperature != 1.0:
        logits = logits / sampling.temperature

    logits = _apply_top_k(logits, sampling.top_k)
    logits = _apply_top_p(logits, sampling.top_p)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
