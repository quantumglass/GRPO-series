from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch
from torch import nn


@dataclass
class LoRAConfig:
    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    )


class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank r must be > 0, got {r}")

        self.base = base_layer
        self.r = r
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        in_features = base_layer.in_features
        out_features = base_layer.out_features
        device = base_layer.weight.device
        dtype = base_layer.weight.dtype
        self.lora_A = nn.Parameter(torch.zeros(r, in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.dropout(x) @ self.lora_A.t()
        lora_out = lora_out @ self.lora_B.t()
        return base_out + self.scaling * lora_out


def _get_parent_module(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def apply_lora_to_model(model: nn.Module, config: LoRAConfig) -> List[str]:
    replaced: List[str] = []
    target_names = set(config.target_modules)
    for module_name, module in list(model.named_modules()):
        short_name = module_name.split(".")[-1]
        if short_name not in target_names:
            continue
        if not isinstance(module, nn.Linear):
            continue
        parent, child_name = _get_parent_module(model, module_name)
        setattr(
            parent,
            child_name,
            LoRALinear(module, r=config.r, alpha=config.alpha, dropout=config.dropout),
        )
        replaced.append(module_name)
    return replaced


def freeze_non_lora_parameters(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = ("lora_A" in name) or ("lora_B" in name)


def get_trainable_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    return (p for p in model.parameters() if p.requires_grad)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return total, trainable


def get_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    lora_weights: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if ("lora_A" in name) or ("lora_B" in name):
            lora_weights[name] = param.detach().cpu()
    return lora_weights


def load_lora_state_dict(
    model: nn.Module, lora_state_dict: dict[str, torch.Tensor], strict: bool = True
) -> None:
    model_state = model.state_dict()
    missing_in_ckpt = []
    for name, tensor in lora_state_dict.items():
        if name in model_state:
            model_state[name].copy_(tensor.to(device=model_state[name].device))
    if strict:
        for name in model_state.keys():
            if (("lora_A" in name) or ("lora_B" in name)) and (name not in lora_state_dict):
                missing_in_ckpt.append(name)
        if missing_in_ckpt:
            raise KeyError(
                "Missing LoRA weights in checkpoint: " + ", ".join(missing_in_ckpt[:8])
            )
