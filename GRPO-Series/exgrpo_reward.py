"""
ExGRPO 训练奖励：accuracy + format，不含 thinking 长度 bonus。

总奖励 = w_accuracy * acc + w_format * format
评测 success_rate 始终按答案是否正确（1/0）统计。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from r1_thinking_reward import (
    AccuracyMode,
    GraderMode,
    R1RewardConfig as _FormatConfig,
    _parse_accuracy_mode,
    _parse_grader_mode,
    compute_accuracy_reward,
    compute_format_reward,
    is_accuracy_correct,
)


@dataclass(frozen=True)
class ExGRPORewardConfig:
    w_accuracy: float = 1.0
    w_format: float = 0.1
    accuracy_mode: AccuracyMode = "signed"
    partial_format_credit: bool = False
    grader_mode: GraderMode = "legacy"

    def format_cfg(self) -> _FormatConfig:
        """供 format 检查复用的最小 R1 配置（thinking 权重恒为 0）。"""
        return _FormatConfig(
            w_thinking=0.0,
            w_accuracy=self.w_accuracy,
            w_format=self.w_format,
            accuracy_mode=self.accuracy_mode,
            partial_format_credit=self.partial_format_credit,
        )


def compute_exgrpo_reward(
    response: str,
    dataset_kind: str,
    cfg: ExGRPORewardConfig,
    ground_truth: str = "",
    numbers: list[int] | None = None,
    target: int | None = None,
    end_token: str | None = None,
) -> dict[str, Any]:
    format_cfg = cfg.format_cfg()
    accuracy = compute_accuracy_reward(
        response=response,
        dataset_kind=dataset_kind,
        ground_truth=ground_truth,
        numbers=numbers,
        target=target,
        end_token=end_token,
        accuracy_mode=cfg.accuracy_mode,
        grader_mode=cfg.grader_mode,
    )
    format_score = compute_format_reward(response, end_token, format_cfg)
    total = cfg.w_accuracy * accuracy + cfg.w_format * format_score
    return {
        "reward": total,
        "reward_info": {
            "accuracy_reward": accuracy,
            "format_reward": format_score,
            "answer_reward": accuracy,
            "accuracy_correct": float(is_accuracy_correct(accuracy)),
            "accuracy_mode": cfg.accuracy_mode,
            "grader_mode": cfg.grader_mode,
        },
    }


def build_exgrpo_reward_function(
    dataset_kind: str,
    cfg: ExGRPORewardConfig | None = None,
) -> Callable[..., dict[str, Any]]:
    reward_cfg = cfg or ExGRPORewardConfig()

    def reward_function(
        response: str,
        ground_truth: str = "",
        numbers: list[int] | None = None,
        target: int | None = None,
        end_token: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        del kwargs
        return compute_exgrpo_reward(
            response=response,
            dataset_kind=dataset_kind,
            cfg=reward_cfg,
            ground_truth=ground_truth,
            numbers=numbers,
            target=target,
            end_token=end_token,
        )

    return reward_function


def exgrpo_reward_config_from_dict(raw: dict[str, Any] | None) -> ExGRPORewardConfig:
    raw = raw or {}
    return ExGRPORewardConfig(
        w_accuracy=float(raw.get("w_accuracy", 1.0)),
        w_format=float(raw.get("w_format", 0.1)),
        accuracy_mode=_parse_accuracy_mode(raw.get("accuracy_mode", "signed")),
        partial_format_credit=bool(raw.get("partial_format_credit", False)),
        grader_mode=_parse_grader_mode(raw.get("grader_mode", "legacy")),
    )
