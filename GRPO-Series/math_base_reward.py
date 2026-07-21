"""Qwen2.5-Math-7B base 模型的 ExGRPO 奖励（\\boxed{} 格式）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from benchmark_task import extract_boxed_answer
from exgrpo_reward import ExGRPORewardConfig, exgrpo_reward_config_from_dict
from r1_thinking_reward import (
    AccuracyMode,
    compute_accuracy_reward,
    is_accuracy_correct,
)


def _strip_end_token(response: str, end_token: str | None) -> str:
    if end_token and response.endswith(end_token):
        return response[: -len(end_token)]
    return response


def compute_math_base_format_reward(
    response: str,
    end_token: str | None,
    *,
    partial_format_credit: bool,
) -> float:
    """Math 格式分：含可解析 \\boxed{} 得 1.0；可选对部分推理给 0.5。"""
    text = _strip_end_token(response, end_token).strip()
    if not text:
        return 0.0
    if extract_boxed_answer(text):
        return 1.0
    if partial_format_credit and len(text) >= 40:
        return 0.5
    return 0.0


def compute_math_base_reward(
    response: str,
    dataset_kind: str,
    cfg: ExGRPORewardConfig,
    ground_truth: str = "",
    numbers: list[int] | None = None,
    target: int | None = None,
    end_token: str | None = None,
) -> dict[str, Any]:
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
    format_score = compute_math_base_format_reward(
        response,
        end_token,
        partial_format_credit=cfg.partial_format_credit,
    )
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


def build_math_base_reward_function(
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
        return compute_math_base_reward(
            response=response,
            dataset_kind=dataset_kind,
            cfg=reward_cfg,
            ground_truth=ground_truth,
            numbers=numbers,
            target=target,
            end_token=end_token,
        )

    return reward_function
