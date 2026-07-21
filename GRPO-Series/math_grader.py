"""统一数学答案判题入口：legacy（仓库原逻辑）与 SimKO（math_equal）。"""

from __future__ import annotations

from typing import Iterable

from benchmark_task import answer_matches, extract_pred_answer
from deepscaler_task import _answers_match as legacy_answers_match
from simko_grader import (
    GraderMode,
    answers_equal_simko,
    extract_answer_simko,
    parse_grader_mode,
)


def iter_math_answer_candidates(
    response: str,
    *,
    end_token: str | None = None,
    dataset_name: str = "math500",
) -> list[str]:
    """收集用于判题的答案候选（legacy 多路径 + SimKO 主提取）。"""
    from r1_thinking_reward import _strip_end_token, iter_r1_answer_candidates

    text = _strip_end_token(response, end_token)
    candidates: list[str] = []
    seen: set[str] = set()

    def _add(raw: str | None) -> None:
        if raw is None:
            return
        s = str(raw).strip()
        if s and s not in seen:
            seen.add(s)
            candidates.append(s)

    simko_primary = extract_answer_simko(text)
    _add(simko_primary)
    for cand in iter_r1_answer_candidates(
        response, dataset_name=dataset_name, end_token=end_token
    ):
        _add(cand)
    fallback = extract_pred_answer(text, dataset_name=dataset_name)
    _add(fallback)
    return candidates


def legacy_answers_equal(pred: str | None, gt: str, *, dataset_name: str = "math500") -> bool:
    if pred is None:
        return False
    if legacy_answers_match(pred, gt):
        return True
    return answer_matches(pred, gt, dataset_name=dataset_name)


def math_answers_equal(
    pred: str | None,
    gt: str,
    *,
    grader_mode: GraderMode = "legacy",
    dataset_name: str = "math500",
    simko_timeout: bool = True,
) -> bool:
    if grader_mode == "simko":
        return answers_equal_simko(pred, gt, timeout=simko_timeout)
    return legacy_answers_equal(pred, gt, dataset_name=dataset_name)


def is_math_response_correct(
    response: str,
    ground_truth: str,
    *,
    grader_mode: GraderMode = "legacy",
    end_token: str | None = None,
    dataset_name: str = "math500",
    simko_timeout: bool = True,
) -> bool:
    """对 response 中所有候选答案尝试匹配 GT。"""
    gt = str(ground_truth)
    if grader_mode == "simko":
        for cand in iter_math_answer_candidates(
            response, end_token=end_token, dataset_name=dataset_name
        ):
            if answers_equal_simko(cand, gt, timeout=simko_timeout):
                return True
        return False

    from r1_thinking_reward import is_r1_benchmark_answer_correct

    return is_r1_benchmark_answer_correct(
        response,
        gt,
        dataset_name=dataset_name,
        end_token=end_token,
    )


def audit_candidates(
    response: str,
    ground_truth: str,
    *,
    grader_mode: GraderMode = "legacy",
    end_token: str | None = None,
) -> Iterable[tuple[str, bool]]:
    """调试：返回 (candidate, matched) 列表。"""
    for cand in iter_math_answer_candidates(response, end_token=end_token):
        yield cand, math_answers_equal(
            cand, ground_truth, grader_mode=grader_mode
        )


__all__ = [
    "is_math_response_correct",
    "iter_math_answer_candidates",
    "legacy_answers_equal",
    "math_answers_equal",
    "parse_grader_mode",
]
