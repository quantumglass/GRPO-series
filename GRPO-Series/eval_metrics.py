"""Shared pass@k metrics for benchmark evaluation."""

from __future__ import annotations

import math
from typing import Any


def pass_at_k_unbiased(num_samples: int, num_correct: int, k: int) -> float:
    """
    Unbiased pass@k estimator (Chen et al., HumanEval).

    num_samples: total independent samples n for one problem
    num_correct: how many of those samples are correct (c)
    k: pass@k cutoff
    """
    if k <= 0:
        return 0.0
    if num_correct <= 0:
        return 0.0
    if num_samples - num_correct < k:
        return 1.0
    if k > num_samples:
        return 0.0
    return 1.0 - math.comb(num_samples - num_correct, k) / math.comb(num_samples, k)


def resolve_pass_at_k_config(
    eval_cfg: dict[str, Any],
    benchmark_name: str | None = None,
) -> tuple[list[int], int]:
    """
    Resolve pass@k settings from eval config (with optional benchmark override).

    Returns:
        pass_at_k: sorted unique k values to report
        num_samples: number of generations per question
    """
    override_cfg: dict[str, Any] = {}
    if benchmark_name:
        override_cfg = (eval_cfg.get("benchmark_overrides") or {}).get(benchmark_name, {}) or {}

    raw_pass_at_k = override_cfg.get("pass_at_k", eval_cfg.get("pass_at_k"))
    if raw_pass_at_k is None:
        pass_at_k = [1]
    else:
        pass_at_k = sorted({int(k) for k in raw_pass_at_k if int(k) > 0})
        if not pass_at_k:
            pass_at_k = [1]

    num_samples = override_cfg.get("num_samples", eval_cfg.get("num_samples"))
    if num_samples is None:
        num_samples = max(pass_at_k)
    else:
        num_samples = int(num_samples)

    max_k = max(pass_at_k)
    if num_samples < max_k:
        raise ValueError(
            f"num_samples={num_samples} must be >= max(pass_at_k)={max_k} "
            f"(benchmark={benchmark_name or 'default'})"
        )
    return pass_at_k, num_samples


def compute_pass_at_k_from_counts(
    correct_counts: list[int],
    num_samples: int,
    pass_at_k: list[int],
) -> dict[str, dict[str, float | int]]:
    """Aggregate per-question correct-counts into pass@k metrics."""
    total = len(correct_counts)
    metrics: dict[str, dict[str, float | int]] = {}
    for k in pass_at_k:
        passed = 0
        rate_sum = 0.0
        for c in correct_counts:
            rate = pass_at_k_unbiased(num_samples, c, k)
            rate_sum += rate
            if rate >= 1.0 - 1e-12:
                passed += 1
        metrics[str(k)] = {
            "rate": rate_sum / total if total else 0.0,
            "passed": passed,
            "total": total,
        }
    return metrics


def build_benchmark_result(
    *,
    correct_counts: list[int],
    num_samples: int,
    pass_at_k: list[int],
) -> dict[str, Any]:
    """Build result dict with legacy accuracy fields plus pass@k block."""
    total = len(correct_counts)
    pass_metrics = compute_pass_at_k_from_counts(correct_counts, num_samples, pass_at_k)
    pass_at_1 = pass_metrics.get("1", {}).get("rate", 0.0)
    correct_at_1 = sum(1 for c in correct_counts if c > 0) if num_samples == 1 else int(
        round(pass_at_1 * total)
    )

    # When num_samples==1, pass@1 equals plain accuracy.
    if num_samples == 1:
        correct_at_1 = sum(1 for c in correct_counts if c > 0)

    result: dict[str, Any] = {
        "accuracy": pass_metrics["1"]["rate"] if "1" in pass_metrics else (correct_at_1 / total if total else 0.0),
        "correct": correct_at_1,
        "total": total,
        "num_samples": num_samples,
        "pass_at_k": pass_metrics,
    }
    return result


def format_pass_at_k_summary(pass_at_k: dict[str, dict[str, float | int]]) -> str:
    parts = []
    for k in sorted(pass_at_k.keys(), key=lambda x: int(x)):
        item = pass_at_k[k]
        parts.append(
            f"pass@{k}={item['rate']:.4f} ({item['passed']}/{item['total']})"
        )
    return ", ".join(parts)
