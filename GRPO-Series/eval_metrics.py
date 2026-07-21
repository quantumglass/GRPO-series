"""Shared pass@k metrics for benchmark evaluation."""

from __future__ import annotations

from typing import Any


def pass_at_k_simple(sample_correct: list[bool], k: int) -> float:
    """
    Simple pass@k: 1.0 if at least one of the first k rollouts is correct.

    sample_correct: per-rollout correctness for one question (length = num_samples)
    k: pass@k cutoff (uses the first k rollouts)
    """
    if k <= 0:
        return 0.0
    return 1.0 if any(sample_correct[:k]) else 0.0


def is_pass_at_k_solved(sample_correct: list[bool], k: int) -> bool:
    """True when at least one of the first k rollouts is correct."""
    return pass_at_k_simple(sample_correct, k) > 0.0


def is_benchmark_enabled(eval_cfg: dict[str, Any], benchmark_name: str) -> bool:
    """Whether a benchmark should run, controlled by benchmark_overrides.<name>.enabled."""
    override_cfg = (eval_cfg.get("benchmark_overrides") or {}).get(benchmark_name, {}) or {}
    if not override_cfg:
        return True
    return bool(override_cfg.get("enabled", True))


def resolve_eval_benchmarks(
    benchmarks: dict[str, Any],
    eval_cfg: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    """Return enabled (name, cfg) pairs; skip those with enabled: false in benchmark_overrides."""
    enabled: list[tuple[str, dict[str, Any]]] = []
    skipped: list[str] = []
    for name, cfg in benchmarks.items():
        if is_benchmark_enabled(eval_cfg, name):
            enabled.append((name, cfg))
        else:
            skipped.append(name)
    if skipped:
        print(f"Skipping disabled benchmarks: {', '.join(skipped)}")
    if not enabled:
        raise ValueError(
            "No benchmarks enabled for evaluation. "
            "Set benchmark_overrides.<name>.enabled: true for at least one dataset."
        )
    return enabled


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


def compute_pass_at_k_from_samples(
    sample_results: list[list[bool]],
    pass_at_k: list[int],
    num_samples: int,
) -> dict[str, dict[str, float | int]]:
    """
    Aggregate per-question rollout results into simple pass@k metrics.

    For each k, a question passes if at least one of its first k rollouts is correct.
    """
    total = len(sample_results)
    metrics: dict[str, dict[str, float | int]] = {}
    for k in pass_at_k:
        passed = sum(1 for s in sample_results if is_pass_at_k_solved(s, k))
        metrics[str(k)] = {
            "rate": passed / total if total else 0.0,
            "passed": passed,
            "total": total,
        }
    if num_samples > 1:
        any_correct = sum(1 for s in sample_results if is_pass_at_k_solved(s, num_samples))
        metrics["_meta"] = {
            "num_samples": num_samples,
            "any_correct": any_correct,
            "total": total,
        }
    return metrics


def resolve_accuracy_fields(
    *,
    sample_results: list[list[bool]],
    num_samples: int,
    first_sample_correct: list[bool] | None,
) -> tuple[float, int]:
    """
    Return (accuracy, correct_count).

    num_samples == 1: plain accuracy (the single sample).
    num_samples > 1: first-sample accuracy when first_sample_correct is provided;
      otherwise derive from the first rollout in sample_results.
    """
    total = len(sample_results)
    if total == 0:
        return 0.0, 0
    if num_samples == 1:
        correct = sum(1 for s in sample_results if s and s[0])
        return correct / total, correct
    if first_sample_correct is not None:
        correct = sum(1 for ok in first_sample_correct if ok)
        return correct / total, correct
    correct = sum(1 for s in sample_results if s and s[0])
    return correct / total, correct


def build_benchmark_result(
    *,
    sample_results: list[list[bool]],
    num_samples: int,
    pass_at_k: list[int],
    first_sample_correct: list[bool] | None = None,
) -> dict[str, Any]:
    """Build result dict with first-sample accuracy plus pass@k block."""
    total = len(sample_results)
    pass_metrics = compute_pass_at_k_from_samples(sample_results, pass_at_k, num_samples)
    accuracy, correct = resolve_accuracy_fields(
        sample_results=sample_results,
        num_samples=num_samples,
        first_sample_correct=first_sample_correct,
    )

    result: dict[str, Any] = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "num_samples": num_samples,
        "pass_at_k": pass_metrics,
    }
    return result


def format_pass_at_k_summary(
    pass_at_k: dict[str, dict[str, float | int]],
    *,
    num_samples: int = 1,
) -> str:
    """
    Format pass@k lines for logging.

    Simple pass@k: for each k, rate = fraction of questions with at least one
    correct among the first k rollouts.
    """
    meta = pass_at_k.get("_meta", {})
    parts = []
    for k in sorted(
        (key for key in pass_at_k.keys() if not str(key).startswith("_")),
        key=lambda x: int(x),
    ):
        item = pass_at_k[k]
        rate = float(item["rate"])
        passed = int(item["passed"])
        total = int(item["total"])
        parts.append(f"pass@{k}={rate:.4f} ({passed}/{total})")
    if num_samples > 1 and meta:
        parts.append(f"[n={int(meta['num_samples'])} samples/q]")
    return ", ".join(parts)
