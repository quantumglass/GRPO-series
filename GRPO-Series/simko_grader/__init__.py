"""
SimKO 风格 MATH 判题（extract_answer_math + math_equal）。

来源: SimKO / verl.utils.reward_score（Hendrycks MATH equivalence + sympy）
"""

from __future__ import annotations

from typing import Literal

from simko_grader.extract_utils import extract_answer_math
from simko_grader.grader import math_equal

GraderMode = Literal["legacy", "simko"]
GRADER_MODES: tuple[GraderMode, ...] = ("legacy", "simko")


def parse_grader_mode(raw: str | None) -> GraderMode:
    if raw is None:
        return "legacy"
    mode = str(raw).lower().strip()
    if mode in GRADER_MODES:
        return mode  # type: ignore[return-value]
    raise ValueError(f"grader_mode must be one of {GRADER_MODES}, got: {raw!r}")


def ensure_simko_dependencies() -> None:
    """SimKO 判题依赖 sympy + regex + latex2sympy2。"""
    missing: list[str] = []
    try:
        import regex  # noqa: F401
    except ImportError:
        missing.append("regex")
    try:
        import latex2sympy2  # noqa: F401
    except ImportError:
        missing.append("latex2sympy2")
    try:
        import sympy  # noqa: F401
    except ImportError:
        missing.append("sympy")
    if missing:
        raise ImportError(
            "SimKO grader requires: "
            + ", ".join(missing)
            + ". Install with: pip install regex latex2sympy2 sympy"
        )


def extract_answer_simko(text: str) -> str:
    """从模型输出提取答案（SimKO extract_answer_math）。"""
    return extract_answer_math(text)


def answers_equal_simko(pred: str | None, gt: str, *, timeout: bool = True) -> bool:
    """SimKO math_equal 等价判定。

    优先走无 multiprocessing 的路径，避免短超时在负载下误杀；
    仅在需要时再启用带 timeout 的符号判定。
    """
    if pred is None:
        return False
    pred_s = str(pred).strip()
    gt_s = str(gt).strip()
    if not pred_s or not gt_s:
        return pred_s == gt_s
    ensure_simko_dependencies()
    # Deterministic first (no process pool)
    if bool(math_equal(pred_s, gt_s, timeout=False)):
        return True
    if not timeout:
        return False
    return bool(math_equal(pred_s, gt_s, timeout=True))


__all__ = [
    "GRADER_MODES",
    "GraderMode",
    "answers_equal_simko",
    "ensure_simko_dependencies",
    "extract_answer_simko",
    "extract_answer_math",
    "math_equal",
    "parse_grader_mode",
]
