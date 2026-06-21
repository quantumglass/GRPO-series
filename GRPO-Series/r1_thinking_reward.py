"""
DeepSeek-R1 风格的 thinking 奖励函数。

设计参考 DeepSeek-R1 论文：
  - Accuracy reward：答案正确性（主信号，0/1）
  - Format reward：是否遵循 <think>...</think> + <answer>...</answer>
  - Thinking length bonus：在格式基本满足时，对 thinking 长度做温和线性鼓励（非 R1 原文项，用于缓解 instruct 模型输出过短）

总奖励 = w_acc * accuracy + w_fmt * format + w_len * length_bonus
评测指标 success_rate 仅使用 accuracy，与 mean_reward 解耦。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from benchmark_task import answer_matches, extract_pred_answer
from deepscaler_task import (
    _answers_match as deepscaler_answers_match,
    extract_answer as deepscaler_extract_answer,
)


@dataclass(frozen=True)
class R1RewardConfig:
  """奖励权重与 thinking 长度鼓励参数。"""

  w_accuracy: float = 1.0
  w_format: float = 0.2
  w_length: float = 0.08
  # thinking 字符数低于 min_chars 不给长度分；到 target_chars 给满 1.0
  min_thinking_chars: int = 80
  target_thinking_chars: int = 512
  # 格式分：R1 为二元；设为 True 时按 think/answer 标签给部分分
  partial_format_credit: bool = False


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
ANSWER_OPEN = "<answer>"
ANSWER_CLOSE = "</answer>"

THINK_BLOCK_RE = re.compile(
  rf"{re.escape(THINK_OPEN)}(.*?){re.escape(THINK_CLOSE)}",
  re.DOTALL,
)
ANSWER_BLOCK_RE = re.compile(
  rf"{re.escape(ANSWER_OPEN)}(.*?){re.escape(ANSWER_CLOSE)}",
  re.DOTALL,
)
FULL_FORMAT_RE = re.compile(
  rf"^{re.escape(THINK_OPEN)}.*?{re.escape(THINK_CLOSE)}\s*\n{re.escape(ANSWER_OPEN)}.*?{re.escape(ANSWER_CLOSE)}$",
  re.DOTALL,
)


def _strip_end_token(response: str, end_token: str | None) -> str:
  if end_token and response.endswith(end_token):
    return response[: -len(end_token)]
  return response


def _normalize_for_format_check(response: str, end_token: str | None) -> str:
  text = _strip_end_token(response, end_token)
  # 与 deepscaler/gsm8k 一致：prefix 已预填 <think>
  if not text.startswith(THINK_OPEN):
    text = THINK_OPEN + text
  return text


def extract_thinking_text(response: str, end_token: str | None = None) -> str:
  text = _normalize_for_format_check(response, end_token)
  match = THINK_BLOCK_RE.search(text)
  if match:
    return match.group(1).strip()
  # 未闭合时：取 open tag 之后到 answer tag 或文末
  if THINK_OPEN in text:
    after = text.split(THINK_OPEN, 1)[1]
    for marker in (THINK_CLOSE, ANSWER_OPEN):
      if marker in after:
        after = after.split(marker, 1)[0]
    return after.strip()
  return ""


def compute_format_reward(response: str, end_token: str | None, cfg: R1RewardConfig) -> float:
  text = _normalize_for_format_check(response, end_token)
  has_think = bool(THINK_BLOCK_RE.search(text))
  has_answer = bool(ANSWER_BLOCK_RE.search(text))
  full = bool(FULL_FORMAT_RE.match(text.strip()))

  if cfg.partial_format_credit:
    score = 0.0
    if has_think:
      score += 0.2
    if has_answer:
      score += 0.5
    if full:
      score = 1.0
    return min(score, 1.0)

  return 1.0 if full else 0.0


def compute_thinking_length_bonus(
  response: str,
  end_token: str | None,
  cfg: R1RewardConfig,
) -> float:
  """温和长度鼓励：仅在有 thinking 内容时生效，线性爬坡至 target。"""
  thinking = extract_thinking_text(response, end_token)
  if not thinking:
    return 0.0
  n = len(thinking)
  if n < cfg.min_thinking_chars:
    return 0.0
  if n >= cfg.target_thinking_chars:
    return 1.0
  span = max(cfg.target_thinking_chars - cfg.min_thinking_chars, 1)
  return (n - cfg.min_thinking_chars) / span


def _countdown_answer_correct(response: str, numbers: list[int], target: int) -> bool:
  tag_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
  if tag_match:
    expr = tag_match.group(1).strip()
  else:
    lines = [ln.strip() for ln in response.splitlines() if ln.strip()]
    expr = lines[-1] if lines else ""
  if not expr:
    return False
  if not re.match(r"^[0-9+\-*/() ]+$", expr):
    return False
  used_numbers = [int(n) for n in re.findall(r"\d+", expr)]
  if sorted(used_numbers) != sorted(numbers):
    return False
  try:
    result = eval(expr, {"__builtins__": None}, {})
    return abs(float(result) - float(target)) < 1e-5
  except Exception:
    return False


def compute_accuracy_reward(
  response: str,
  dataset_kind: str,
  ground_truth: str = "",
  numbers: list[int] | None = None,
  target: int | None = None,
  end_token: str | None = None,
) -> float:
  text = _strip_end_token(response, end_token)
  if dataset_kind == "countdown":
    ok = bool(
      numbers is not None
      and target is not None
      and _countdown_answer_correct(text, numbers, target)
    )
    return 1.0 if ok else 0.0

  pred = deepscaler_extract_answer(text)
  if pred is None:
    pred = extract_pred_answer(text, dataset_name="math500")
  if pred is None:
    return 0.0
  if deepscaler_answers_match(pred, str(ground_truth)):
    return 1.0
  if answer_matches(pred, str(ground_truth), dataset_name="math500"):
    return 1.0
  return 0.0


def compute_r1_reward(
  response: str,
  dataset_kind: str,
  cfg: R1RewardConfig,
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
  )
  format_score = compute_format_reward(response, end_token, cfg)
  length_bonus = compute_thinking_length_bonus(response, end_token, cfg)
  # 无格式时不给长度分，避免模型刷无结构长文本
  if format_score <= 0.0 and not cfg.partial_format_credit:
    length_bonus = 0.0
  elif format_score < 0.2 and cfg.partial_format_credit:
    length_bonus = 0.0

  total = (
    cfg.w_accuracy * accuracy
    + cfg.w_format * format_score
    + cfg.w_length * length_bonus
  )
  thinking_chars = len(extract_thinking_text(response, end_token))
  return {
    "reward": total,
    "reward_info": {
      "accuracy_reward": accuracy,
      "format_reward": format_score,
      "length_bonus": length_bonus,
      "answer_reward": accuracy,
      "thinking_chars": float(thinking_chars),
    },
  }


def build_r1_reward_function(
  dataset_kind: str,
  cfg: R1RewardConfig | None = None,
) -> Callable[..., dict[str, Any]]:
  reward_cfg = cfg or R1RewardConfig()

  def reward_function(
    response: str,
    ground_truth: str = "",
    numbers: list[int] | None = None,
    target: int | None = None,
    end_token: str | None = None,
    **kwargs,
  ) -> dict[str, Any]:
    del kwargs
    return compute_r1_reward(
      response=response,
      dataset_kind=dataset_kind,
      cfg=reward_cfg,
      ground_truth=ground_truth,
      numbers=numbers,
      target=target,
      end_token=end_token,
    )

  return reward_function


def r1_reward_config_from_dict(raw: dict[str, Any] | None) -> R1RewardConfig:
  raw = raw or {}
  return R1RewardConfig(
    w_accuracy=float(raw.get("w_accuracy", 1.0)),
    w_format=float(raw.get("w_format", 0.2)),
    w_length=float(raw.get("w_length", 0.08)),
    min_thinking_chars=int(raw.get("min_thinking_chars", 80)),
    target_thinking_chars=int(raw.get("target_thinking_chars", 512)),
    partial_format_credit=bool(raw.get("partial_format_credit", False)),
  )
