"""
DeepSeek-R1 风格的 thinking 奖励函数。

三类奖励（权重均可设为 0 以关闭）：
  - thinking：thinking 段长度温和线性鼓励（非 R1 原文项）
  - acc：答案正确性，计分模式 binary(1/0) 或 signed(1/-1)
  - format：<think> + <answer> 结构遵循

总奖励 = w_thinking * thinking + w_accuracy * acc + w_format * format
评测 success_rate 始终按 acc 是否正确（1/0）统计，与 mean_reward 解耦。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Literal

AccuracyMode = Literal["binary", "signed"]
ACCURACY_MODES: tuple[AccuracyMode, ...] = ("binary", "signed")

from simko_grader import GraderMode, parse_grader_mode as _parse_grader_mode

from benchmark_task import answer_matches, extract_boxed_answer, extract_pred_answer
from deepscaler_task import (
    _answers_match as deepscaler_answers_match,
    extract_answer as deepscaler_extract_answer,
)


@dataclass(frozen=True)
class R1RewardConfig:
  """奖励权重与 thinking 长度鼓励参数。"""

  w_thinking: float = 0.08
  w_accuracy: float = 1.0
  w_format: float = 0.2
  # binary: 正确 1 / 错误或未解析 0；signed: 正确 1 / 错误或未解析 -1
  accuracy_mode: AccuracyMode = "binary"
  # thinking 字符数低于 min_chars 不给分；到 target_chars 给满 1.0
  min_thinking_chars: int = 80
  target_thinking_chars: int = 512
  # 格式分：R1 为二元；设为 True 时按 think/answer 标签给部分分
  partial_format_credit: bool = False

  @property
  def w_length(self) -> float:
    """兼容旧配置键 w_length。"""
    return self.w_thinking


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
  if THINK_OPEN not in text:
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


def score_accuracy(correct: bool, mode: AccuracyMode = "binary") -> float:
  if correct:
    return 1.0
  return 0.0 if mode == "binary" else -1.0


def _unwrap_latex_wrappers(text: str) -> str:
  text = text.strip()
  m = re.match(r"^\\\((.+?)\\\)$", text, re.DOTALL)
  if m:
    return m.group(1).strip()
  m = re.match(r"^\$\$(.+?)\$\$$", text, re.DOTALL)
  if m:
    return m.group(1).strip()
  m = re.match(r"^\$(.+?)\$$", text, re.DOTALL)
  if m:
    return m.group(1).strip()
  return text


def _iter_display_math_blocks(text: str):
  """提取 \\[ ... \\] 块（非贪婪 .+? 无法处理 \\frac{}{} 嵌套）。"""
  pos = 0
  open_marker = "\\["
  close_marker = "\\]"
  while pos < len(text):
    start = text.find(open_marker, pos)
    if start < 0:
      break
    next_open = text.find(open_marker, start + len(open_marker))
    end = text.find(close_marker, start + len(open_marker))
    if end < 0:
      break
    # 若块内还有 \\[，延伸到本段最后一个 \\]
    if next_open >= 0 and next_open < end:
      end = text.rfind(close_marker, start, next_open)
    else:
      end = text.rfind(close_marker, start)
    if end <= start:
      break
    yield text[start + len(open_marker) : end].strip()
    pos = end + len(close_marker)


def _trailing_equals_rhs(text: str) -> str | None:
  text = text.strip()
  pos = text.rfind("=")
  if pos < 0:
    return None
  rhs = text[pos + 1 :].strip().rstrip(".")
  return rhs or None


def _push_candidate(candidates: list[str], seen: set[str], raw: str) -> None:
  for item in (raw, _unwrap_latex_wrappers(raw)):
    s = item.strip()
    if not s or s in seen:
      continue
    seen.add(s)
    candidates.append(s)


def _candidates_from_answer_block(block: str) -> list[str]:
  """从 <answer> 正文中挖掘多个可比对候选（兼容模型输出自然语言包裹的答案）。"""
  out: list[str] = []
  seen: set[str] = set()
  block = block.strip()
  if not block:
    return out

  boxed = extract_boxed_answer(block)
  if boxed:
    _push_candidate(out, seen, boxed)

  for inner in _iter_display_math_blocks(block):
    _push_candidate(out, seen, inner)
    rhs = _trailing_equals_rhs(inner)
    if rhs:
      _push_candidate(out, seen, rhs)

  # 单行 \( ... \) 包裹
  for m in re.finditer(r"\\\((.+?)\\\)", block, re.DOTALL):
    _push_candidate(out, seen, m.group(1).strip())

  _push_candidate(out, seen, block)

  lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
  if lines:
    last = lines[-1].rstrip(".")
    rhs = _trailing_equals_rhs(last)
    if rhs:
      _push_candidate(out, seen, rhs)
    _push_candidate(out, seen, last)

  return out


def _candidates_from_thinking(think: str) -> list[str]:
  """thinking 段回退：\\boxed{}、display math 右侧、行末数值等式。"""
  out: list[str] = []
  seen: set[str] = set()
  boxed = extract_boxed_answer(think)
  if boxed:
    _push_candidate(out, seen, boxed)
  for inner in _iter_display_math_blocks(think):
    rhs = _trailing_equals_rhs(inner)
    if rhs:
      _push_candidate(out, seen, rhs)
  for line in think.splitlines():
    line = line.strip()
    if not line:
      continue
    rhs = _trailing_equals_rhs(line)
    if rhs:
      _push_candidate(out, seen, rhs)
    # 行末纯数值等式，如 "3 \\times 3 = 9"
    num_eq = re.search(r"=\s*([-+]?\d+(?:\.\d+)?)\s*$", line)
    if num_eq:
      _push_candidate(out, seen, num_eq.group(1))
  return out


def iter_r1_answer_candidates(
  response: str,
  *,
  dataset_name: str = "math500",
  end_token: str | None = None,
) -> list[str]:
  """
  R1 评测答案候选列表（按优先级）。
  模型常在 <answer> 中输出冗长自然语言；此处从 answer / thinking / \\boxed 多路径提取。
  """
  text = _normalize_for_format_check(response, end_token)
  candidates: list[str] = []
  seen: set[str] = set()

  ans_m = ANSWER_BLOCK_RE.search(text)
  if ans_m:
    for cand in _candidates_from_answer_block(ans_m.group(1)):
      _push_candidate(candidates, seen, cand)

  think = extract_thinking_text(text, end_token=None)
  for cand in _candidates_from_thinking(think):
    _push_candidate(candidates, seen, cand)

  boxed_all = extract_boxed_answer(text)
  if boxed_all:
    _push_candidate(candidates, seen, boxed_all)

  fallback = extract_pred_answer(text, dataset_name=dataset_name)
  if fallback:
    _push_candidate(candidates, seen, fallback)
  return candidates


def extract_r1_pred_answer(
  response: str,
  *,
  dataset_name: str = "math500",
  end_token: str | None = None,
) -> str | None:
  """返回首个候选答案（用于日志）；正确性判断请用 is_r1_benchmark_answer_correct。"""
  cands = iter_r1_answer_candidates(
    response, dataset_name=dataset_name, end_token=end_token
  )
  return cands[0] if cands else None


def is_r1_benchmark_answer_correct(
  response: str,
  ground_truth: str,
  *,
  dataset_name: str = "math500",
  end_token: str | None = None,
) -> bool:
  """R1 评测：遍历候选答案，任一与 GT 匹配即判对（与训练奖励同一套 matcher）。"""
  gt = str(ground_truth)
  for pred in iter_r1_answer_candidates(
    response, dataset_name=dataset_name, end_token=end_token
  ):
    if deepscaler_answers_match(pred, gt):
      return True
    if answer_matches(pred, gt, dataset_name=dataset_name):
      return True
  return False


def compute_thinking_reward(
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


def compute_thinking_length_bonus(
  response: str,
  end_token: str | None,
  cfg: R1RewardConfig,
) -> float:
  """兼容旧函数名。"""
  return compute_thinking_reward(response, end_token, cfg)


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
  accuracy_mode: AccuracyMode = "binary",
  grader_mode: GraderMode = "legacy",
) -> float:
  text = _strip_end_token(response, end_token)
  if dataset_kind == "countdown":
    ok = bool(
      numbers is not None
      and target is not None
      and _countdown_answer_correct(text, numbers, target)
    )
    return score_accuracy(ok, accuracy_mode)

  if dataset_kind == "math":
    from math_grader import is_math_response_correct

    ok = is_math_response_correct(
      text,
      str(ground_truth),
      grader_mode=grader_mode,
      end_token=end_token,
      dataset_name="math500",
    )
    return score_accuracy(ok, accuracy_mode)

  pred = deepscaler_extract_answer(text)
  if pred is None:
    pred = extract_pred_answer(text, dataset_name="math500")
  if pred is None:
    return score_accuracy(False, accuracy_mode)
  if deepscaler_answers_match(pred, str(ground_truth)):
    return score_accuracy(True, accuracy_mode)
  if answer_matches(pred, str(ground_truth), dataset_name="math500"):
    return score_accuracy(True, accuracy_mode)
  return score_accuracy(False, accuracy_mode)


def is_accuracy_correct(accuracy_reward: float) -> bool:
  """从 acc 分项判断答案是否正确（兼容 binary / signed）。"""
  return accuracy_reward > 0.0


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
    accuracy_mode=cfg.accuracy_mode,
  )
  format_score = compute_format_reward(response, end_token, cfg)
  thinking_score = compute_thinking_reward(response, end_token, cfg)
  # 无格式时不给 thinking 分，避免模型刷无结构长文本
  if format_score <= 0.0 and not cfg.partial_format_credit:
    thinking_score = 0.0
  elif format_score < 0.2 and cfg.partial_format_credit:
    thinking_score = 0.0

  total = (
    cfg.w_thinking * thinking_score
    + cfg.w_accuracy * accuracy
    + cfg.w_format * format_score
  )
  thinking_chars = len(extract_thinking_text(response, end_token))
  return {
    "reward": total,
    "reward_info": {
      "thinking_reward": thinking_score,
      "length_bonus": thinking_score,
      "accuracy_reward": accuracy,
      "format_reward": format_score,
      "answer_reward": accuracy,
      "accuracy_correct": float(is_accuracy_correct(accuracy)),
      "thinking_chars": float(thinking_chars),
      "accuracy_mode": cfg.accuracy_mode,
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


def _parse_accuracy_mode(raw: str) -> AccuracyMode:
  mode = str(raw).lower().strip()
  if mode in ("binary", "1/0", "one_zero"):
    return "binary"
  if mode in ("signed", "1/-1", "one_neg_one", "plus_minus"):
    return "signed"
  raise ValueError(
    f"accuracy_mode must be one of {ACCURACY_MODES} "
    f"(aliases: 1/0, 1/-1), got: {raw!r}"
  )


def r1_reward_config_from_dict(raw: dict[str, Any] | None) -> R1RewardConfig:
  raw = raw or {}
  w_thinking = raw.get("w_thinking", raw.get("w_length", 0.08))
  return R1RewardConfig(
    w_thinking=float(w_thinking),
    w_accuracy=float(raw.get("w_accuracy", 1.0)),
    w_format=float(raw.get("w_format", 0.2)),
    accuracy_mode=_parse_accuracy_mode(raw.get("accuracy_mode", "binary")),
    min_thinking_chars=int(raw.get("min_thinking_chars", 80)),
    target_thinking_chars=int(raw.get("target_thinking_chars", 512)),
    partial_format_credit=bool(raw.get("partial_format_credit", False)),
  )
