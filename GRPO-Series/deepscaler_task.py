# deepscaler_task.py

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from torch.utils.data import Dataset

from tokenizer import Tokenizer

# ---------------------------------------------------------------------------
# Prompt templates（与 GSM8K 保持一致）
# ---------------------------------------------------------------------------

SYSTEM_MESSAGE = (
    "You are a helpful assistant. You first think about the reasoning process "
    "in your mind and then provide the user with the answer."
)

RESPONSE_PROMPT = "Let me solve this step by step.\n<think>"


# ---------------------------------------------------------------------------
# MiniBatch
# ---------------------------------------------------------------------------

@dataclass
class DeepScalerMiniBatch:
    """Batch structure for DeepScaleR training."""
    problem: List[str]
    ground_truth: List[str]
    prefix: List[str]
    prefix_tokens: List[List[str]]
    prefix_token_ids: List[List[int]]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DeepScalerDataset(Dataset):
    """
    PyTorch Dataset for agentica-org/DeepScaleR-Preview-Dataset.

    JSON schema (per record):
        problem  : str  — the math problem text
        answer   : str  — the ground truth answer (may contain LaTeX)
        solution : str  — full solution (unused during RL training)

    Args:
        tokenizer    : Tokenizer instance
        json_path    : path to deepscaler.json
        split        : "train" or "test"
        test_size    : number of samples reserved for test split
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        json_path: str,
        split: str = "train",
        test_size: int = 200,
    ):
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # 与 GSM8K 保持一致：末尾 test_size 条作为测试集
        total = len(raw)
        if split == "train":
            self.data = raw[: total - test_size]
        else:
            self.data = raw[total - test_size :]

        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.data[idx]

        problem: str = record["problem"]
        # answer 字段已经是纯净答案字符串，无需额外提取
        ground_truth: str = str(record["answer"]).strip()

        encoded = self._encode_prefix(problem)

        return {
            "problem": problem,
            "ground_truth": ground_truth,
            **encoded,
        }

    def _encode_prefix(self, problem: str) -> Dict[str, Any]:
        """构造模型实际接收的 prefix（system + user + response prompt）。"""
        prefix = self.tokenizer.encode_chat_with_response_prompt(
            [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user",   "content": problem},
            ],
            RESPONSE_PROMPT,
        )
        tokens = self.tokenizer.tokenize(prefix)
        return {
            "prefix": prefix,
            "prefix_tokens": tokens.tokens,
            "prefix_token_ids": tokens.ids,
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> DeepScalerMiniBatch:
        return DeepScalerMiniBatch(
            problem=[item["problem"] for item in batch],
            ground_truth=[item["ground_truth"] for item in batch],
            prefix=[item["prefix"] for item in batch],
            prefix_tokens=[item["prefix_tokens"] for item in batch],
            prefix_token_ids=[item["prefix_token_ids"] for item in batch],
        )


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def _normalize_number(text: str) -> Optional[str]:
    """
    将字符串标准化为可比较的数值字符串。
    处理：千位逗号、货币符号、小数点尾零、百分号。
    """
    text = text.strip()
    text = text.replace(",", "").replace("$", "").replace("%", "")
    text = text.rstrip(".")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text if text else None


def extract_answer(response: str) -> Optional[str]:
    """
    从模型输出中提取最终答案。

    DeepScaleR 的答案可能包含分数、根号、LaTeX 等非整数形式。
    提取策略：
      1. 优先匹配 <answer>...</answer> 标签
      2. 回退匹配 \\boxed{...}（竞赛数学常见格式，支持嵌套花括号）
      3. 不做过度宽松的兜底，避免引入噪声
    """
    # 第 1 层：<answer>...</answer> 标签
    tag_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if tag_match:
        return tag_match.group(1).strip()

    # 第 2 层：\boxed{...}（延迟导入避免与 benchmark_task 循环依赖）
    from benchmark_task import extract_boxed_answer

    boxed = extract_boxed_answer(response)
    if boxed is not None:
        return boxed.strip()

    return None


def _answers_match(pred: str, gt: str) -> bool:
    """
    比较预测答案与 ground truth。

    DeepScaleR 的答案多样性远高于 GSM8K，包括：
      - 整数：35、-7
      - 分数：\\frac{2}{3}、9/10
      - 根号：\\sqrt{3}、2\\sqrt{2}
      - 百分比：80%
      - 混合表达式：3+2\\sqrt{3}

    匹配策略（按优先级）：
      1. 字符串精确匹配（去除空白后）
      2. 数值近似匹配（对可转换为 float 的答案）
      3. LaTeX 规范化后精确匹配
    """
    pred = pred.strip()
    gt   = gt.strip()

    # 策略 1：字符串精确匹配
    if pred == gt:
        return True

    # 策略 2：数值近似匹配
    pred_norm = _normalize_number(pred)
    gt_norm   = _normalize_number(gt)
    if pred_norm and gt_norm:
        try:
            if abs(float(pred_norm) - float(gt_norm)) < 1e-5:
                return True
        except ValueError:
            pass

    # 策略 3：LaTeX 规范化（去除空格、统一斜杠）
    def _latex_normalize(s: str) -> str:
        s = re.sub(r"\s+", "", s)
        s = s.replace("\\left", "").replace("\\right", "")
        s = s.replace("{", "").replace("}", "")
        return s.lower()

    if _latex_normalize(pred) == _latex_normalize(gt):
        return True

    return False


def format_reward_function(
    response: str,
    end_token: Optional[str] = None,
) -> float:
    """
    检查输出是否遵循 <think>...</think>\n<answer>...</answer> 格式。
    与 GSM8K 版本保持完全一致的评分逻辑。
    """
    if end_token and response.endswith(end_token):
        response = response[: -len(end_token)]

    reward = 0.0
    if re.search(r"<think>.*?</think>", response, re.DOTALL):
        reward += 0.1
    if re.search(r"<answer>.*?</answer>", response, re.DOTALL):
        reward += 0.5
    return reward


def answer_reward_function(response: str, ground_truth: str) -> float:
    """
    提取模型输出的最终答案并与 ground_truth 比较。
    返回 1.0（正确）或 0.0（错误）。
    """
    pred = extract_answer(response)
    if pred is None:
        return 0.0
    return 1.0 if _answers_match(pred, ground_truth) else 0.0


def reward_function(
    response: str,
    ground_truth: str = "",
    end_token: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    DeepScaleR 奖励函数。接口与 gsm8k_task.reward_function 完全一致。
    Total reward = format_reward * 0.1 + answer_reward
    """
    format_reward = format_reward_function("<think>" + response, end_token)
    answer_reward = answer_reward_function(response, ground_truth)
    return {
        "reward": format_reward * 0.1 + answer_reward,
        "reward_info": {
            "format_reward": format_reward,
            "answer_reward": answer_reward,
        },
    }
