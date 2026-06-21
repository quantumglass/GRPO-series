# dapo_math_task.py

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from torch.utils.data import Dataset

from data_types import Episode
from tokenizer import Tokenizer

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_MESSAGE = (
    "You are a helpful assistant. You first think about the reasoning process "
    "in your mind and then provide the user with the answer."
)

RESPONSE_PROMPT = "Let me solve this step by step.\n<think>"


# ---------------------------------------------------------------------------
# MiniBatch（DAPO 专用）
# ---------------------------------------------------------------------------

@dataclass
class DAPOMiniBatch:
    """Batch of data for each DAPO-Math training step."""
    prompt: List[str]
    ground_truth: List[str]
    prefix: List[str]
    prefix_tokens: List[List[str]]
    prefix_token_ids: List[List[int]]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DAPOMathDataset(Dataset):
    """
    PyTorch Dataset for DAPO-Math-17k loaded from a local Parquet file.

    Expected Parquet schema (per row):
        prompt        : list of dict, e.g. [{"content": "<question text>", "role": "user"}]
        reward_model  : dict, e.g. {"ground_truth": "42", "style": "..."}
        data_source   : str  (unused)
        ability       : str  (unused)
        extra_info    : dict (unused)
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        parquet_path: str,
        split: str = "train",
        test_size: int = 200,
    ):
        path = Path(parquet_path)
        if not path.exists():
            raise FileNotFoundError(f"Parquet file not found: {path}")

        df = pd.read_parquet(path)

        # 按照与 CountdownTasksDataset 相同的切分逻辑：末尾 test_size 条为测试集
        if split == "train":
            self.data = df.iloc[:-test_size].reset_index(drop=True)
        else:
            self.data = df.iloc[-test_size:].reset_index(drop=True)

        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.data.iloc[idx]

        # prompt 字段是 list of dict，取第一个元素的 content
        prompt_list = row["prompt"]
        question_text: str = prompt_list[0]["content"]

        # ground_truth 存储在 reward_model 字典中
        reward_model = row["reward_model"]
        ground_truth: str = str(reward_model["ground_truth"])

        encoded = self._encode_prefix(question_text)
        return {
            "prompt": question_text,
            "ground_truth": ground_truth,
            **encoded,
        }

    def _encode_prefix(self, question_text: str) -> Dict[str, Any]:
        """构造模型实际接收的 prefix（system + user + response prompt）。"""
        prefix = self.tokenizer.encode_chat_with_response_prompt(
            [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user",   "content": question_text},
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
    def collate_fn(batch: List[Dict[str, Any]]) -> DAPOMiniBatch:
        return DAPOMiniBatch(
            prompt=[item["prompt"] for item in batch],
            ground_truth=[item["ground_truth"] for item in batch],
            prefix=[item["prefix"] for item in batch],
            prefix_tokens=[item["prefix_tokens"] for item in batch],
            prefix_token_ids=[item["prefix_token_ids"] for item in batch],
        )


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def _extract_answer(response: str) -> Optional[str]:
    """
    从模型输出中提取最终答案。
    优先匹配 <answer>...</answer> 标签（模型按 RESPONSE_PROMPT 引导的输出格式）；
    回退匹配数据集原始要求的 "Answer: $Answer" 末行格式。
    """
    tag_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if tag_match:
        return tag_match.group(1).strip()

    line_match = re.search(r"Answer:\s*\$?\s*(.+)", response)
    if line_match:
        return line_match.group(1).strip()

    return None


def format_reward_function(
    response: str,
    end_token: Optional[str] = None,
) -> float:
    """
    检查输出是否遵循 <think>...</think>\n<answer>...</answer> 格式。
    评分逻辑与原 countdown_task.format_reward_function 完全一致。
    """
    if end_token and response.endswith(end_token):
        response = response[: -len(end_token)]

    think_regex       = r"<think>.*?</think>"
    answer_regex      = r"<answer>.*?</answer>"
    full_format_regex = r"^<think>.*?</think>\n<answer>.*?</answer>$"

    think_match       = re.search(think_regex,      response, re.DOTALL)
    answer_match      = re.search(answer_regex,     response, re.DOTALL)
    full_format_match = re.match(full_format_regex, response, re.DOTALL)

    if full_format_match:
        return 1.0

    reward = 0.0
    if think_match:
        reward += 0.1
    if answer_match:
        reward += 0.5
    return reward


def answer_reward_function(
    response: str,
    ground_truth: str,
) -> float:
    """
    提取模型输出的最终答案与 ground_truth 做匹配。
    先做字符串精确匹配，再做数值容差比较以容忍格式细微差异。
    """
    predicted = _extract_answer(response)
    if predicted is None:
        return 0.0

    if predicted.strip() == ground_truth.strip():
        return 1.0

    # 数值容差比较（处理 "34" vs "34.0" 等边界情况）
    try:
        if abs(float(predicted.strip()) - float(ground_truth.strip())) < 1e-5:
            return 1.0
    except ValueError:
        pass

    return 0.0


def reward_function(
    response: str,
    ground_truth: str = "",
    end_token: Optional[str] = None,
    **kwargs,  # 兼容 grpo.py 中可能透传的 numbers/target 等字段
) -> Dict[str, Any]:
    """
    DAPO-Math 奖励函数，接口与 countdown_task.reward_function 保持一致。
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
