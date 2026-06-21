# gsm8k_task.py

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from torch.utils.data import Dataset
from tokenizer import Tokenizer


SYSTEM_MESSAGE = (
    "You are a helpful assistant. You first think about the reasoning process "
    "in your mind and then provide the user with the answer."
)

RESPONSE_PROMPT = "Let me solve this step by step.\n<think>"


@dataclass
class GSM8KMiniBatch:
    question: List[str]
    ground_truth: List[str]
    prefix: List[str]
    prefix_tokens: List[List[str]]
    prefix_token_ids: List[List[int]]


class GSM8KDataset(Dataset):

    def __init__(
        self,
        tokenizer: Tokenizer,
        parquet_path: str,
    ):
        path = Path(parquet_path)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found")

        self.data = pd.read_parquet(path)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]
        question = row["question"]
        full_answer = row["answer"]

        # 提取 ground truth
        gt_match = re.search(r"####\s*([-]?\d+)", full_answer)
        if gt_match:
            ground_truth = gt_match.group(1)
        else:
            ground_truth = ""

        encoded = self._encode_prefix(question)

        return {
            "question": question,
            "ground_truth": ground_truth,
            **encoded,
        }

    def _encode_prefix(self, question):

        prefix = self.tokenizer.encode_chat_with_response_prompt(
            [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": question},
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
    def collate_fn(batch):

        return GSM8KMiniBatch(
            question=[item["question"] for item in batch],
            ground_truth=[item["ground_truth"] for item in batch],
            prefix=[item["prefix"] for item in batch],
            prefix_tokens=[item["prefix_tokens"] for item in batch],
            prefix_token_ids=[item["prefix_token_ids"] for item in batch],
        )


# ---------------- Reward ---------------- #

# def extract_answer(response: str):

#     tag_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
#     if tag_match:
#         return tag_match.group(1).strip()

#     hash_match = re.search(r"####\s*([-]?\d+)", response)
#     if hash_match:
#         return hash_match.group(1)

#     return None


# import re

def extract_answer(response: str):
    """
    从 <answer>...</answer> 中提取最终数值答案。
    保持格式奖励的严格性，仅在 answer 标签内部做最小解析。
    """

    # 1. 必须存在 <answer> 标签（保持你的格式约束）
    tag_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if not tag_match:
        return None

    answer_text = tag_match.group(1)

    # 2. 从 answer 内容中提取数字（允许自然语言包裹）
    nums = re.findall(r"[-]?\d+(?:,\d{3})*(?:\.\d+)?", answer_text)
    if not nums:
        return None

    # 3. GSM8K 数学答案位于句末，取最后一个数字
    final_num = nums[-1]

    # 4. 标准化
    final_num = final_num.replace(",", "").strip()
    if "." in final_num:
        final_num = final_num.rstrip("0").rstrip(".")

    return final_num


def format_reward_function(response, end_token=None):

    if end_token and response.endswith(end_token):
        response = response[:-len(end_token)]

    think_regex = r"<think>.*?</think>"
    answer_regex = r"<answer>.*?</answer>"

    reward = 0.0

    if re.search(think_regex, response, re.DOTALL):
        reward += 0.1

    if re.search(answer_regex, response, re.DOTALL):
        reward += 0.5

    return reward


def answer_reward_function(response, ground_truth):

    pred = extract_answer(response)

    if pred is None:
        return 0.0

    if pred.strip() == ground_truth.strip():
        return 1.0

    try:
        if abs(float(pred) - float(ground_truth)) < 1e-5:
            return 1.0
    except:
        pass

    return 0.0


def reward_function(response, ground_truth="", end_token=None, **kwargs):

    format_reward = format_reward_function("<think>" + response, end_token)
    answer_reward = answer_reward_function(response, ground_truth)

    return {
        "reward": format_reward * 0.1 + answer_reward,
        "reward_info": {
            "format_reward": format_reward,
            "answer_reward": answer_reward,
        },
    }
