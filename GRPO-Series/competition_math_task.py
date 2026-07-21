# competition_math_task.py

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from torch.utils.data import Dataset

from benchmark_task import extract_boxed_answer, normalize_math_answer
from tokenizer import Tokenizer

# ---------------------------------------------------------------------------
# Prompt templates（与其他 math 训练集保持一致）
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
class CompetitionMathMiniBatch:
    """Batch structure for MATH (competition_math) training."""
    problem: List[str]
    ground_truth: List[str]
    prefix: List[str]
    prefix_tokens: List[List[str]]
    prefix_token_ids: List[List[int]]


# ---------------------------------------------------------------------------
# Ground truth extraction
# ---------------------------------------------------------------------------

def extract_ground_truth_from_solution(solution: str) -> Optional[str]:
    """
    从 MATH 数据集的 solution 字段提取 \\boxed{...} 中的最终答案。
    使用 benchmark_task 中带嵌套花括号支持的解析逻辑。
    """
    boxed = extract_boxed_answer(str(solution))
    if boxed is None:
        return None
    normalized = normalize_math_answer(boxed)
    if not normalized:
        return None
    return normalized


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

# Hendrycks MATH 官方划分（HF qwedsacf/competition_math: 7500 train + 5000 test）
OFFICIAL_MATH_TRAIN_SIZE = 7500
OFFICIAL_MATH_TEST_SIZE = 5000
OFFICIAL_MATH_RAW_TOTAL_SIZE = OFFICIAL_MATH_TRAIN_SIZE + OFFICIAL_MATH_TEST_SIZE


class CompetitionMathDataset(Dataset):
    """
    PyTorch Dataset for Hendrycks MATH (HuggingFace: competition_math).

    Parquet schema (per row):
        problem  : str — the competition math problem
        solution : str — step-by-step solution with final answer in \\boxed{}
        level    : str — difficulty level (Level 1–5)
        type     : str — subject (Algebra, Geometry, etc.)

    Args:
        tokenizer    : Tokenizer instance
        parquet_path : path to train parquet file
        split        : "train" or "test"
        test_size    : tail_holdout 模式下测试集样本数
        split_mode   : "official"（7500/5000）或 "tail_holdout"（末尾 test_size 条）
        max_samples  : 可选，限制当前 split 最多使用的样本数（用于训练中快速 eval）
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        parquet_path: str,
        split: str = "train",
        test_size: int = 200,
        split_mode: str = "tail_holdout",
        max_samples: int | None = None,
    ):
        path = Path(parquet_path)
        if not path.exists():
            raise FileNotFoundError(f"Parquet file not found: {path}")

        df_raw = pd.read_parquet(path)
        mode = split_mode.lower().strip()
        if mode == "official":
            if len(df_raw) != OFFICIAL_MATH_RAW_TOTAL_SIZE:
                raise ValueError(
                    "split_mode=official requires the full MATH parquet "
                    f"({OFFICIAL_MATH_RAW_TOTAL_SIZE} rows), got {len(df_raw)}. "
                    "Use split_mode=tail_holdout or provide the complete "
                    "qwedsacf/competition_math export."
                )
            if split == "train":
                df_slice = df_raw.iloc[:OFFICIAL_MATH_TRAIN_SIZE]
            elif split == "test":
                df_slice = df_raw.iloc[OFFICIAL_MATH_TRAIN_SIZE:]
            else:
                raise ValueError(f"split must be 'train' or 'test', got: {split}")
            df = self._filter_valid_rows(df_slice)
            print(
                f"[CompetitionMathDataset] official split {split}: "
                f"{len(df)} usable rows (raw segment size "
                f"{OFFICIAL_MATH_TRAIN_SIZE if split == 'train' else OFFICIAL_MATH_TEST_SIZE})"
            )
        elif mode == "tail_holdout":
            df = self._filter_valid_rows(df_raw)
            if split == "train":
                self.data = df.iloc[:-test_size].reset_index(drop=True)
            elif split == "test":
                self.data = df.iloc[-test_size:].reset_index(drop=True)
            else:
                raise ValueError(f"split must be 'train' or 'test', got: {split}")
        else:
            raise ValueError(
                f"split_mode must be 'official' or 'tail_holdout', got: {split_mode}"
            )

        if mode == "official":
            self.data = df.reset_index(drop=True)
        # tail_holdout assigns self.data above

        if max_samples is not None and max_samples > 0:
            self.data = self.data.iloc[:max_samples].reset_index(drop=True)

        self.tokenizer = tokenizer

    @staticmethod
    def _filter_valid_rows(df: pd.DataFrame) -> pd.DataFrame:
        """丢弃无法从 solution 中解析出 \\boxed 答案的样本。"""
        valid_mask = df["solution"].apply(
            lambda s: extract_ground_truth_from_solution(s) is not None
        )
        dropped = int((~valid_mask).sum())
        if dropped:
            print(
                f"[CompetitionMathDataset] Dropped {dropped} rows "
                "without extractable \\boxed answer."
            )
        return df[valid_mask].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.data.iloc[idx]

        problem: str = str(row["problem"]).strip()
        ground_truth = extract_ground_truth_from_solution(str(row["solution"]))
        assert ground_truth is not None

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
                {"role": "user", "content": problem},
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
    def collate_fn(batch: List[Dict[str, Any]]) -> CompetitionMathMiniBatch:
        return CompetitionMathMiniBatch(
            problem=[item["problem"] for item in batch],
            ground_truth=[item["ground_truth"] for item in batch],
            prefix=[item["prefix"] for item in batch],
            prefix_tokens=[item["prefix_tokens"] for item in batch],
            prefix_token_ids=[item["prefix_token_ids"] for item in batch],
        )


# ---------------------------------------------------------------------------
# Reward functions（复用 DeepScaleR 的竞赛数学答案匹配逻辑）
# ---------------------------------------------------------------------------

from deepscaler_task import (  # noqa: E402
    answer_reward_function,
    extract_answer,
    format_reward_function,
    reward_function,
)

__all__ = [
    "CompetitionMathDataset",
    "CompetitionMathMiniBatch",
    "extract_ground_truth_from_solution",
    "extract_answer",
    "format_reward_function",
    "answer_reward_function",
    "reward_function",
]
