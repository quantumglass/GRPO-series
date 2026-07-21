import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from torch.utils.data import Dataset

from deepscaler_task import _answers_match as math_answers_match
from tokenizer import Tokenizer


def _canonical_benchmark_name(dataset_name: str) -> str:
    return dataset_name.lower().replace("_", "-")


def is_math_benchmark(dataset_name: str) -> bool:
    key = _canonical_benchmark_name(dataset_name).replace("-", "")
    return key in {"math500", "math", "competitionmath"}


def is_aime_benchmark(dataset_name: str) -> bool:
    return "aime" in _canonical_benchmark_name(dataset_name)


def normalize_aime_answer(text: str) -> str:
    """Normalize AIME integer answer (0-999), stripping leading zeros."""
    s = str(text).strip()
    nums = re.findall(r"[-]?\d+", s)
    if not nums:
        return s
    return str(int(nums[-1]))


SYSTEM_MESSAGE = (
    "You are a helpful assistant. You first think about the reasoning process "
    "in your mind and then provide the user with the answer."
)
RESPONSE_PROMPT = "Let me solve this step by step.\n<think>"


@dataclass
class BenchmarkMiniBatch:
    question: List[str]
    ground_truth: List[str]
    prefix: List[str]
    prefix_tokens: List[List[str]]
    prefix_token_ids: List[List[int]]


class GenericMathBenchmarkDataset(Dataset):
    """
    Generic benchmark dataset for GSM8K/Math500-like evaluation.

    Supports JSON, JSONL, and Parquet files.
    """

    def __init__(self, tokenizer: Tokenizer, dataset_name: str, path: str):
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {file_path}")
        self.dataset_name = dataset_name.lower()
        self.rows = self._load_rows(file_path)
        self._validate_rows(file_path)
        self.tokenizer = tokenizer

    def _load_rows(self, path: Path) -> List[Dict[str, Any]]:
        if path.suffix == ".parquet":
            return pd.read_parquet(path).to_dict(orient="records")
        if path.suffix == ".jsonl":
            rows = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            return rows
        if path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, list):
                return payload
            if isinstance(payload, dict):
                if "data" in payload and isinstance(payload["data"], list):
                    return payload["data"]
                raise ValueError("JSON benchmark must be a list or contain a list under 'data'.")
        raise ValueError(f"Unsupported benchmark file type: {path.suffix}")

    def __len__(self) -> int:
        return len(self.rows)

    def _validate_rows(self, file_path: Path) -> None:
        if not self.rows:
            raise ValueError(f"Benchmark file is empty: {file_path}")
        sample = self.rows[0]
        if not isinstance(sample, dict):
            raise ValueError(
                f"Benchmark rows must be JSON-like objects, got: {type(sample).__name__}"
            )

        sample_keys = set(sample.keys())
        if self.dataset_name == "gsm8k":
            required = {"question", "answer"}
            missing = required - sample_keys
            if missing:
                raise KeyError(
                    "GSM8K row is missing required fields "
                    f"{sorted(missing)}. Available keys: {sorted(sample_keys)}"
                )
            return

        if is_math_benchmark(self.dataset_name):
            if "problem" not in sample_keys and "question" not in sample_keys:
                raise KeyError(
                    "MATH500 row requires one of ['problem', 'question']. "
                    f"Available keys: {sorted(sample_keys)}"
                )
            if "answer" not in sample_keys and "solution" not in sample_keys:
                raise KeyError(
                    "MATH500 row requires one of ['answer', 'solution']. "
                    f"Available keys: {sorted(sample_keys)}"
                )
            return

        if is_aime_benchmark(self.dataset_name):
            if "problem" not in sample_keys and "question" not in sample_keys:
                raise KeyError(
                    "AIME row requires one of ['problem', 'question']. "
                    f"Available keys: {sorted(sample_keys)}"
                )
            if "answer" not in sample_keys:
                raise KeyError(
                    "AIME row requires 'answer'. "
                    f"Available keys: {sorted(sample_keys)}"
                )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        question = self._extract_question(row)
        ground_truth = self._extract_ground_truth(row)
        encoded = self._encode_prefix(question)
        return {
            "question": question,
            "ground_truth": ground_truth,
            **encoded,
        }

    def _extract_question(self, row: Dict[str, Any]) -> str:
        candidates = ["question", "problem", "query", "input"]
        for key in candidates:
            if key in row and row[key] is not None:
                return str(row[key])
        raise KeyError("Cannot find question field. Expected one of question/problem/query/input.")

    def _extract_ground_truth(self, row: Dict[str, Any]) -> str:
        if self.dataset_name == "gsm8k":
            ans = str(row.get("answer", ""))
            m = re.search(r"####\s*([-]?\d+(?:,\d{3})*(?:\.\d+)?)", ans)
            return m.group(1).replace(",", "").strip() if m else ans.strip()

        if is_math_benchmark(self.dataset_name):
            if row.get("answer") is not None:
                return normalize_math_answer(str(row["answer"]))
            solution = str(row.get("solution", "")).strip()
            boxed = re.search(r"\\boxed\{([^}]*)\}", solution)
            if boxed:
                return normalize_math_answer(boxed.group(1))
            return solution

        if is_aime_benchmark(self.dataset_name):
            if row.get("answer") is not None:
                return normalize_aime_answer(str(row["answer"]))
            raise KeyError("AIME row is missing required field 'answer'.")

        candidates = ["answer", "final_answer", "ground_truth", "target"]
        for key in candidates:
            if key in row and row[key] is not None:
                value = row[key]
                if isinstance(value, dict):
                    for sub_key in ("answer", "final_answer", "value"):
                        if sub_key in value:
                            return str(value[sub_key]).strip()
                return str(value).strip()

        solution = str(row.get("solution", "")).strip()
        boxed = re.search(r"\\boxed\{([^}]*)\}", solution)
        if boxed:
            return boxed.group(1).strip()
        return solution

    def _encode_prefix(self, question: str) -> Dict[str, Any]:
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
    def collate_fn(batch: List[Dict[str, Any]]) -> BenchmarkMiniBatch:
        return BenchmarkMiniBatch(
            question=[item["question"] for item in batch],
            ground_truth=[item["ground_truth"] for item in batch],
            prefix=[item["prefix"] for item in batch],
            prefix_tokens=[item["prefix_tokens"] for item in batch],
            prefix_token_ids=[item["prefix_token_ids"] for item in batch],
        )


def normalize_math_answer(text: str) -> str:
    """Light normalization for MATH-style LaTeX answers."""
    s = text.strip()
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mbox\{([^}]*)\}", r"\1", s)
    s = s.replace("$", "")
    return s.strip()


def _extract_braced_content(text: str, start: int) -> str | None:
    """Return content inside the first balanced {...} block starting at ``start``."""
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : idx]
    return None


def extract_boxed_answer(text: str) -> str | None:
    """Extract the last \\boxed{...} answer, supporting nested braces."""
    marker = "\\boxed"
    last_match: str | None = None
    search_from = 0
    while True:
        pos = text.find(marker, search_from)
        if pos < 0:
            break
        brace_start = pos + len(marker)
        while brace_start < len(text) and text[brace_start].isspace():
            brace_start += 1
        content = _extract_braced_content(text, brace_start)
        if content is not None:
            last_match = content.strip()
        search_from = pos + len(marker)
    return last_match


def extract_pred_answer(response: str, dataset_name: str = "gsm8k") -> str | None:
    name = _canonical_benchmark_name(dataset_name)
    if is_math_benchmark(name):
        tag_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if tag_match:
            return normalize_math_answer(tag_match.group(1))

        boxed = extract_boxed_answer(response)
        if boxed:
            return normalize_math_answer(boxed)

        # Fallback for plain-text generations without <answer>/\boxed.
        # Keep this conservative and prefer explicit "final answer" cues.
        final_patterns = [
            r"(?:final\s+answer\s*(?:is|:)\s*)(.+)",
            r"(?:answer\s*(?:is|:)\s*)(.+)",
        ]
        for pattern in final_patterns:
            m = re.search(pattern, response, re.IGNORECASE)
            if m:
                cand = m.group(1).strip()
                if cand:
                    return normalize_math_answer(cand.rstrip("。.!? \n\t"))

        lines = [ln.strip() for ln in response.splitlines() if ln.strip()]
        if lines:
            tail = lines[-1].rstrip("。.!? ")
            if tail:
                return normalize_math_answer(tail)
        return None

    tag_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if tag_match:
        answer_text = tag_match.group(1)
    else:
        answer_text = response

    boxed = re.search(r"\\boxed\{([^}]*)\}", answer_text)
    if boxed:
        answer_text = boxed.group(1)

    nums = re.findall(r"[-]?\d+(?:,\d{3})*(?:\.\d+)?", answer_text)
    if nums:
        val = nums[-1].replace(",", "").strip()
        if is_aime_benchmark(name):
            return normalize_aime_answer(val)
        if "." in val:
            val = val.rstrip("0").rstrip(".")
        return val

    cleaned = answer_text.strip()
    if cleaned and is_aime_benchmark(name):
        return normalize_aime_answer(cleaned)
    return cleaned if cleaned else None


def answer_matches(pred: str | None, gt: str, dataset_name: str = "gsm8k") -> bool:
    if pred is None:
        return False

    name = _canonical_benchmark_name(dataset_name)
    if is_math_benchmark(name):
        return math_answers_match(normalize_math_answer(pred), normalize_math_answer(gt))

    if is_aime_benchmark(name):
        try:
            return int(normalize_aime_answer(pred)) == int(normalize_aime_answer(gt))
        except ValueError:
            return False

    p = pred.strip()
    g = str(gt).strip()
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) < 1e-5
    except ValueError:
        pass

    def normalize_tex(s: str) -> str:
        s = re.sub(r"\s+", "", s)
        s = s.replace("\\left", "").replace("\\right", "")
        s = s.replace("{", "").replace("}", "")
        return s.lower()

    return normalize_tex(p) == normalize_tex(g)
