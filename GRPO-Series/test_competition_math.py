"""competition_math 数据集与奖励判题单元测试。"""

from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd
import yaml

from competition_math_task import (
    OFFICIAL_MATH_RAW_TOTAL_SIZE,
    OFFICIAL_MATH_TEST_SIZE,
    OFFICIAL_MATH_TRAIN_SIZE,
    CompetitionMathDataset,
    extract_ground_truth_from_solution,
)
from deepscaler_task import extract_answer, _answers_match
from exgrpo_reward import ExGRPORewardConfig, compute_exgrpo_reward
from math_base_reward import compute_math_base_reward
from r1_thinking_reward import compute_accuracy_reward, is_r1_benchmark_answer_correct

PROJECT_ROOT = Path(__file__).resolve().parent
PARQUET = (
    PROJECT_ROOT
    / "data/competition_math/data/train-00000-of-00001-7320a6f3aba8ebd2.parquet"
)


class TestBoxedExtraction(unittest.TestCase):
    def test_nested_fraction_extract_answer(self) -> None:
        resp = r"Steps... \boxed{\frac{1}{2}}"
        self.assertEqual(extract_answer(resp), r"\frac{1}{2}")

    def test_nested_sqrt_extract_answer(self) -> None:
        resp = r"Thus \boxed{\sqrt{2}+1}"
        self.assertEqual(extract_answer(resp), r"\sqrt{2}+1")

    def test_answer_tag_priority(self) -> None:
        resp = "<answer>42</answer> with \\boxed{99}"
        self.assertEqual(extract_answer(resp), "42")


class TestOfficialSplit(unittest.TestCase):
    @unittest.skipUnless(PARQUET.is_file(), "competition_math parquet missing")
    def test_official_split_sizes(self) -> None:
        from tokenizer import Tokenizer

        tok_path = PROJECT_ROOT / "Qwen2.5-3B-Instruct/tokenizer.json"
        if not tok_path.is_file():
            self.skipTest("3B tokenizer missing")
        tok = Tokenizer(str(tok_path))

        train_ds = CompetitionMathDataset(
            tok, str(PARQUET), split="train", split_mode="official"
        )
        test_ds = CompetitionMathDataset(
            tok,
            str(PARQUET),
            split="test",
            split_mode="official",
            max_samples=256,
        )
        # 过滤后略少于官方计数，但应接近
        self.assertGreater(len(train_ds), OFFICIAL_MATH_TRAIN_SIZE - 10)
        self.assertLessEqual(len(train_ds), OFFICIAL_MATH_TRAIN_SIZE)
        self.assertEqual(len(test_ds), 256)

    @unittest.skipUnless(PARQUET.is_file(), "competition_math parquet missing")
    def test_raw_parquet_row_count(self) -> None:
        df = pd.read_parquet(PARQUET)
        self.assertEqual(len(df), OFFICIAL_MATH_RAW_TOTAL_SIZE)


class TestRewardNoise(unittest.TestCase):
    @unittest.skipUnless(PARQUET.is_file(), "competition_math parquet missing")
    def test_self_consistency_on_ground_truth_boxed(self) -> None:
        """用 GT 构造 \\boxed{gt} 响应，判题应与 GT 一致（噪声上界审计）。"""
        df = pd.read_parquet(PARQUET)
        cfg = ExGRPORewardConfig(w_accuracy=1.0, w_format=0.2, accuracy_mode="signed")
        false_neg = 0
        nested = 0
        checked = 0
        for _, row in df.iterrows():
            gt = extract_ground_truth_from_solution(str(row["solution"]))
            if gt is None:
                continue
            if "{" in gt or "\\frac" in gt or "\\sqrt" in gt:
                nested += 1
            resp = f"Reasoning.\n\\boxed{{{gt}}}"
            acc = compute_accuracy_reward(
                resp, dataset_kind="math", ground_truth=gt, accuracy_mode="signed"
            )
            if acc <= 0:
                false_neg += 1
            checked += 1
        fn_rate = false_neg / max(checked, 1)
        self.assertLess(
            fn_rate,
            0.01,
            f"false-negative rate {fn_rate:.4f} too high ({false_neg}/{checked})",
        )
        self.assertGreater(nested, 1000, "sanity: dataset should have many nested answers")

    def test_r1_candidate_matches_nested_gt(self) -> None:
        gt = r"\frac{20}{3}"
        resp = (
            "<think>work</think>\n"
            f"<answer>\\boxed{{{gt}}}</answer>"
        )
        self.assertTrue(
            is_r1_benchmark_answer_correct(resp, gt, dataset_name="math500")
        )

    def test_math_base_format_reward(self) -> None:
        cfg = ExGRPORewardConfig(w_accuracy=1.0, w_format=0.2)
        out = compute_math_base_reward(
            r"Step \boxed{4}",
            dataset_kind="math",
            cfg=cfg,
            ground_truth="4",
        )
        self.assertGreater(out["reward_info"]["accuracy_reward"], 0.0)
        self.assertGreater(out["reward_info"]["format_reward"], 0.9)

    def test_exgrpo_format_for_r1_response(self) -> None:
        cfg = ExGRPORewardConfig(w_accuracy=1.0, w_format=0.2)
        resp = (
            "<think>reason</think>\n"
            "<answer>\\boxed{7}</answer>"
        )
        out = compute_exgrpo_reward(
            resp, dataset_kind="math", cfg=cfg, ground_truth="7"
        )
        self.assertGreater(out["reward"], 0.5)


class TestExperimentConfigs(unittest.TestCase):
    def test_main_exgrpo_config_supports_competition_math(self) -> None:
        path = PROJECT_ROOT / "configs/train_exgrpo.yaml"
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.assertIn(
            "competition_math_parquet_path",
            cfg["dataset"],
        )
        bs = cfg["training"]["batch_size"]
        nq = cfg["training"]["num_questions_per_batch"]
        self.assertEqual(bs % nq, 0)
        self.assertEqual(bs // nq, cfg["training"]["exgrpo"]["K"])
        self.assertIsNone(cfg["training"]["resume_lora_ckpt"])

    def test_exgrpo_7b_config_batch_consistency(self) -> None:
        path = PROJECT_ROOT / "configs/train_exgrpo_7b.yaml"
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.assertEqual(cfg["model"]["preset"], "qwen2.5-math-7b")
        bs = cfg["training"]["batch_size"]
        nq = cfg["training"]["num_questions_per_batch"]
        self.assertEqual(bs // nq, cfg["training"]["exgrpo"]["K"])
        self.assertLessEqual(
            cfg["training"]["rollout_chunk_size"],
            bs,
        )
        self.assertIsNone(cfg["training"]["resume_lora_ckpt"])
        self.assertTrue(cfg["training"]["memory_efficient_adamw"])
        self.assertEqual(cfg["training"]["exgrpo"]["K"], 8)


class TestLegacyMatcher(unittest.TestCase):
    def test_answers_match_fraction(self) -> None:
        self.assertTrue(_answers_match(r"\frac{1}{2}", r"\frac{1}{2}"))


if __name__ == "__main__":
    unittest.main()
