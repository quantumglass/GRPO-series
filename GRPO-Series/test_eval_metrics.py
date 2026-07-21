"""Unit tests for eval pass@k metrics."""

from __future__ import annotations

import unittest

from eval_metrics import (
    build_benchmark_result,
    compute_pass_at_k_from_samples,
    format_pass_at_k_summary,
    is_pass_at_k_solved,
    pass_at_k_simple,
    resolve_accuracy_fields,
)


class TestEvalMetrics(unittest.TestCase):
    def test_pass_at_k_simple_aime_like(self) -> None:
        samples = [[True] + [False] * 15] * 7 + [[False] * 16] * 23
        n = 16
        metrics = compute_pass_at_k_from_samples(samples, [1, 4, 8, 16], n)
        self.assertEqual(metrics["16"]["rate"], 7 / 30)
        self.assertEqual(metrics["16"]["passed"], 7)
        self.assertEqual(metrics["1"]["rate"], 7 / 30)
        self.assertEqual(metrics["1"]["passed"], 7)
        self.assertEqual(metrics["_meta"]["any_correct"], 7)

    def test_pass_at_k_simple_late_correct(self) -> None:
        samples = [[False] * 5 + [True] + [False] * 10] + [[False] * 16] * 29
        n = 16
        metrics = compute_pass_at_k_from_samples(samples, [1, 4, 8, 16], n)
        self.assertEqual(metrics["1"]["rate"], 0.0)
        self.assertEqual(metrics["4"]["rate"], 0.0)
        self.assertEqual(metrics["8"]["rate"], 1 / 30)
        self.assertEqual(metrics["16"]["rate"], 1 / 30)

    def test_is_pass_at_k_solved(self) -> None:
        sample = [False, True] + [False] * 14
        self.assertFalse(is_pass_at_k_solved(sample, 1))
        self.assertTrue(is_pass_at_k_solved(sample, 2))
        self.assertTrue(is_pass_at_k_solved(sample, 16))

    def test_pass_at_k_simple_helper(self) -> None:
        self.assertEqual(pass_at_k_simple([True, False], 1), 1.0)
        self.assertEqual(pass_at_k_simple([False, True], 1), 0.0)
        self.assertEqual(pass_at_k_simple([False, True], 2), 1.0)

    def test_accuracy_uses_first_sample_when_multi_sample(self) -> None:
        samples = [[True, True], [False, False], [False, True]]
        first = [True, False, False]
        accuracy, correct = resolve_accuracy_fields(
            sample_results=samples,
            num_samples=16,
            first_sample_correct=first,
        )
        self.assertEqual(accuracy, 1 / 3)
        self.assertEqual(correct, 1)

    def test_build_benchmark_result_single_sample(self) -> None:
        result = build_benchmark_result(
            sample_results=[[True], [False], [True]],
            num_samples=1,
            pass_at_k=[1],
        )
        self.assertEqual(result["accuracy"], 2 / 3)
        self.assertEqual(result["correct"], 2)
        self.assertEqual(result["pass_at_k"]["1"]["passed"], 2)
        self.assertNotIn("_meta", result["pass_at_k"])

    def test_format_pass_at_k_summary_multi_sample(self) -> None:
        samples = [[True] + [False] * 15] + [[False] * 16] * 29
        metrics = compute_pass_at_k_from_samples(samples, [1, 4, 16], 16)
        text = format_pass_at_k_summary(metrics, num_samples=16)
        self.assertIn("pass@1=0.0333 (1/30)", text)
        self.assertIn("pass@16=0.0333 (1/30)", text)
        self.assertIn("[n=16 samples/q]", text)


if __name__ == "__main__":
    unittest.main()
