"""SimKO math_equal 判题与 math_grader 分发测试。"""

from __future__ import annotations

import unittest

from exgrpo_reward import ExGRPORewardConfig, exgrpo_reward_config_from_dict
from math_grader import (
    is_math_response_correct,
    legacy_answers_equal,
    math_answers_equal,
    parse_grader_mode,
)


class TestSimkoGrader(unittest.TestCase):
    def test_fraction_numeric_equivalence(self) -> None:
        self.assertTrue(
            math_answers_equal("0.5", r"\frac{1}{2}", grader_mode="simko")
        )

    def test_legacy_distinguishes_fraction_from_float(self) -> None:
        # legacy 字符串匹配通常无法认定 0.5 == \frac{1}{2}
        self.assertFalse(
            legacy_answers_equal("0.5", r"\frac{1}{2}")
        )

    def test_nested_boxed_response(self) -> None:
        resp = r"Steps \boxed{\frac{20}{3}}"
        self.assertTrue(
            is_math_response_correct(
                resp, r"\frac{20}{3}", grader_mode="simko"
            )
        )

    def test_r1_answer_tag_with_simko(self) -> None:
        resp = (
            "<think>work</think>\n"
            r"<answer>\boxed{\frac{1}{2}}</answer>"
        )
        self.assertTrue(
            is_math_response_correct(resp, "0.5", grader_mode="simko")
        )

    def test_legacy_mode_unchanged(self) -> None:
        resp = r"\boxed{4}"
        self.assertTrue(
            is_math_response_correct(resp, "4", grader_mode="legacy")
        )

    def test_parse_grader_mode(self) -> None:
        self.assertEqual(parse_grader_mode("simko"), "simko")
        self.assertEqual(parse_grader_mode(None), "legacy")
        with self.assertRaises(ValueError):
            parse_grader_mode("unknown")

    def test_reward_config_grader_mode(self) -> None:
        cfg = exgrpo_reward_config_from_dict({"grader_mode": "simko"})
        self.assertEqual(cfg.grader_mode, "simko")
        self.assertIsInstance(cfg, ExGRPORewardConfig)


if __name__ == "__main__":
    unittest.main()
