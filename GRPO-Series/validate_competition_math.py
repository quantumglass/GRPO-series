#!/usr/bin/env python3
"""
competition_math 实验前审计脚本（供人工 / Codex 检查）。

用法:
  python validate_competition_math.py
  python validate_competition_math.py --config configs/config_exgrpo_compmath_3b.yaml
"""

from __future__ import annotations

import argparse
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
from model_registry import resolve_model_config, validate_training_seq_limits
from math_grader import is_math_response_correct, parse_grader_mode
from tokenizer import Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PARQUET = (
    PROJECT_ROOT
    / "data/competition_math/data/train-00000-of-00001-7320a6f3aba8ebd2.parquet"
)


def audit_extraction(df: pd.DataFrame, *, grader_mode: str = "simko") -> dict[str, float]:
    fn = 0
    nested = 0
    total = 0
    mode = parse_grader_mode(grader_mode)
    for _, row in df.iterrows():
        gt = extract_ground_truth_from_solution(str(row["solution"]))
        if gt is None:
            continue
        total += 1
        if "{" in gt or "\\frac" in gt:
            nested += 1
        resp = f"\\boxed{{{gt}}}"
        ok = is_math_response_correct(
            resp, gt, grader_mode=mode
        )
        if not ok:
            fn += 1
    return {
        "samples": float(total),
        "nested_gt": float(nested),
        "false_negative_rate": fn / max(total, 1),
    }


def audit_configs(config_paths: list[Path]) -> None:
    print("\n=== Config audit ===")
    for path in config_paths:
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        model = resolve_model_config(cfg["model"])
        validate_training_seq_limits(
            model,
            max_prompt_len=int(cfg["training"]["max_prompt_len"]),
            max_gen_len=int(cfg["training"]["max_gen_len"]),
        )
        bs = int(cfg["training"]["batch_size"])
        chunk = int(cfg["training"]["rollout_chunk_size"])
        k = bs // int(cfg["training"]["num_questions_per_batch"])
        ok_chunk = chunk < bs
        print(f"  {path.name}: model={model.path.name}, K={k}, "
              f"rollout_chunk={chunk} (<batch {ok_chunk}), "
              f"dataset={cfg['dataset']['name']}, "
              f"split={cfg['dataset'].get('split_mode', 'tail_holdout')}")


def audit_dataset(parquet: Path, tokenizer: Tokenizer) -> None:
    print("\n=== Dataset audit ===")
    raw = pd.read_parquet(parquet)
    print(f"  raw rows: {len(raw)} (expected {OFFICIAL_MATH_RAW_TOTAL_SIZE})")
    train_ds = CompetitionMathDataset(
        tokenizer, str(parquet), split="train", split_mode="official"
    )
    test_ds = CompetitionMathDataset(
        tokenizer, str(parquet), split="test", split_mode="official", max_samples=256
    )
    print(f"  official train usable: {len(train_ds)} (raw segment {OFFICIAL_MATH_TRAIN_SIZE})")
    print(f"  official test eval cap: {len(test_ds)} (raw segment {OFFICIAL_MATH_TEST_SIZE})")
    stats = audit_extraction(raw, grader_mode="simko")
    print(f"  grader_mode: simko")
    print(
        f"  extraction FN rate: {stats['false_negative_rate']:.4%} "
        f"({int(stats['false_negative_rate']*stats['samples'])}/{int(stats['samples'])})"
    )
    print(f"  nested GT count: {int(stats['nested_gt'])}")


def audit_rollout_chunking(config_path: Path) -> None:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    bs = int(cfg["training"]["batch_size"])
    nq = int(cfg["training"]["num_questions_per_batch"])
    chunk = cfg["training"].get("rollout_chunk_size")
    total = bs
    will_chunk = chunk is not None and total > int(chunk)
    print("\n=== Rollout chunking ===")
    print(f"  {config_path.name}: trajectories/step={total}, chunk={chunk}, "
          f"chunking_active={will_chunk}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument(
        "--config",
        action="append",
        type=Path,
        default=[
            PROJECT_ROOT / "configs/config_exgrpo_compmath_3b.yaml",
            PROJECT_ROOT / "configs/config_exgrpo_compmath_7b.yaml",
        ],
    )
    args = parser.parse_args()

    if not args.parquet.is_file():
        raise SystemExit(f"Parquet not found: {args.parquet}")

    tok_path = PROJECT_ROOT / "Qwen2.5-3B-Instruct/tokenizer.json"
    if not tok_path.is_file():
        raise SystemExit(f"Tokenizer not found: {tok_path}")
    tokenizer = Tokenizer(str(tok_path))

    print("competition_math pre-flight audit")
    audit_dataset(args.parquet, tokenizer)
    audit_configs(args.config)
    for cfg_path in args.config:
        audit_rollout_chunking(cfg_path)
    print("\nAudit complete.")


if __name__ == "__main__":
    main()
