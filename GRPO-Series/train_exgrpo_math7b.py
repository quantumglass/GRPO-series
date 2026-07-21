"""
ExGRPO + GSPO 训练入口 — Qwen2.5-Math-7B base。

与 train_exgrpo.py（3B-Instruct / R1 格式）独立：
  - Math 原生 system prompt + \\boxed{} 输出格式
  - rollout 在  处停止（非 <|endoftext|>）
  - format 奖励检查 \\boxed{} 而非 <think>/<answer>

用法:
  uv run python train_exgrpo_math7b.py --config configs/config_exgrpo_7b.yaml
"""

from argparse import ArgumentParser

from checkpoint import add_resume_arguments
from exgrpo_training_hooks import ExGRPOTrainingHooks
from math_base_prompt import (
    resolve_math_rollout_stop,
    resolve_math_rollout_stop_ids,
    rewrite_batch_with_math_prefix,
)
from math_base_reward import build_math_base_reward_function
from model_registry import MATH_BASE_MODEL_PRESETS
from train_exgrpo import main

MATH_BASE_HOOKS = ExGRPOTrainingHooks(
    rewrite_batch_prefix=rewrite_batch_with_math_prefix,
    build_reward_function=build_math_base_reward_function,
    resolve_rollout_stop=resolve_math_rollout_stop,
    treat_max_length_as_finished=True,
    resolve_rollout_stop_ids=resolve_math_rollout_stop_ids,
)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_exgrpo_7b.yaml",
        help="Math-7B base 训练配置",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=sorted(MATH_BASE_MODEL_PRESETS),
        help="Override model.preset (default: qwen2.5-math-7b)",
    )
    add_resume_arguments(parser)
    args = parser.parse_args()
    main(
        args.config,
        args.resume_lora_ckpt,
        args.resume_log_dir,
        args.model,
        training_hooks=MATH_BASE_HOOKS,
    )
