"""ExGRPO 训练管线可插拔 hooks（3B R1 与 Math base 共用 train_exgrpo.main）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from exgrpo_reward import ExGRPORewardConfig, build_exgrpo_reward_function
from tokenizer import Tokenizer


@dataclass(frozen=True)
class ExGRPOTrainingHooks:
    rewrite_batch_prefix: Callable[[Any, Tokenizer, str], Any]
    build_reward_function: Callable[[str, ExGRPORewardConfig], Callable[..., dict[str, Any]]]
    resolve_rollout_stop: Callable[[Tokenizer], tuple[str | None, int | None]]
    # base 模型常不输出 chat 停止符；打满 max_gen_len 时仍视为 finished
    treat_max_length_as_finished: bool = False
    # 额外停止 token id（如 base 用 <|endoftext|> + im_end）
    resolve_rollout_stop_ids: Callable[[Tokenizer], list[int]] | None = None
