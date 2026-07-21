"""Qwen2.5-Math-7B base 模型的 prompt 与 rollout 停止词。"""

from __future__ import annotations

import dataclasses
from typing import Any

from tokenizer import Tokenizer

# Qwen2.5-Math chat_template 默认 system（与 tokenizer_config 一致）
MATH_SYSTEM_MESSAGE = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

# Qwen2 chat 轮次结束符；base 模型 config.eos 为 <|endoftext|>，但对话应在 im_end 处停止
QWEN_CHAT_IM_END_TOKEN_ID = 151645


def resolve_math_rollout_stop_ids(tokenizer: Tokenizer) -> list[int]:
    """Math base rollout 停止符：native eos + chat im_end（base 更常输出前者）。"""
    ids = [tokenizer.eos_token_id, QWEN_CHAT_IM_END_TOKEN_ID]
    return [tid for tid in ids if tid is not None]


def resolve_math_rollout_stop(tokenizer: Tokenizer) -> tuple[str, int]:
    """返回用于 reward 截断的停止 token 字符串（优先 im_end）。"""
    token_id = QWEN_CHAT_IM_END_TOKEN_ID
    token_str = tokenizer.tokenizer.id_to_token(token_id)
    if token_str is None:
        raise ValueError(
            f"Math base rollout stop token id {token_id} not found in tokenizer"
        )
    return token_str, token_id


def build_math_chat_prefix(tokenizer: Tokenizer, question: str) -> str:
    """构造 Math base 原生 chat prefix（无 R1 thinking 预填）。"""
    return tokenizer.encode_chat(
        [
            {"role": "system", "content": MATH_SYSTEM_MESSAGE},
            {"role": "user", "content": question},
        ]
    )


def _extract_questions_from_batch(batch: Any, dataset_kind: str) -> list[str]:
    """从 batch 提取题目文本（避免依赖 train_exgrpo，防止循环导入）。"""
    if dataset_kind == "countdown":
        return [
            (
                f"Using the numbers {list(numbers)}, create an equation that equals {target}. "
                "You can use basic arithmetic operations (+, -, *, /), and each number can only be used once."
            )
            for numbers, target in zip(batch.numbers, batch.target)
        ]
    if dataset_kind == "math":
        if hasattr(batch, "problem"):
            return [str(x) for x in batch.problem]
        if hasattr(batch, "prompt"):
            return [str(x) for x in batch.prompt]
    raise ValueError(f"Unsupported dataset_kind/batch combination: {dataset_kind}")


def rewrite_batch_with_math_prefix(
    batch: Any,
    tokenizer: Tokenizer,
    dataset_kind: str,
) -> Any:
    """Math base 风格 prefix：原生 system + user，assistant 留空待生成。"""
    questions = _extract_questions_from_batch(batch, dataset_kind)
    prefixes: list[str] = []
    prefix_tokens: list[list[str]] = []
    prefix_token_ids: list[list[int]] = []
    for q in questions:
        prefix = build_math_chat_prefix(tokenizer, q)
        tok = tokenizer.tokenize(prefix)
        prefixes.append(prefix)
        prefix_tokens.append(tok.tokens)
        prefix_token_ids.append(tok.ids)
    return dataclasses.replace(
        batch,
        prefix=prefixes,
        prefix_tokens=prefix_tokens,
        prefix_token_ids=prefix_token_ids,
    )
