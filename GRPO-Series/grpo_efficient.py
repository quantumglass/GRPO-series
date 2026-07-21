"""
显存友好版 GRPO 工具：分块 rollout，降低 KV cache 峰值占用。
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, List

import torch

from data_types import Episode
from grpo import rollout


def _slice_batch(batch: Any, start: int, end: int) -> Any:
    fields = {}
    for key in dataclasses.fields(batch):
        value = getattr(batch, key.name)
        if isinstance(value, list):
            fields[key.name] = value[start:end]
        else:
            fields[key.name] = value
    return dataclasses.replace(batch, **fields)


def rollout_chunked(
    model,
    batch,
    tokenizer,
    max_gen_len: int,
    num_answer_per_question: int,
    reward_function: Callable,
    device: torch.device,
    dtype: torch.dtype,
    sampling=None,
    rollout_chunk_size: int | None = None,
    end_token: str | None = None,
    end_token_id: int | None = None,
    stop_token_ids: list[int] | None = None,
    treat_max_length_as_finished: bool = False,
) -> List[Episode]:
    """
    将一个大 batch 拆成多个小 batch 依次 rollout，降低 KV cache 峰值。

    rollout_chunk_size：每次并行的 trajectory 数（= 子 batch 题数 × num_answer_per_question）。
    """
    num_questions = len(batch.prefix)
    total_trajectories = num_questions * num_answer_per_question
    if rollout_chunk_size is None or total_trajectories <= rollout_chunk_size:
        return rollout(
            model=model,
            batch=batch,
            tokenizer=tokenizer,
            max_gen_len=max_gen_len,
            num_answer_per_question=num_answer_per_question,
            reward_function=reward_function,
            device=device,
            dtype=dtype,
            sampling=sampling,
            end_token=end_token,
            end_token_id=end_token_id,
            stop_token_ids=stop_token_ids,
            treat_max_length_as_finished=treat_max_length_as_finished,
        )

    questions_per_chunk = max(rollout_chunk_size // num_answer_per_question, 1)
    episodes: List[Episode] = []
    for start in range(0, num_questions, questions_per_chunk):
        end = min(start + questions_per_chunk, num_questions)
        sub_batch = _slice_batch(batch, start, end)
        sub_eps = rollout(
            model=model,
            batch=sub_batch,
            tokenizer=tokenizer,
            max_gen_len=max_gen_len,
            num_answer_per_question=num_answer_per_question,
            reward_function=reward_function,
            device=device,
            dtype=dtype,
            sampling=sampling,
            end_token=end_token,
            end_token_id=end_token_id,
            stop_token_ids=stop_token_ids,
            treat_max_length_as_finished=treat_max_length_as_finished,
        )
        episodes.extend(sub_eps)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    return episodes
