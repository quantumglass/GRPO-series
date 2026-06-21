"""
GRPO / GSPO / PPO 核心算法：rollout 采样与策略更新。

- rollout: 逐 token 自回归生成，记录 log_prob
- update_policy: 组内优势归一化 + PPO clip + 可选 KL 惩罚
"""

import dataclasses
import gc
import math
from collections import defaultdict
from typing import Callable, List

import numpy as np
import torch

from data_types import Episode, MiniBatch
from qwen2_model import Transformer
from sampling import SamplingConfig, sample_next_token
from tokenizer import Tokenizer


def _flush_cuda_memory(device: torch.device) -> None:
    """Wait for async GPU ops to finish and return cached blocks to the allocator."""
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


@torch.no_grad()
def rollout(
    model: Transformer,
    batch: MiniBatch,
    tokenizer: Tokenizer,
    max_gen_len: int,
    num_answer_per_question: int,
    reward_function: Callable,
    device: torch.device,
    dtype: torch.dtype,
    sampling: SamplingConfig | None = None,
) -> List[Episode]:
    sampling = sampling or SamplingConfig()
    end_token = tokenizer.eos_token
    end_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    prefix_token_ids = batch.prefix_token_ids
    bsz = len(batch.prefix) * num_answer_per_question
    min_prompt_len = min(len(t) for t in prefix_token_ids)
    max_prompt_len = max(len(t) for t in prefix_token_ids)
    total_len = max_gen_len + max_prompt_len
    model.init_kv_cache(
        max_batch_size=bsz,
        max_seq_len=total_len,
        device=device,
        dtype=dtype,
    )
    tokens = torch.full((bsz, total_len), pad_token_id, dtype=torch.long, device=device)
    for k, t in enumerate(prefix_token_ids):
        offset = k * num_answer_per_question
        for i in range(num_answer_per_question):
            tokens[offset + i, : len(t)] = torch.tensor(
                t, dtype=torch.long, device=device
            )

    prev_pos = 0
    input_text_mask = tokens != pad_token_id
    assert min_prompt_len < total_len
    is_finished = torch.zeros((bsz,), dtype=torch.bool, device=device)
    generated_log_probs: list[list[float]] = [[] for _ in range(bsz)]

    for cur_pos in range(min_prompt_len, total_len):
        print(
            f"\r* Generating trajectories: {cur_pos-min_prompt_len:>4d}/{total_len-min_prompt_len:>4d}",
            flush=True,
            end="",
        )
        with torch.autocast(device_type=device.type, dtype=dtype):
            logits = model.inference(tokens[:, prev_pos:cur_pos], prev_pos)
        sampling_logits = logits[:, -1].float()
        if pad_token_id is not None:
            sampling_logits[:, pad_token_id] = float("-inf")
        next_token = sample_next_token(sampling_logits, sampling)
        next_token = torch.where(
            input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
        )
        # if an rollout is finished, we fill the rest of the tokens with pad_token_id
        next_token = torch.where(is_finished, pad_token_id, next_token)
        is_generated_token = ~input_text_mask[:, cur_pos]
        with torch.no_grad():
            step_log_probs = torch.log_softmax(logits[:, -1].float(), dim=-1)
            selected_log_probs = step_log_probs.gather(
                1, next_token.unsqueeze(-1)
            ).squeeze(-1)
        # Only record log-probs for tokens that are actually kept in the trajectory.
        # After EOS, positions are still "generated" slots but filled with pad and
        # stripped later — do not append log-probs for those steps.
        for b in range(bsz):
            if is_generated_token[b] and not is_finished[b]:
                generated_log_probs[b].append(selected_log_probs[b].item())
        tokens[:, cur_pos] = next_token
        if end_token_id is not None:
            is_end_token = next_token == end_token_id
            is_finished = is_finished | (is_end_token & is_generated_token)
        prev_pos = cur_pos
        if is_finished.all():
            break
    model.del_kv_cache()
    _flush_cuda_memory(device)
    gc.collect()
    is_finished_list = is_finished.tolist()
    tokens_list = tokens.tolist()
    del tokens, is_finished
    _flush_cuda_memory(device)

    # prepare the output episodes
    episodes = []
    for i in range(bsz // num_answer_per_question):
        for j in range(num_answer_per_question):
            idx = i * num_answer_per_question + j
            prefix_len = len(batch.prefix_token_ids[i])
            generated_token_ids = tokens_list[idx][prefix_len:]
            # remove padding tokens
            if pad_token_id in generated_token_ids:
                generated_token_ids = generated_token_ids[
                    : generated_token_ids.index(pad_token_id)
                ]
            generated_text = tokenizer.detokenize(generated_token_ids)
            ##用于countdown tasks
            # rewards = reward_function(
            #     response=generated_text,
            #     numbers=batch.numbers[i],
            #     target=batch.target[i],
            #     end_token=end_token,
            # )

            ##用于countdown tasks和dapo-math-17k
            # grpo.py — rollout 函数内，替换原有的 rewards = reward_function(...) 调用

            reward_kwargs = {
                "response": generated_text,
                "end_token": end_token,
            }
            # 兼容 CountdownTasksDataset（含 numbers/target）和 DAPOMathDataset（含 ground_truth）
            if hasattr(batch, "ground_truth"):
                reward_kwargs["ground_truth"] = batch.ground_truth[i]
            if hasattr(batch, "numbers"):
                reward_kwargs["numbers"] = batch.numbers[i]
            if hasattr(batch, "target"):
                reward_kwargs["target"] = batch.target[i]

            rewards = reward_function(**reward_kwargs)


            token_log_probs = generated_log_probs[idx]
            if len(token_log_probs) != len(generated_token_ids):
                raise ValueError(
                    "Mismatch between generated tokens and stored log-probs: "
                    f"{len(generated_token_ids)=} vs {len(token_log_probs)=}"
                )
            episode = Episode(
                prefix=batch.prefix[i],
                text=batch.prefix[i] + generated_text,
                prefix_token_ids=batch.prefix_token_ids[i],
                prefix_tokens=batch.prefix_tokens[i],
                generated_token_ids=generated_token_ids,
                generated_token_log_probs=token_log_probs,
                is_finished=is_finished_list[idx],
                reward=rewards["reward"],
                reward_info=rewards["reward_info"],
            )
            episodes.append(episode)
    # clear the output line
    print("\r", end=" " * 100, flush=True)
    return episodes


def normalize_rewards_per_group(episodes: List[Episode]) -> List[Episode]:
    """Normalize rewards per group. A group is defined by the prefix."""
    groups = defaultdict(list)
    for episode in episodes:
        groups[tuple(episode.prefix)].append(episode)
    output = []
    for group in groups.values():
        group_rewards = [item.reward for item in group]
        mean_reward = np.mean(group_rewards)
        std_reward = np.std(group_rewards)
        for episode in group:
            normalized_reward = (episode.reward - mean_reward) / (std_reward + 1e-4)
            episode = dataclasses.replace(episode, reward=normalized_reward)
            output.append(episode)
    return output


def normalize_and_filter_groups(
    episodes: List[Episode],
    std_threshold: float = 1e-6,
    center_advantages: bool = True,
    scale_by_std: bool = True,
    std_epsilon: float = 1e-4,
    drop_zero_adv_groups: bool = True,
) -> tuple[List[Episode], dict[str, float]]:
    """Group-normalize advantages and drop groups with (near) zero reward std.

    A group whose rollouts all receive the same reward (e.g. all correct or all
    wrong) carries no learning signal. If ``drop_zero_adv_groups`` is True, such
    groups are removed to avoid useless forward/backward work and denominator
    dilution. Advantage transformation supports:
      - center + scale (classic)
      - center-only (no std scaling, no epsilon involved)
      - raw reward passthrough (if both center/scale disabled)
    """
    groups = defaultdict(list)
    for episode in episodes:
        groups[tuple(episode.prefix)].append(episode)

    kept: List[Episode] = []
    num_groups = len(groups)
    num_dropped_groups = 0
    num_dropped_episodes = 0
    for group in groups.values():
        group_rewards = [item.reward for item in group]
        std_reward = float(np.std(group_rewards))
        has_signal = std_reward >= std_threshold
        if drop_zero_adv_groups and not has_signal:
            num_dropped_groups += 1
            num_dropped_episodes += len(group)
            continue
        mean_reward = float(np.mean(group_rewards))
        for episode in group:
            advantage = float(episode.reward)
            if center_advantages:
                advantage = advantage - mean_reward
            if scale_by_std:
                # Keep epsilon only for std scaling mode.
                advantage = advantage / (std_reward + std_epsilon)
            kept.append(dataclasses.replace(episode, reward=advantage))

    total_episodes = sum(len(g) for g in groups.values())
    stats = {
        "num_groups": float(num_groups),
        "num_zero_adv_groups": float(num_dropped_groups),
        "num_dropped_episodes": float(num_dropped_episodes),
        "nonzero_adv_frac": (
            1.0 - num_dropped_episodes / total_episodes if total_episodes else 0.0
        ),
    }
    return kept, stats


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(probs * logits, dim=-1)
    return entropy


def compute_token_log_probs(
    logits: torch.Tensor,
    target_token_ids: torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    return -torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target_token_ids.reshape(-1),
        ignore_index=pad_token_id,
        reduction="none",
    ).reshape(target_token_ids.shape)


def build_old_log_probs_tensor(
    batch_episodes: List[Episode],
    target_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Map per-episode rollout log-probs onto target token positions."""
    rows = []
    for episode in batch_episodes:
        row = torch.zeros(target_len, device=device, dtype=torch.float32)
        start = len(episode.prefix_token_ids) - 1
        for k, log_prob in enumerate(episode.generated_token_log_probs):
            pos = start + k
            if 0 <= pos < target_len:
                row[pos] = log_prob
        rows.append(row)
    return torch.stack(rows, dim=0)


def compute_ppo_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    target_masks: torch.Tensor,
    loss_denominator: float,
    clip_eps: float,
    clip_ratio_low: float | None = None,
    clip_ratio_high: float | None = None,
    use_ppo_clip: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Policy surrogate with optional PPO clipping:
      ratio = exp(new_logp - old_logp)
      L = -E[min(ratio * A, clip(ratio) * A)]   if use_ppo_clip
      L = -E[ratio * A]                          otherwise
    """
    eps_low = clip_ratio_low if clip_ratio_low is not None else clip_eps
    eps_high = clip_ratio_high if clip_ratio_high is not None else clip_eps
    log_ratio = new_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    adv = advantages[:, None]
    surr1 = ratio * adv
    if use_ppo_clip:
        surr2 = torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high) * adv
        ppo_obj = torch.minimum(surr1, surr2)
    else:
        ppo_obj = surr1
    masked_obj = ppo_obj * target_masks
    loss = -masked_obj.sum() / max(loss_denominator, 1.0)

    with torch.no_grad():
        num_tokens = target_masks.sum()
        if use_ppo_clip:
            clipped = (ratio < (1.0 - eps_low)) | (ratio > (1.0 + eps_high))
            clip_tokens = (clipped & target_masks).float().sum()
        else:
            clip_tokens = torch.zeros((), device=new_log_probs.device)
        kl_sum = ((old_log_probs - new_log_probs) * target_masks).sum()
        ratio_sum = (ratio * target_masks).sum()

    # Return token-weighted sums so callers can aggregate correctly across
    # micro-batches of different sizes.
    metrics = {
        "num_tokens": num_tokens.item(),
        "clip_tokens": clip_tokens.item(),
        "kl_sum": kl_sum.item(),
        "ratio_sum": ratio_sum.item(),
        "num_responses": 0.0,
        "clip_responses": 0.0,
        "response_ratio_sum": 0.0,
    }
    return loss, metrics


def _compute_len_scaled_eps(
    lengths: torch.Tensor,
    eps: float,
    scaling: str,
    min_scale: float = 1e-6,
) -> torch.Tensor:
    """Length-aware clip scaling.

    - linear: eps / L       (GSPO-style linear normalization)
    - sqrt:   eps / sqrt(L) (FSPO-style)
    - none:   eps
    """
    safe_lengths = torch.clamp(lengths.float(), min=1.0)
    if scaling == "linear":
        scale = 1.0 / safe_lengths
    elif scaling == "sqrt":
        scale = 1.0 / torch.sqrt(safe_lengths)
    elif scaling == "none":
        scale = torch.ones_like(safe_lengths)
    else:
        raise ValueError(f"Unsupported gspo_clip_len_scaling: {scaling}")
    scale = torch.clamp(scale, min=min_scale)
    return eps * scale


def compute_gspo_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    target_masks: torch.Tensor,
    loss_denominator: float,
    clip_eps: float,
    clip_ratio_low: float | None = None,
    clip_ratio_high: float | None = None,
    use_ppo_clip: bool = True,
    gspo_clip_len_scaling: str = "linear",
) -> tuple[torch.Tensor, dict[str, float]]:
    """GSPO-style sequence-level objective (response-wise importance ratio).

    s_i = exp(mean_t [log pi_new - log pi_old]) over generated tokens in response i.
    """
    eps_low = clip_ratio_low if clip_ratio_low is not None else clip_eps
    eps_high = clip_ratio_high if clip_ratio_high is not None else clip_eps

    lengths = torch.clamp(target_masks.sum(dim=1).float(), min=1.0)
    log_ratio = new_log_probs - old_log_probs
    seq_log_ratio = (log_ratio * target_masks).sum(dim=1) / lengths
    seq_ratio = torch.exp(seq_log_ratio)

    surr1 = seq_ratio * advantages
    if use_ppo_clip:
        eps_low_vec = _compute_len_scaled_eps(
            lengths=lengths,
            eps=eps_low,
            scaling=gspo_clip_len_scaling,
        )
        eps_high_vec = _compute_len_scaled_eps(
            lengths=lengths,
            eps=eps_high,
            scaling=gspo_clip_len_scaling,
        )
        clipped_ratio = torch.minimum(
            torch.maximum(seq_ratio, 1.0 - eps_low_vec), 1.0 + eps_high_vec
        )
        surr2 = clipped_ratio * advantages
        gspo_obj = torch.minimum(surr1, surr2)
        clipped_resp = ((seq_ratio < (1.0 - eps_low_vec)) | (seq_ratio > (1.0 + eps_high_vec))).float()
    else:
        gspo_obj = surr1
        clipped_resp = torch.zeros_like(seq_ratio)

    loss = -gspo_obj.sum() / max(loss_denominator, 1.0)

    with torch.no_grad():
        kl_sum = ((old_log_probs - new_log_probs) * target_masks).sum()
        num_tokens = target_masks.sum()
        num_responses = torch.tensor(
            float(target_masks.shape[0]),
            device=target_masks.device,
        )

    metrics = {
        "num_tokens": num_tokens.item(),
        "clip_tokens": 0.0,
        "kl_sum": kl_sum.item(),
        "ratio_sum": 0.0,
        "num_responses": num_responses.item(),
        "clip_responses": clipped_resp.sum().item(),
        "response_ratio_sum": seq_ratio.sum().item(),
    }
    return loss, metrics


def update_policy(
    model,
    optimizer,
    episodes: List[Episode],
    micro_batch_size: int,
    pad_token_id: int,
    max_grad_norm: float,
    device: torch.device,
    dtype: torch.dtype,
    clip_eps: float = 0.2,
    clip_ratio_low: float | None = None,
    clip_ratio_high: float | None = None,
    use_ppo_clip: bool = True,
    ppo_epochs: int = 1,
    advantage_std_threshold: float = 1e-6,
    center_advantages: bool = True,
    scale_advantages_by_std: bool = True,
    advantage_std_epsilon: float = 1e-4,
    drop_zero_adv_groups: bool = True,
    beta: float = 0.0,
    advantage_mode: str = "grpo",
    gspo_clip_len_scaling: str = "linear",
):
    """Update the policy using GRPO advantages + (optional) PPO clipped surrogate.

    Runs ``ppo_epochs`` optimization passes over one rollout batch. The
    ``old_log_probs`` are the behavior-policy log-probs recorded during rollout
    and are held fixed across epochs, so for epoch > 0 the importance ratio
    exp(new_logp - old_logp) departs from 1 and the PPO clipping genuinely
    constrains the update. With a single epoch the ratio stays ~1 and clipping
    is largely inert.
    """
    episodes, adv_stats = normalize_and_filter_groups(
        episodes,
        std_threshold=advantage_std_threshold,
        center_advantages=center_advantages,
        scale_by_std=scale_advantages_by_std,
        std_epsilon=advantage_std_epsilon,
        drop_zero_adv_groups=drop_zero_adv_groups,
    )

    base_metrics = {
        "loss": 0.0,
        "ppo_loss": 0.0,
        "kl_loss": 0.0,
        "grad_norm": 0.0,
        "entropy": 0.0,
        "clip_fraction": 0.0,
        "approx_kl": 0.0,
        "ratio_mean": 0.0,
        "ppo_epochs_ran": 0,
        **adv_stats,
    }
    if not episodes:
        # Every group had ~zero reward std -> no learning signal this step.
        return base_metrics

    # Ensure rollout KV / token buffers are fully released before the first forward.
    _flush_cuda_memory(device)

    # sort episodes by token length for efficient (micro-)batching
    episodes.sort(key=lambda x: len(x.prefix_token_ids) + len(x.generated_token_ids))
    num_target_tokens = sum(len(ep.generated_token_ids) for ep in episodes)
    if num_target_tokens <= 0:
        # No generated target tokens -> skip update to avoid invalid
        # normalization/entropy statistics.
        return {
            **base_metrics,
            "num_target_tokens": 0.0,
        }
    objective_denominator = (
        float(max(num_target_tokens, 1))
        if advantage_mode == "grpo"
        else float(max(len(episodes), 1))
    )
    kl_denominator = float(max(num_target_tokens, 1))
    micro_batches = [
        episodes[i : i + micro_batch_size]
        for i in range(0, len(episodes), micro_batch_size)
    ]

    last_metrics = base_metrics
    for epoch in range(max(ppo_epochs, 1)):
        optimizer.zero_grad(set_to_none=True)
        entropy = 0.0
        loss_sum = 0.0
        ppo_loss_sum = 0.0
        kl_loss_sum = 0.0
        tok_total = 0.0
        resp_total = 0.0
        clip_resp_total = 0.0
        clip_tok_total = 0.0
        kl_total = 0.0
        ratio_total = 0.0
        response_ratio_total = 0.0

        for mb_idx, batch_episodes in enumerate(micro_batches):
            print(
                f"\r* Policy gradient epoch {epoch + 1}/{max(ppo_epochs, 1)} "
                f"micro-batch {mb_idx + 1}/{len(micro_batches)}",
                flush=True,
                end="",
            )
            batch_lengths = [
                len(ep.prefix_token_ids) + len(ep.generated_token_ids)
                for ep in batch_episodes
            ]
            batch_max_length = max(batch_lengths)
            batch_token_ids = [
                ep.prefix_token_ids
                + ep.generated_token_ids
                + [pad_token_id] * (batch_max_length - batch_lengths[k])
                for k, ep in enumerate(batch_episodes)
            ]
            batch_masks = [
                [0] * len(ep.prefix_token_ids)
                + [1] * len(ep.generated_token_ids)
                + [0] * (batch_max_length - batch_lengths[k])
                for k, ep in enumerate(batch_episodes)
            ]
            batch_advantages = [ep.reward for ep in batch_episodes]
            batch_token_ids = torch.tensor(
                batch_token_ids, device=device, dtype=torch.long
            )
            batch_masks = torch.tensor(batch_masks, device=device, dtype=torch.bool)
            batch_advantages = torch.tensor(
                batch_advantages, device=device, dtype=torch.float32
            )

            with torch.autocast(device_type=device.type, dtype=dtype):
                input_token_ids = batch_token_ids[:, :-1]
                target_token_ids = batch_token_ids[:, 1:]
                target_masks = batch_masks[:, 1:]
                logits = model.forward(input_token_ids).float()

            new_log_probs = -torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_token_ids.reshape(-1),
                ignore_index=pad_token_id,
                reduction="none",
            ).reshape(input_token_ids.shape[0], -1)

            old_log_probs = build_old_log_probs_tensor(
                batch_episodes,
                target_len=new_log_probs.shape[1],
                device=device,
            ).detach()

            with torch.no_grad():
                token_entropy = compute_entropy(logits)
                entropy = entropy + (token_entropy * target_masks).sum() / max(
                    num_target_tokens, 1
                )

            if advantage_mode == "grpo":
                ppo_loss, ppo_metrics = compute_ppo_loss(
                    new_log_probs=new_log_probs,
                    old_log_probs=old_log_probs,
                    advantages=batch_advantages,
                    target_masks=target_masks,
                    loss_denominator=objective_denominator,
                    clip_eps=clip_eps,
                    clip_ratio_low=clip_ratio_low,
                    clip_ratio_high=clip_ratio_high,
                    use_ppo_clip=use_ppo_clip,
                )
            elif advantage_mode == "gspo":
                ppo_loss, ppo_metrics = compute_gspo_loss(
                    new_log_probs=new_log_probs,
                    old_log_probs=old_log_probs,
                    advantages=batch_advantages,
                    target_masks=target_masks,
                    loss_denominator=objective_denominator,
                    clip_eps=clip_eps,
                    clip_ratio_low=clip_ratio_low,
                    clip_ratio_high=clip_ratio_high,
                    use_ppo_clip=use_ppo_clip,
                    gspo_clip_len_scaling=gspo_clip_len_scaling,
                )
            else:
                raise ValueError(
                    f"advantage_mode must be one of ['grpo', 'gspo'], got: {advantage_mode}"
                )
            kl_penalty = ((old_log_probs - new_log_probs) * target_masks).sum() / max(
                kl_denominator, 1.0
            )
            loss = ppo_loss + beta * kl_penalty
            loss.backward()

            # loss is already divided by the global token denominator, so the sum
            # of per-micro-batch losses equals the full-batch per-token loss.
            loss_sum += loss.item()
            ppo_loss_sum += ppo_loss.item()
            kl_loss_sum += kl_penalty.item()
            tok_total += ppo_metrics["num_tokens"]
            resp_total += ppo_metrics.get("num_responses", 0.0)
            clip_resp_total += ppo_metrics.get("clip_responses", 0.0)
            clip_tok_total += ppo_metrics["clip_tokens"]
            kl_total += ppo_metrics["kl_sum"]
            ratio_total += ppo_metrics["ratio_sum"]
            response_ratio_total += ppo_metrics.get("response_ratio_sum", 0.0)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=max_grad_norm
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        tok_denom = max(tok_total, 1.0)
        resp_denom = max(resp_total, 1.0)
        if advantage_mode == "gspo":
            clip_fraction = clip_resp_total / resp_denom
            ratio_mean = response_ratio_total / resp_denom
        else:
            clip_fraction = clip_tok_total / tok_denom
            ratio_mean = ratio_total / tok_denom
        last_metrics = {
            "loss": loss_sum,
            "ppo_loss": ppo_loss_sum,
            "kl_loss": kl_loss_sum,
            "grad_norm": grad_norm.item(),
            "entropy": float(entropy.item() if torch.is_tensor(entropy) else entropy),
            "clip_fraction": clip_fraction,
            "approx_kl": kl_total / tok_denom,
            "ratio_mean": ratio_mean,
            "num_target_tokens": float(tok_total),
            "num_responses": float(resp_total),
            "advantage_mode": advantage_mode,
            "ppo_epochs_ran": epoch + 1,
            **adv_stats,
        }

    return last_metrics
