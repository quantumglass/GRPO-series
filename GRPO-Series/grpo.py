import dataclasses
import gc
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, List

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


def _resolve_clip_bounds(
    clip_eps: float | str,
    clip_ratio_low: float | str | None = None,
    clip_ratio_high: float | str | None = None,
) -> tuple[float, float]:
    """Resolve asymmetric clip bounds; coerce YAML strings like ``3e-4`` to float."""
    base = float(clip_eps)
    eps_low = float(clip_ratio_low) if clip_ratio_low is not None else base
    eps_high = float(clip_ratio_high) if clip_ratio_high is not None else base
    return eps_low, eps_high


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
    end_token: str | None = None,
    end_token_id: int | None = None,
    stop_token_ids: list[int] | None = None,
    treat_max_length_as_finished: bool = False,
) -> List[Episode]:
    sampling = sampling or SamplingConfig()
    if end_token is None:
        end_token = tokenizer.eos_token
    if end_token_id is None:
        end_token_id = tokenizer.eos_token_id
    if stop_token_ids is None:
        stop_token_ids = [end_token_id] if end_token_id is not None else []
    else:
        stop_token_ids = [tid for tid in stop_token_ids if tid is not None]
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
            # pad 与 eos 相同时，仍需允许采样停止符
            for stop_id in stop_token_ids:
                if stop_id == pad_token_id:
                    sampling_logits[:, stop_id] = logits[:, -1].float()[:, stop_id]
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
        if stop_token_ids:
            is_end_token = torch.zeros_like(is_finished)
            for stop_id in stop_token_ids:
                is_end_token = is_end_token | (next_token == stop_id)
            is_finished = is_finished | (is_end_token & is_generated_token)
        elif end_token_id is not None:
            is_end_token = next_token == end_token_id
            is_finished = is_finished | (is_end_token & is_generated_token)
        prev_pos = cur_pos
        if is_finished.all():
            break
    model.del_kv_cache()
    _flush_cuda_memory(device)
    gc.collect()
    is_finished_list = is_finished.tolist()
    if treat_max_length_as_finished:
        is_finished_list = [True] * bsz
    tokens_list = tokens.tolist()
    del tokens, is_finished
    _flush_cuda_memory(device)

    # prepare the output episodes
    episodes = []
    for i in range(bsz // num_answer_per_question):
        for j in range(num_answer_per_question):
            idx = i * num_answer_per_question + j
            prefix_len = len(batch.prefix_token_ids[i])
            token_log_probs = generated_log_probs[idx]
            # Slice by recorded log-prob count: each generated step (incl. stop
            # tokens) appends exactly one log-prob.  Do not truncate at pad_token_id
            # — on Qwen, pad == eos (<|endoftext|>), so index(pad) would drop the
            # final stop token while its log-prob remains (off-by-one crash).
            gen_len = len(token_log_probs)
            generated_token_ids = tokens_list[idx][prefix_len : prefix_len + gen_len]
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


def group_episodes_by_prefix(episodes: List[Episode]) -> dict[tuple[str, ...], list[Episode]]:
    groups: dict[tuple[str, ...], list[Episode]] = defaultdict(list)
    for episode in episodes:
        groups[tuple(episode.prefix)].append(episode)
    return groups


def group_has_learning_signal(group_rewards: list[float], std_threshold: float) -> bool:
    return float(np.std(group_rewards)) >= std_threshold


def split_groups_by_signal(
    episodes: List[Episode],
    std_threshold: float,
    drop_zero_adv_groups: bool,
) -> tuple[list[list[Episode]], list[list[Episode]]]:
    """Split episode groups into kept vs dropped by reward std."""
    kept_groups: list[list[Episode]] = []
    dropped_groups: list[list[Episode]] = []
    for group in group_episodes_by_prefix(episodes).values():
        rewards = [float(item.reward) for item in group]
        if drop_zero_adv_groups and not group_has_learning_signal(rewards, std_threshold):
            dropped_groups.append(group)
        else:
            kept_groups.append(group)
    return kept_groups, dropped_groups


def flatten_episode_groups(groups: list[list[Episode]]) -> List[Episode]:
    output: List[Episode] = []
    for group in groups:
        output.extend(group)
    return output


def _group_reward_std(group: list[Episode]) -> float:
    if not group:
        return 0.0
    return float(np.std([float(item.reward) for item in group]))


def summarize_group_reward_std_stats(
    kept_groups: list[list[Episode]],
    dropped_groups: list[list[Episode]],
) -> dict[str, float]:
    """Aggregate per-group raw reward std for monitoring adv-drop health."""
    all_groups = kept_groups + dropped_groups
    all_stds = [_group_reward_std(group) for group in all_groups]
    kept_stds = [_group_reward_std(group) for group in kept_groups]
    dropped_stds = [_group_reward_std(group) for group in dropped_groups]

    def _mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    def _median(values: list[float]) -> float:
        return float(np.median(values)) if values else 0.0

    num_groups = len(all_groups)
    return {
        "group_reward_std_mean": _mean(all_stds),
        "group_reward_std_median": _median(all_stds),
        "group_reward_std_min": float(min(all_stds)) if all_stds else 0.0,
        "group_reward_std_max": float(max(all_stds)) if all_stds else 0.0,
        "kept_group_reward_std_mean": _mean(kept_stds),
        "dropped_group_reward_std_mean": _mean(dropped_stds),
        "group_below_threshold_frac": (
            float(len(dropped_groups)) / num_groups if num_groups else 0.0
        ),
    }


def preview_episodes_after_adv_drop(
    episodes: List[Episode],
    std_threshold: float,
    drop_zero_adv_groups: bool,
) -> List[Episode]:
    """Return episodes that would survive ``drop_zero_adv_groups`` (raw rewards)."""
    if not drop_zero_adv_groups:
        return list(episodes)
    kept_groups, _ = split_groups_by_signal(
        episodes,
        std_threshold=std_threshold,
        drop_zero_adv_groups=True,
    )
    return flatten_episode_groups(kept_groups)


def trim_groups_to_target_episodes(
    groups: list[list[Episode]],
    target_episode_count: int,
) -> tuple[List[Episode], int]:
    """Keep whole groups up to ``target_episode_count`` episodes."""
    if target_episode_count <= 0:
        return [], 0
    selected: list[list[Episode]] = []
    count = 0
    for group in groups:
        group_size = len(group)
        if count + group_size > target_episode_count:
            break
        selected.append(group)
        count += group_size
    return flatten_episode_groups(selected), count


@dataclass(frozen=True)
class AdvZeroDropReplenishConfig:
    """Dynamic rollout replenishment when zero-advantage groups are dropped."""

    enabled: bool = False
    target_batch_size: int = 0
    max_rounds: int = 8
    std_threshold: float = 1e-4


def adv_zero_drop_replenish_config_from_dict(
    training_cfg: dict[str, Any],
    default_batch_size: int,
) -> AdvZeroDropReplenishConfig:
    drop_zero_adv_groups = bool(training_cfg.get("drop_zero_adv_groups", True))
    enabled = bool(training_cfg.get("adv_zero_drop_replenish", False))
    if enabled and not drop_zero_adv_groups:
        raise ValueError(
            "adv_zero_drop_replenish requires drop_zero_adv_groups=true"
        )
    target = training_cfg.get("adv_zero_drop_target_batch_size", default_batch_size)
    return AdvZeroDropReplenishConfig(
        enabled=enabled and drop_zero_adv_groups,
        target_batch_size=int(target),
        max_rounds=max(0, int(training_cfg.get("adv_zero_drop_max_rounds", 8))),
        std_threshold=float(training_cfg.get("advantage_std_threshold", 1e-4)),
    )


def collect_episodes_with_adv_zero_drop_replenish(
    episodes: List[Episode],
    *,
    replenish_cfg: AdvZeroDropReplenishConfig,
    rollout_more: Callable[[], List[Episode]] | None = None,
) -> tuple[List[Episode], dict[str, float]]:
    """Replenish rollouts until enough signal-bearing episodes for stable gradients.

  When ``drop_zero_adv_groups`` removes many groups, the effective update batch
  shrinks and gradient variance grows. This helper optionally performs extra
  rollout rounds and returns only groups that would survive adv-drop, trimmed
  to ``target_batch_size`` complete groups.
    """
    pool = list(episodes)
    initial_usable = preview_episodes_after_adv_drop(
        pool,
        std_threshold=replenish_cfg.std_threshold,
        drop_zero_adv_groups=True,
    )
    stats: dict[str, float] = {
        "adv_zero_drop_replenish_rounds": 0.0,
        "adv_zero_drop_initial_usable": float(len(initial_usable)),
        "adv_zero_drop_final_usable": float(len(initial_usable)),
        "adv_zero_drop_target": float(replenish_cfg.target_batch_size),
        "adv_zero_drop_reached_target": float(
            len(initial_usable) >= replenish_cfg.target_batch_size
        ),
        "adv_zero_drop_pool_episodes": float(len(pool)),
    }
    if not replenish_cfg.enabled:
        return pool, stats

    usable = initial_usable
    rounds = 0
    while (
        len(usable) < replenish_cfg.target_batch_size
        and rounds < replenish_cfg.max_rounds
    ):
        if rollout_more is None:
            break
        rounds += 1
        extra = rollout_more()
        if not extra:
            break
        pool.extend(extra)
        usable = preview_episodes_after_adv_drop(
            pool,
            std_threshold=replenish_cfg.std_threshold,
            drop_zero_adv_groups=True,
        )

    kept_groups, dropped_groups = split_groups_by_signal(
        pool,
        std_threshold=replenish_cfg.std_threshold,
        drop_zero_adv_groups=True,
    )
    if len(usable) >= replenish_cfg.target_batch_size:
        final_episodes, final_count = trim_groups_to_target_episodes(
            kept_groups,
            replenish_cfg.target_batch_size,
        )
    else:
        final_episodes = usable
        final_count = len(final_episodes)

    stats.update(
        {
            "adv_zero_drop_replenish_rounds": float(rounds),
            "adv_zero_drop_final_usable": float(final_count),
            "adv_zero_drop_reached_target": float(
                final_count >= replenish_cfg.target_batch_size
            ),
            "adv_zero_drop_pool_episodes": float(len(pool)),
            "num_zero_adv_groups": float(len(dropped_groups)),
            "num_dropped_episodes": float(
                sum(len(group) for group in dropped_groups)
            ),
        }
    )
    return final_episodes, stats


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
    kept_groups, dropped_groups = split_groups_by_signal(
        episodes,
        std_threshold=std_threshold,
        drop_zero_adv_groups=drop_zero_adv_groups,
    )

    kept: List[Episode] = []
    num_groups = len(kept_groups) + len(dropped_groups)
    num_dropped_groups = len(dropped_groups)
    num_dropped_episodes = sum(len(group) for group in dropped_groups)
    std_stats = summarize_group_reward_std_stats(kept_groups, dropped_groups)
    for group in kept_groups:
        group_rewards = [item.reward for item in group]
        std_reward = float(np.std(group_rewards))
        mean_reward = float(np.mean(group_rewards))
        for episode in group:
            advantage = float(episode.reward)
            if center_advantages:
                advantage = advantage - mean_reward
            if scale_by_std:
                # Keep epsilon only for std scaling mode.
                advantage = advantage / (std_reward + std_epsilon)
            kept.append(dataclasses.replace(episode, reward=advantage))

    advantage_values = [float(episode.reward) for episode in kept]
    advantage_stats = {
        "advantage_std": float(np.std(advantage_values)) if advantage_values else 0.0,
        "advantage_abs_mean": (
            float(np.mean(np.abs(advantage_values))) if advantage_values else 0.0
        ),
    }

    total_episodes = sum(len(group) for group in kept_groups) + num_dropped_episodes
    stats = {
        "num_groups": float(num_groups),
        "num_zero_adv_groups": float(num_dropped_groups),
        "num_dropped_episodes": float(num_dropped_episodes),
        "nonzero_adv_frac": (
            1.0 - num_dropped_episodes / total_episodes if total_episodes else 0.0
        ),
        **std_stats,
        **advantage_stats,
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


def build_past_log_probs_tensor(
    batch_episodes: List[Episode],
    target_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Map π_θ_past log-probs for replay trajectories onto target positions."""
    rows = []
    for episode in batch_episodes:
        row = torch.zeros(target_len, device=device, dtype=torch.float32)
        if not episode.is_replay or not episode.past_token_log_probs:
            rows.append(row)
            continue
        start = len(episode.prefix_token_ids) - 1
        for k, log_prob in enumerate(episode.past_token_log_probs):
            pos = start + k
            if 0 <= pos < target_len:
                row[pos] = log_prob
        rows.append(row)
    return torch.stack(rows, dim=0)


def compute_exgrpo_policy_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    past_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    target_masks: torch.Tensor,
    replay_mask: torch.Tensor,
    episode_weights: torch.Tensor,
    shaping_beta: float,
    clip_eps: float,
    clip_ratio_low: float | None = None,
    clip_ratio_high: float | None = None,
    use_ppo_clip: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """ExGRPO Eq. 4 + Dr.GRPO token aggregation.

    Per trajectory: sum_t surrogate (no 1/|o| length norm).
    Per group/question: uniform 1/|G| over trajectories via ``episode_weights``.
    Global mix: (1-ρ) on B_on + ρ on B_exp, encoded in ``episode_weights``.
    Replay trajectory uses policy shaping f(w)=w/(w+β) with w=π_θ/π_θ_past.
    """
    eps_low, eps_high = _resolve_clip_bounds(clip_eps, clip_ratio_low, clip_ratio_high)
    adv = advantages[:, None]
    replay_token_mask = replay_mask[:, None] & target_masks
    onpolicy_token_mask = (~replay_mask[:, None]) & target_masks

    on_log_ratio = new_log_probs - old_log_probs
    on_ratio = torch.exp(on_log_ratio)
    on_surr1 = on_ratio * adv
    if use_ppo_clip:
        on_surr2 = torch.clamp(on_ratio, 1.0 - eps_low, 1.0 + eps_high) * adv
        on_obj = torch.minimum(on_surr1, on_surr2)
        clip_tokens = (
            _ppo_clip_active(on_surr1, on_surr2) & onpolicy_token_mask
        ).float().sum()
    else:
        on_obj = on_surr1
        clip_tokens = torch.zeros((), device=new_log_probs.device)

    # Replay: w* = π_θ(current) / π_θ_past (stored at buffer collection time).
    replay_log_ratio = new_log_probs - past_log_probs
    replay_ratio = torch.exp(replay_log_ratio)
    shaping = replay_ratio / (replay_ratio + shaping_beta)
    replay_obj = shaping * adv

    mixed_obj = torch.where(replay_token_mask, replay_obj, on_obj)
    token_surr = mixed_obj * target_masks
    per_episode_obj = token_surr.sum(dim=1)
    loss = -(per_episode_obj * episode_weights).sum()

    with torch.no_grad():
        num_tokens = target_masks.sum()
        on_tokens = onpolicy_token_mask.float().sum()
        replay_tokens = replay_token_mask.float().sum()
        behavior_log_probs = torch.where(replay_token_mask, past_log_probs, old_log_probs)
        kl_sum = ((behavior_log_probs - new_log_probs) * target_masks).sum()
        ratio_sum = (on_ratio * onpolicy_token_mask).sum() + (
            replay_ratio * replay_token_mask
        ).sum()
        replay_episode_weight = (
            float(episode_weights[replay_mask].sum().item())
            if replay_mask.any()
            else 0.0
        )
        on_episode_weight = (
            float(episode_weights[~replay_mask].sum().item())
            if (~replay_mask).any()
            else 0.0
        )

    metrics = {
        "num_tokens": num_tokens.item(),
        "clip_tokens": clip_tokens.item(),
        "kl_sum": kl_sum.item(),
        "ratio_sum": ratio_sum.item(),
        "num_responses": 0.0,
        "clip_responses": 0.0,
        "response_ratio_sum": 0.0,
        "replay_tokens": replay_tokens.item(),
        "onpolicy_tokens": on_tokens.item(),
        "exgrpo_replay_episode_weight_sum": replay_episode_weight,
        "exgrpo_on_episode_weight_sum": on_episode_weight,
    }
    return loss, metrics


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
    eps_low, eps_high = _resolve_clip_bounds(clip_eps, clip_ratio_low, clip_ratio_high)
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
            clip_tokens = (
                _ppo_clip_active(surr1, surr2) & target_masks
            ).float().sum()
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


def _ppo_clip_active(surr1: torch.Tensor, surr2: torch.Tensor) -> torch.Tensor:
    """True where the clipped surrogate is tighter (verl ``pg_clipfrac`` semantics)."""
    return surr2 < surr1


def _normalize_gspo_clip_scaling(scaling: str) -> str:
    """Map config alias to internal clip mode: ``none`` (GSPO) or ``sqrt`` (FSPO)."""
    mode = str(scaling).lower()
    if mode in {"sqrt", "fspo"}:
        return "sqrt"
    if mode in {"none", "gspo"}:
        return "none"
    if mode == "linear":
        raise ValueError(
            "gspo_clip_len_scaling='linear' is removed; use 'none' (GSPO ratio clip) "
            "or 'sqrt' (FSPO log-space clip with c*sqrt(L) band)."
        )
    raise ValueError(
        f"Unsupported gspo_clip_len_scaling: {scaling!r}; expected none|gspo|sqrt|fspo"
    )


def _sequence_importance_from_log_ratio(
    log_ratio: torch.Tensor,
    target_masks: torch.Tensor,
    scaling: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build sequence-level log importance and ratio.

    - GSPO (``none``): S = mean_t log r_t,  ratio = exp(S)
    - FSPO (``sqrt``): S = sum_t log r_t,   ratio = exp(S)  (arXiv:2509.09177 Eq. 3–4)
    """
    lengths = torch.clamp(target_masks.sum(dim=1).float(), min=1.0)
    seq_log_sum = (log_ratio * target_masks).sum(dim=1)
    if scaling == "sqrt":
        seq_log = seq_log_sum
    else:
        seq_log = seq_log_sum / lengths
    return seq_log, torch.exp(seq_log), lengths


def _clipped_sequence_objective(
    seq_log: torch.Tensor,
    seq_ratio: torch.Tensor,
    advantages: torch.Tensor,
    lengths: torch.Tensor,
    eps_low: float,
    eps_high: float,
    scaling: str,
    use_ppo_clip: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PPO pessimistic surrogate at sequence level (GSPO or FSPO clip semantics)."""
    surr1 = seq_ratio * advantages
    if not use_ppo_clip:
        return surr1, torch.zeros_like(seq_ratio)

    if scaling == "sqrt":
        # FSPO: clip log-IS sum S to [-c_lower*sqrt(L), c_upper*sqrt(L)], mu_hat=0.
        sqrt_l = torch.sqrt(lengths)
        b_low = float(eps_low) * sqrt_l
        b_high = float(eps_high) * sqrt_l
        clipped_log = torch.minimum(torch.maximum(seq_log, -b_low), b_high)
        clipped_ratio = torch.exp(clipped_log)
    else:
        # GSPO: clip ratio in probability space around 1.
        low = 1.0 - float(eps_low)
        high = 1.0 + float(eps_high)
        clipped_ratio = torch.clamp(seq_ratio, min=low, max=high)

    surr2 = clipped_ratio * advantages
    obj = torch.minimum(surr1, surr2)
    clipped_resp = _ppo_clip_active(surr1, surr2).float()
    return obj, clipped_resp


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
    gspo_clip_len_scaling: str = "none",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Sequence-level policy surrogate (GSPO or FSPO clip mode).

    GSPO (``gspo_clip_len_scaling=none``):
      s_i = exp(mean_t log pi_new/pi_old), clip(s_i, 1-eps_low, 1+eps_high).
      eps ~ 3e-4 / 4e-4 (arXiv:2507.18071).

    FSPO (``gspo_clip_len_scaling=sqrt``):
      S_i = sum_t log pi_new/pi_old, clip(S_i, -c_low*sqrt(L), c_high*sqrt(L)),
      ratio = exp(S_i).  c ~ 0.03 in log-space (arXiv:2509.09177 Eq. 3–5).
    """
    clip_mode = _normalize_gspo_clip_scaling(gspo_clip_len_scaling)
    eps_low, eps_high = _resolve_clip_bounds(clip_eps, clip_ratio_low, clip_ratio_high)

    log_ratio = new_log_probs - old_log_probs
    seq_log, seq_ratio, lengths = _sequence_importance_from_log_ratio(
        log_ratio, target_masks, clip_mode
    )
    gspo_obj, clipped_resp = _clipped_sequence_objective(
        seq_log=seq_log,
        seq_ratio=seq_ratio,
        advantages=advantages,
        lengths=lengths,
        eps_low=eps_low,
        eps_high=eps_high,
        scaling=clip_mode,
        use_ppo_clip=use_ppo_clip,
    )

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


def compute_exgrpo_gspo_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    past_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    target_masks: torch.Tensor,
    replay_mask: torch.Tensor,
    episode_weights: torch.Tensor,
    shaping_beta: float,
    clip_eps: float,
    clip_ratio_low: float | None = None,
    clip_ratio_high: float | None = None,
    use_ppo_clip: bool = True,
    gspo_clip_len_scaling: str = "none",
) -> tuple[torch.Tensor, dict[str, float]]:
    """ExGRPO Eq. 4 with sequence-level IS (GSPO or FSPO on-policy clip).

    On-policy: GSPO mean-ratio or FSPO sum-log-ratio with mode-specific clip.
    Replay: w*_i = exp(S_i) / (exp(S_i)+beta) using the same IS definition (no clip).
    """
    clip_mode = _normalize_gspo_clip_scaling(gspo_clip_len_scaling)
    eps_low, eps_high = _resolve_clip_bounds(clip_eps, clip_ratio_low, clip_ratio_high)

    on_log_ratio = new_log_probs - old_log_probs
    on_seq_log, on_seq_ratio, lengths = _sequence_importance_from_log_ratio(
        on_log_ratio, target_masks, clip_mode
    )
    on_obj, clipped_resp = _clipped_sequence_objective(
        seq_log=on_seq_log,
        seq_ratio=on_seq_ratio,
        advantages=advantages,
        lengths=lengths,
        eps_low=eps_low,
        eps_high=eps_high,
        scaling=clip_mode,
        use_ppo_clip=use_ppo_clip,
    )

    replay_log_ratio = new_log_probs - past_log_probs
    _, replay_seq_ratio, _ = _sequence_importance_from_log_ratio(
        replay_log_ratio, target_masks, clip_mode
    )
    shaping = replay_seq_ratio / (replay_seq_ratio + shaping_beta)
    replay_obj = shaping * advantages

    per_episode_obj = torch.where(replay_mask, replay_obj, on_obj)
    loss = -(per_episode_obj * episode_weights).sum()

    with torch.no_grad():
        num_tokens = target_masks.sum()
        onpolicy_mask = ~replay_mask
        replay_episode_weight = (
            float(episode_weights[replay_mask].sum().item())
            if replay_mask.any()
            else 0.0
        )
        on_episode_weight = (
            float(episode_weights[onpolicy_mask].sum().item())
            if onpolicy_mask.any()
            else 0.0
        )
        kl_sum = 0.0
        if replay_mask.any() or onpolicy_mask.any():
            behavior_log_probs = torch.where(
                replay_mask[:, None], past_log_probs, old_log_probs
            )
            kl_sum = ((behavior_log_probs - new_log_probs) * target_masks).sum()
        num_responses = torch.tensor(
            float(target_masks.shape[0]),
            device=target_masks.device,
        )
        on_clip_resp = clipped_resp * (~replay_mask).float()
        ratio_sum = (on_seq_ratio * (~replay_mask).float()).sum() + (
            replay_seq_ratio * replay_mask.float()
        ).sum()

    metrics = {
        "num_tokens": num_tokens.item(),
        "clip_tokens": 0.0,
        "kl_sum": kl_sum.item(),
        "ratio_sum": ratio_sum.item(),
        "num_responses": num_responses.item(),
        "clip_responses": on_clip_resp.sum().item(),
        "response_ratio_sum": ratio_sum.item(),
        "replay_tokens": (target_masks * replay_mask[:, None]).float().sum().item(),
        "onpolicy_tokens": (target_masks * (~replay_mask[:, None])).float().sum().item(),
        "exgrpo_replay_episode_weight_sum": replay_episode_weight,
        "exgrpo_on_episode_weight_sum": on_episode_weight,
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
    gspo_clip_len_scaling: str = "none",
    use_exgrpo_loss: bool = False,
    exgrpo_shaping_beta: float = 0.1,
    exgrpo_rho: float = 0.5,
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

    exgrpo_weight_stats: dict[str, float] = {}
    episode_weights: list[float] | None = None
    if use_exgrpo_loss:
        from exgrpo import (
            build_group_preserving_micro_batches,
            compute_exgrpo_episode_weights,
        )

        episode_weights, exgrpo_weight_stats = compute_exgrpo_episode_weights(
            episodes, rho=exgrpo_rho
        )
        paired = sorted(
            zip(episodes, episode_weights),
            key=lambda item: len(item[0].prefix_token_ids)
            + len(item[0].generated_token_ids),
        )
        episodes = [item[0] for item in paired]
        episode_weights = [item[1] for item in paired]
        micro_batches = build_group_preserving_micro_batches(
            episodes, micro_batch_size
        )
    else:
        micro_batches = [
            episodes[i : i + micro_batch_size]
            for i in range(0, len(episodes), micro_batch_size)
        ]

    last_metrics = {**base_metrics, **exgrpo_weight_stats}
    # Rollout records log-probs via incremental KV-cache inference; training
    # recomputes them with full-sequence forward. The per-token drift (~1e-3)
    # dwarfs GSPO's tight clip band (3e-4). Freeze behavior log-probs from the
    # first training forward (epoch 0, before any optimizer step).
    behavior_old_log_probs_cache: dict[int, torch.Tensor] = {}

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

            if mb_idx not in behavior_old_log_probs_cache:
                behavior_old_log_probs_cache[mb_idx] = new_log_probs.detach()
            old_log_probs = behavior_old_log_probs_cache[mb_idx]

            past_log_probs = build_past_log_probs_tensor(
                batch_episodes,
                target_len=new_log_probs.shape[1],
                device=device,
            ).detach()
            replay_mask = torch.tensor(
                [ep.is_replay for ep in batch_episodes],
                device=device,
                dtype=torch.bool,
            )

            with torch.no_grad():
                token_entropy = compute_entropy(logits)
                entropy = entropy + (token_entropy * target_masks).sum() / max(
                    num_target_tokens, 1
                )

            if advantage_mode == "grpo":
                if use_exgrpo_loss and episode_weights is not None:
                    batch_weight_offset = sum(
                        len(micro_batches[j]) for j in range(mb_idx)
                    )
                    batch_episode_weights = torch.tensor(
                        episode_weights[
                            batch_weight_offset : batch_weight_offset
                            + len(batch_episodes)
                        ],
                        device=device,
                        dtype=torch.float32,
                    )
                    ppo_loss, ppo_metrics = compute_exgrpo_policy_loss(
                        new_log_probs=new_log_probs,
                        old_log_probs=old_log_probs,
                        past_log_probs=past_log_probs,
                        advantages=batch_advantages,
                        target_masks=target_masks,
                        replay_mask=replay_mask,
                        episode_weights=batch_episode_weights,
                        shaping_beta=exgrpo_shaping_beta,
                        clip_eps=clip_eps,
                        clip_ratio_low=clip_ratio_low,
                        clip_ratio_high=clip_ratio_high,
                        use_ppo_clip=use_ppo_clip,
                    )
                else:
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
                if use_exgrpo_loss and episode_weights is not None:
                    batch_weight_offset = sum(
                        len(micro_batches[j]) for j in range(mb_idx)
                    )
                    batch_episode_weights = torch.tensor(
                        episode_weights[
                            batch_weight_offset : batch_weight_offset
                            + len(batch_episodes)
                        ],
                        device=device,
                        dtype=torch.float32,
                    )
                    ppo_loss, ppo_metrics = compute_exgrpo_gspo_loss(
                        new_log_probs=new_log_probs,
                        old_log_probs=old_log_probs,
                        past_log_probs=past_log_probs,
                        advantages=batch_advantages,
                        target_masks=target_masks,
                        replay_mask=replay_mask,
                        episode_weights=batch_episode_weights,
                        shaping_beta=exgrpo_shaping_beta,
                        clip_eps=clip_eps,
                        clip_ratio_low=clip_ratio_low,
                        clip_ratio_high=clip_ratio_high,
                        use_ppo_clip=use_ppo_clip,
                        gspo_clip_len_scaling=gspo_clip_len_scaling,
                    )
                else:
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
            if use_exgrpo_loss and episode_weights is not None:
                behavior_log_probs = torch.where(
                    replay_mask[:, None], past_log_probs, old_log_probs
                )
            else:
                behavior_log_probs = old_log_probs
            kl_penalty = ((behavior_log_probs - new_log_probs) * target_masks).sum() / max(
                kl_denominator, 1.0
            )
            loss = ppo_loss + beta * kl_penalty
            loss.backward()

            # Per-micro-batch losses are additive because each micro-batch owns
            # a disjoint subset of episodes/tokens.
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

            del (
                logits,
                new_log_probs,
                old_log_probs,
                past_log_probs,
                batch_token_ids,
                batch_masks,
                input_token_ids,
                target_token_ids,
                target_masks,
                batch_advantages,
                replay_mask,
                loss,
                ppo_loss,
                kl_penalty,
            )
            if use_exgrpo_loss and episode_weights is not None:
                del batch_episode_weights
            _flush_cuda_memory(device)

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
            **exgrpo_weight_stats,
        }

    return last_metrics
