"""
ExGRPO + GSPO 训练入口。

特性:
  1. 奖励 = accuracy + format（无 thinking 长度 bonus）
  2. R1 风格 prefix 引导推理格式
  3. 分块 rollout 降低 KV cache 峰值
  4. ExGRPO 经验回放 + GSPO 序列级策略更新
"""

import dataclasses
import html
import math
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from benchmark_task import answer_matches, extract_pred_answer
from checkpoint import (
    add_resume_arguments,
    format_run_timestamp,
    load_lora_training_checkpoint,
    resolve_resume_paths,
    save_run_config_snapshot,
    save_training_checkpoint,
)
from competition_math_task import CompetitionMathDataset
from countdown_task import CountdownTasksDataset
from dapo_math_task import DAPOMathDataset
from deepscaler_task import DeepScalerDataset
from grpo import (
    adv_zero_drop_replenish_config_from_dict,
    collect_episodes_with_adv_zero_drop_replenish,
    update_policy,
)
from exgrpo import (
    ExGRPOConfig,
    ExGRPOManager,
    compute_pass_at_1,
    format_exgrpo_storage_summary,
    merge_replay_and_fresh_rollouts,
    question_id_from_prefix,
)
from grpo_efficient import rollout_chunked
from lora import (
    LoRAConfig,
    apply_lora_to_model,
    count_parameters,
    freeze_non_lora_parameters,
    get_trainable_parameters,
)
from model_registry import (
    MODEL_PRESETS,
    resolve_model_config,
    validate_training_seq_limits,
)
from optimizer import MemoryEfficientAdamW
from qwen2_model import Transformer
from exgrpo_reward import (
    build_exgrpo_reward_function,
    exgrpo_reward_config_from_dict,
)
from exgrpo_training_hooks import ExGRPOTrainingHooks
from r1_thinking_reward import is_accuracy_correct
from sampling import sampling_config_from_dict
from tokenizer import Tokenizer


SYSTEM_MESSAGE = (
    "You are a helpful assistant. You first think about the reasoning process "
    "in your mind and then provide the user with the answer."
)
RESPONSE_PROMPT = "Let me solve this step by step.\n<think>"


def build_dataset_and_collate(config: dict[str, Any], tokenizer: Tokenizer, split: str):
    dataset_name = config["dataset"]["name"].lower()
    dcfg = config["dataset"]
    if dataset_name == "countdown":
        dataset = CountdownTasksDataset(
            tokenizer=tokenizer,
            data_path=dcfg["countdown_path"],
            split=split,
            test_size=dcfg["test_size"],
        )
        return dataset, CountdownTasksDataset.collate_fn, "countdown"
    if dataset_name == "dapo_math_17k":
        dataset = DAPOMathDataset(
            tokenizer=tokenizer,
            parquet_path=dcfg["dapo_parquet_path"],
            split=split,
            test_size=dcfg["test_size"],
        )
        return dataset, DAPOMathDataset.collate_fn, "math"
    if dataset_name == "deepscaler":
        dataset = DeepScalerDataset(
            tokenizer=tokenizer,
            json_path=dcfg["deepscaler_json_path"],
            split=split,
            test_size=dcfg["test_size"],
        )
        return dataset, DeepScalerDataset.collate_fn, "math"
    if dataset_name in ("competition_math", "math"):
        split_mode = str(dcfg.get("split_mode", "tail_holdout"))
        max_samples = dcfg.get("max_samples")
        if split == "test":
            eval_cap = config.get("training", {}).get("eval_max_samples")
            if eval_cap is not None:
                max_samples = int(eval_cap) if max_samples is None else min(
                    int(max_samples), int(eval_cap)
                )
        dataset = CompetitionMathDataset(
            tokenizer=tokenizer,
            parquet_path=dcfg["competition_math_parquet_path"],
            split=split,
            test_size=dcfg["test_size"],
            split_mode=split_mode,
            max_samples=max_samples,
        )
        return dataset, CompetitionMathDataset.collate_fn, "math"
    raise ValueError(
        "dataset.name must be one of countdown, dapo_math_17k, deepscaler, "
        "competition_math. "
        f"Got: {config['dataset']['name']}"
    )


def _countdown_user_prompt(numbers: list[int], target: int) -> str:
    return (
        f"Using the numbers {numbers}, create an equation that equals {target}. "
        "You can use basic arithmetic operations (+, -, *, /), and each number can only be used once."
    )


def _extract_questions_from_batch(batch: Any, dataset_kind: str) -> list[str]:
    if dataset_kind == "countdown":
        return [
            _countdown_user_prompt(numbers, target)
            for numbers, target in zip(batch.numbers, batch.target)
        ]
    if dataset_kind == "math":
        if hasattr(batch, "problem"):
            return [str(x) for x in batch.problem]
        if hasattr(batch, "prompt"):
            return [str(x) for x in batch.prompt]
    raise ValueError(f"Unsupported dataset_kind/batch combination: {dataset_kind}")


def rewrite_batch_with_r1_prefix(batch: Any, tokenizer: Tokenizer, dataset_kind: str):
    """R1 风格 prefix：system + user + 预填 thinking 开头。"""
    questions = _extract_questions_from_batch(batch, dataset_kind)
    prefixes: list[str] = []
    prefix_tokens: list[list[str]] = []
    prefix_token_ids: list[list[int]] = []
    for q in questions:
        prefix = tokenizer.encode_chat_with_response_prompt(
            [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": q},
            ],
            RESPONSE_PROMPT,
        )
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


def _default_r1_hooks() -> ExGRPOTrainingHooks:
    return ExGRPOTrainingHooks(
        rewrite_batch_prefix=rewrite_batch_with_r1_prefix,
        build_reward_function=build_exgrpo_reward_function,
        resolve_rollout_stop=lambda tokenizer: (
            tokenizer.eos_token,
            tokenizer.eos_token_id,
        ),
    )


def evaluate_accuracy(
    model: Transformer,
    tokenizer: Tokenizer,
    device: torch.device,
    dtype: torch.dtype,
    config: dict[str, Any],
    collate_fn: Callable,
    dataset_kind: str,
    training_hooks: ExGRPOTrainingHooks | None = None,
    grader_mode: str = "legacy",
) -> float:
    """评测仅用答案正确率，与训练 reward 解耦。"""
    hooks = training_hooks or _default_r1_hooks()
    end_token, end_token_id = hooks.resolve_rollout_stop(tokenizer)
    stop_token_ids = (
        hooks.resolve_rollout_stop_ids(tokenizer)
        if hooks.resolve_rollout_stop_ids is not None
        else None
    )
    test_dataset, _, _ = build_dataset_and_collate(config, tokenizer=tokenizer, split="test")
    batch_size = max(config["training"]["batch_size"] // 2, 1)
    dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        generator=torch.Generator(device=device),
        batch_size=batch_size,
        drop_last=False,
    )
    sampling = sampling_config_from_dict(config["training"].get("sampling"))
    rollout_chunk_size = config["training"].get("rollout_chunk_size")
    correct = []
    for batch in dataloader:
        batch = hooks.rewrite_batch_prefix(
            batch, tokenizer=tokenizer, dataset_kind=dataset_kind
        )
        episodes = rollout_chunked(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            max_gen_len=config["training"]["max_gen_len"],
            num_answer_per_question=1,
            reward_function=lambda **_: {"reward": 0.0, "reward_info": {}},
            device=device,
            dtype=dtype,
            sampling=sampling,
            rollout_chunk_size=rollout_chunk_size,
            end_token=end_token,
            end_token_id=end_token_id,
            stop_token_ids=stop_token_ids,
            treat_max_length_as_finished=hooks.treat_max_length_as_finished,
        )
        for i, ep in enumerate(episodes):
            q_idx = i
            response = ep.text[len(ep.prefix) :]
            if dataset_kind == "countdown":
                from r1_thinking_reward import compute_accuracy_reward

                gt_numbers = batch.numbers[q_idx]
                gt_target = batch.target[q_idx]
                acc = compute_accuracy_reward(
                    response,
                    dataset_kind="countdown",
                    numbers=gt_numbers,
                    target=gt_target,
                )
            else:
                from math_grader import is_math_response_correct, parse_grader_mode

                mode = parse_grader_mode(grader_mode)
                acc = 1.0 if is_math_response_correct(
                    response,
                    batch.ground_truth[q_idx],
                    grader_mode=mode,
                    end_token=end_token,
                ) else 0.0
            correct.append(acc)
    return float(np.mean(correct)) if correct else 0.0


def _next_train_batch_optional(dataloader_iter):
    try:
        return next(dataloader_iter), dataloader_iter, False
    except StopIteration:
        return None, dataloader_iter, True


def _slice_batch_by_indices(batch: Any, question_indices: list[int]) -> Any:
    """Keep selected questions from a collated batch."""
    if not question_indices:
        raise ValueError("question_indices must be non-empty")
    if len(question_indices) >= len(batch.prefix):
        return batch
    ordered_indices = sorted(question_indices)
    fields: dict[str, Any] = {}
    for key in dataclasses.fields(batch):
        value = getattr(batch, key.name)
        if isinstance(value, list):
            fields[key.name] = [value[i] for i in ordered_indices]
        else:
            fields[key.name] = value
    return dataclasses.replace(batch, **fields)


def _build_batch_from_dataset_indices(
    dataset,
    collate_fn: Callable,
    indices: list[int],
):
    samples = [dataset[i] for i in indices]
    return collate_fn(samples)


def _sample_indices_by_acc_gaussian(
    *,
    dataset_size: int,
    batch_size: int,
    index_to_qid: dict[int, str],
    acc_tracker: dict[str, float],
    mu: float,
    sigma: float,
    rng: np.random.Generator,
) -> list[int]:
    if dataset_size <= 0 or batch_size <= 0:
        return []
    all_indices = np.arange(dataset_size, dtype=np.int64)
    weights = np.ones(dataset_size, dtype=np.float64)
    scale = max(sigma, 1e-8)
    for i in range(dataset_size):
        q_id = index_to_qid.get(i)
        acc = float(acc_tracker.get(q_id, 0.0)) if q_id is not None else 0.0
        z = (acc - mu) / scale
        weights[i] = math.exp(-0.5 * z * z)
    total = float(weights.sum())
    probs = np.ones_like(weights) / len(weights) if total <= 0 else weights / total
    pick_n = min(batch_size, dataset_size)
    chosen = rng.choice(all_indices, size=pick_n, replace=False, p=probs).tolist()
    chosen.sort()
    return chosen


def _rollout_batch(
    batch,
    *,
    model,
    tokenizer,
    dataset_kind,
    max_gen_len,
    num_answers_per_question,
    reward_function,
    device,
    dtype,
    sampling,
    rollout_chunk_size,
    skip_unfinished_episodes,
    rewrite_prefix: bool = True,
    training_hooks: ExGRPOTrainingHooks | None = None,
):
    hooks = training_hooks or _default_r1_hooks()
    end_token, end_token_id = hooks.resolve_rollout_stop(tokenizer)
    stop_token_ids = (
        hooks.resolve_rollout_stop_ids(tokenizer)
        if hooks.resolve_rollout_stop_ids is not None
        else None
    )
    if rewrite_prefix:
        batch = hooks.rewrite_batch_prefix(
            batch, tokenizer=tokenizer, dataset_kind=dataset_kind
        )
    episodes = rollout_chunked(
        model=model,
        tokenizer=tokenizer,
        batch=batch,
        max_gen_len=max_gen_len,
        num_answer_per_question=num_answers_per_question,
        reward_function=reward_function,
        device=device,
        dtype=dtype,
        sampling=sampling,
        rollout_chunk_size=rollout_chunk_size,
        end_token=end_token,
        end_token_id=end_token_id,
        stop_token_ids=stop_token_ids,
        treat_max_length_as_finished=hooks.treat_max_length_as_finished,
    )
    if skip_unfinished_episodes:
        episodes = [ep for ep in episodes if ep.is_finished]
    return episodes


def _episode_accuracy_value(ep) -> float:
    return float(ep.reward_info.get("accuracy_reward", ep.reward_info.get("answer_reward", 0.0)))


def _success_rate_from_episodes(episodes: list) -> float | None:
    if not episodes:
        return None
    return float(np.mean([is_accuracy_correct(_episode_accuracy_value(ep)) for ep in episodes]))


def _fmt_optional_metric(value: float | None) -> str:
    return "na" if value is None else f"{value:.3f}"


def main(
    config_path: str,
    resume_lora_ckpt: str | None = None,
    resume_log_dir: str | None = None,
    model_preset: str | None = None,
    training_hooks: ExGRPOTrainingHooks | None = None,
):
    hooks = training_hooks or _default_r1_hooks()
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if model_preset is not None:
        config.setdefault("model", {})["preset"] = model_preset

    resolved_model = resolve_model_config(config["model"])
    pretrained_model_path = resolved_model.path
    validate_training_seq_limits(
        resolved_model,
        max_prompt_len=int(config["training"]["max_prompt_len"]),
        max_gen_len=int(config["training"]["max_gen_len"]),
    )
    device = torch.device(config["model"]["device"])
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(config["model"]["dtype"], torch.bfloat16)
    torch.set_default_device(device)
    torch.random.manual_seed(config["training"]["random_seed"])

    batch_size = config["training"]["batch_size"]
    num_questions_per_batch = config["training"]["num_questions_per_batch"]
    num_answers_per_question = batch_size // num_questions_per_batch

    resume_ckpt_path, resumed_log_dir = resolve_resume_paths(
        config["training"],
        cli_resume_lora_ckpt=resume_lora_ckpt,
        cli_resume_log_dir=resume_log_dir,
    )
    session_timestamp = format_run_timestamp()
    if resumed_log_dir is not None:
        run_log_dir = resumed_log_dir
        print(f"Resuming in-place into log dir: {run_log_dir}")
    else:
        run_log_dir = (
            Path(config["training"]["log_dir"]) / f"{session_timestamp}-{resolved_model.run_tag}"
        )
        if resume_ckpt_path is not None:
            print(f"Resuming from checkpoint into new log dir: {run_log_dir}")
    run_log_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot_path = save_run_config_snapshot(
        config_path,
        run_log_dir,
        session_timestamp,
        resume_from_ckpt=resume_ckpt_path,
    )
    print(f"Saved config snapshot: {config_snapshot_path}")

    tokenizer = Tokenizer(str(pretrained_model_path / "tokenizer.json"))
    train_dataset, collate_fn, dataset_kind = build_dataset_and_collate(
        config, tokenizer=tokenizer, split="train"
    )
    reward_cfg = exgrpo_reward_config_from_dict(config["training"].get("reward"))
    reward_function = hooks.build_reward_function(
        dataset_kind=dataset_kind,
        cfg=reward_cfg,
    )
    train_dataset_size = len(train_dataset)

    print(
        f"Model: {pretrained_model_path.name} "
        f"(preset={resolved_model.preset or 'custom'}, "
        f"max_pos={resolved_model.max_position_embeddings})"
    )

    model = Transformer.from_pretrained(pretrained_model_path, device=device).train()
    mode = config["training"]["mode"].lower()
    use_lora = mode == "lora"
    if mode not in {"full", "lora"}:
        raise ValueError("training.mode must be 'full' or 'lora'")

    lora_cfg = config["training"].get("lora", {})
    if use_lora:
        lora_config = LoRAConfig(
            r=lora_cfg.get("r", 8),
            alpha=lora_cfg.get("alpha", 16.0),
            dropout=lora_cfg.get("dropout", 0.0),
            target_modules=tuple(
                lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
            ),
        )
        replaced = apply_lora_to_model(model, lora_config)
        freeze_non_lora_parameters(model)
        total_params, trainable_params = count_parameters(model)
        print(
            f"Training mode=lora, replaced {len(replaced)} layers, "
            f"trainable {trainable_params}/{total_params} parameters."
        )
    else:
        total_params, trainable_params = count_parameters(model)
        print(
            f"Training mode=full, trainable {trainable_params}/{total_params} parameters."
        )

    optimizer = MemoryEfficientAdamW(
        get_trainable_parameters(model) if use_lora else model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        betas=config["training"]["betas"],
        enabled=config["training"]["memory_efficient_adamw"],
    )

    exgrpo_cfg = ExGRPOConfig.from_dict(config["training"].get("exgrpo"))
    exgrpo_mgr = ExGRPOManager(exgrpo_cfg)

    resumed_step = 0
    if resume_ckpt_path is not None:
        resumed_step = load_lora_training_checkpoint(
            resume_ckpt_path,
            model,
            optimizer,
            use_lora=use_lora,
            exgrpo_manager=exgrpo_mgr if exgrpo_cfg.enabled else None,
        )
        if exgrpo_cfg.enabled and resume_ckpt_path is not None:
            print(format_exgrpo_storage_summary(manager=exgrpo_mgr))
    tb_writer_kwargs: dict[str, Any] = {"log_dir": str(run_log_dir)}
    if resumed_log_dir is not None and resumed_step > 0:
        # Reusing one log dir across resumed sessions requires purging stale steps.
        tb_writer_kwargs["purge_step"] = resumed_step + 1
    tb_writer = SummaryWriter(**tb_writer_kwargs)

    ckpt_dir = run_log_dir
    sampling = sampling_config_from_dict(config["training"].get("sampling"))
    rollout_chunk_size = config["training"].get("rollout_chunk_size")
    clip_eps = config["training"].get("clip_eps", 0.2)
    clip_low = config["training"].get("clip_ratio_low")
    clip_high = config["training"].get("clip_ratio_high")
    use_ppo_clip = config["training"].get("use_ppo_clip", True)
    ppo_epochs = int(config["training"].get("ppo_epochs", 1))
    advantage_std_threshold = float(config["training"].get("advantage_std_threshold", 1e-6))
    center_advantages = bool(config["training"].get("center_advantages", True))
    scale_advantages_by_std = bool(config["training"].get("scale_advantages_by_std", True))
    advantage_std_epsilon = float(config["training"].get("advantage_std_epsilon", 1e-4))
    drop_zero_adv_groups = bool(config["training"].get("drop_zero_adv_groups", True))
    adv_zero_drop_cfg = adv_zero_drop_replenish_config_from_dict(
        config["training"],
        default_batch_size=batch_size,
    )
    tb_verbose_metrics = bool(config["training"].get("tb_verbose_metrics", False))
    tb_text_samples_default = 4 if tb_verbose_metrics else 0
    tb_text_samples = max(
        0, int(config["training"].get("tb_text_samples", tb_text_samples_default))
    )
    beta = float(config["training"].get("beta", 0.0))
    advantage_mode = str(config["training"].get("advantage_mode", "grpo")).lower()
    gspo_clip_len_scaling = str(config["training"].get("gspo_clip_len_scaling", "none")).lower()

    print(
        "ExGRPO reward: "
        f"w_acc={reward_cfg.w_accuracy}, w_fmt={reward_cfg.w_format}, "
        f"acc_mode={reward_cfg.accuracy_mode}, grader={reward_cfg.grader_mode}"
    )
    print(f"Policy update: advantage_mode={advantage_mode}, gspo_clip={gspo_clip_len_scaling}")
    print(f"Rollout chunk size: {rollout_chunk_size}")
    print(f"max_gen_len: {config['training']['max_gen_len']}")
    if adv_zero_drop_cfg.enabled:
        print(
            "Adv-zero-drop replenish: "
            f"target_batch_size={adv_zero_drop_cfg.target_batch_size}, "
            f"max_rounds={adv_zero_drop_cfg.max_rounds}"
        )
    if exgrpo_cfg.enabled:
        print(
            "ExGRPO enabled: "
            f"rho={exgrpo_cfg.rho}, beta={exgrpo_cfg.beta}, K={exgrpo_cfg.K}, "
            f"activation_threshold={exgrpo_cfg.activation_threshold}, "
            f"mix_acc_threshold={exgrpo_cfg.mix_acc_threshold}, "
            f"mu={exgrpo_cfg.mu}, sigma={exgrpo_cfg.sigma}"
        )
    print(
        "TensorBoard logging: "
        f"verbose_metrics={tb_verbose_metrics}, text_samples={tb_text_samples}"
    )
    print(f"Checkpoints will be saved under: {ckpt_dir}")
    start_time = time.time()
    skip_unfinished_episodes = bool(config["training"]["skip_unfinished_episodes"])
    batches_per_epoch = max(
        int(math.ceil(train_dataset_size / max(num_questions_per_batch, 1))), 1
    )
    consumed_batches = 0
    sequential_cursor = 0
    index_to_qid: dict[int, str] = {}
    switched_to_acc_sampling = False

    def has_full_acc_coverage() -> bool:
        if len(index_to_qid) < train_dataset_size:
            return False
        return all(
            q_id in exgrpo_mgr.buffer.acc_tracker for q_id in index_to_qid.values()
        )

    def fetch_next_batch() -> tuple[Any | None, list[int], bool]:
        nonlocal consumed_batches, sequential_cursor, switched_to_acc_sampling
        if consumed_batches >= batches_per_epoch:
            return None, [], True
        if has_full_acc_coverage():
            if not switched_to_acc_sampling:
                switched_to_acc_sampling = True
                print("Batch sampling switched to acc-gaussian mode.")
            indices = _sample_indices_by_acc_gaussian(
                dataset_size=train_dataset_size,
                batch_size=num_questions_per_batch,
                index_to_qid=index_to_qid,
                acc_tracker=exgrpo_mgr.buffer.acc_tracker,
                mu=exgrpo_cfg.mu,
                sigma=exgrpo_cfg.sigma,
                rng=exgrpo_mgr.buffer.rng,
            )
        else:
            start = sequential_cursor
            end = min(start + num_questions_per_batch, train_dataset_size)
            if end <= start:
                return None, [], True
            indices = list(range(start, end))
            sequential_cursor = end
        if not indices:
            return None, [], True
        batch = _build_batch_from_dataset_indices(train_dataset, collate_fn, indices)
        consumed_batches += 1
        return batch, indices, False

    local_step = 0

    while True:
        batch, batch_indices, exhausted = fetch_next_batch()
        if exhausted:
            break
        local_step += 1
        step = resumed_step + local_step
        rollout_kwargs = dict(
            model=model,
            tokenizer=tokenizer,
            dataset_kind=dataset_kind,
            max_gen_len=config["training"]["max_gen_len"],
            reward_function=reward_function,
            device=device,
            dtype=dtype,
            sampling=sampling,
            rollout_chunk_size=rollout_chunk_size,
            skip_unfinished_episodes=skip_unfinished_episodes,
            training_hooks=hooks,
        )
        rollout_batch = hooks.rewrite_batch_prefix(
            batch, tokenizer=tokenizer, dataset_kind=dataset_kind
        )
        for idx, prefix in zip(batch_indices, rollout_batch.prefix):
            index_to_qid[idx] = question_id_from_prefix(prefix)
        exgrpo_k = exgrpo_cfg.K if exgrpo_cfg.enabled else num_answers_per_question
        current_batch_questions = len(rollout_batch.prefix)
        n_exp_target, _ = exgrpo_mgr.build_mixed_batch_plan(current_batch_questions)
        episodes: list = []
        exgrpo_stats: dict[str, float] = {
            "exgrpo_n_exp_target": float(n_exp_target),
            "exgrpo_n_exp_candidate": 0.0,
            "exgrpo_n_exp": 0.0,
            "exgrpo_n_on": float(current_batch_questions),
            "exgrpo_buffer_size": float(len(exgrpo_mgr.buffer)),
            "exgrpo_activated": float(exgrpo_mgr.activated),
        }
        if exgrpo_cfg.enabled and rollout_batch is not None:
            exgrpo_mgr.enrich_meta_from_batch(rollout_batch)

        if exgrpo_mgr.activated and rollout_batch is not None:
            # Current-batch full gating: every question is examined for replay mixing.
            exgrpo_stats["exgrpo_n_exp_candidate"] = float(len(rollout_batch.prefix))

            mix_pairs: list[tuple[int, str]] = []
            mix_qids: list[str] = []
            on_indices: list[int] = []
            for idx, prefix in enumerate(rollout_batch.prefix):
                q_id = question_id_from_prefix(prefix)
                hist_acc = float(exgrpo_mgr.buffer.acc_tracker.get(q_id, 0.0))
                has_replay = q_id in exgrpo_mgr.buffer.buffer and bool(
                    exgrpo_mgr.buffer.buffer[q_id]
                )
                if hist_acc >= exgrpo_cfg.mix_acc_threshold and has_replay:
                    mix_pairs.append((idx, q_id))
                    mix_qids.append(q_id)
                else:
                    on_indices.append(idx)
            replay_lookup = exgrpo_mgr.build_replay_for_question_ids(
                mix_qids,
                model,
                device=device,
                dtype=dtype,
                pad_token_id=tokenizer.pad_token_id,
            )
            final_mix_pairs: list[tuple[int, str]] = []
            for idx, q_id in mix_pairs:
                if q_id in replay_lookup:
                    final_mix_pairs.append((idx, q_id))
                else:
                    on_indices.append(idx)

            # 3) Non-mixed questions run pure on-policy K rollouts.
            on_indices = sorted(set(on_indices))
            exgrpo_stats["exgrpo_n_exp"] = float(len(final_mix_pairs))
            exgrpo_stats["exgrpo_n_on"] = float(len(on_indices))
            if on_indices:
                on_batch = _slice_batch_by_indices(rollout_batch, on_indices)
                on_episodes = _rollout_batch(
                    on_batch,
                    num_answers_per_question=exgrpo_k,
                    rewrite_prefix=False,
                    **rollout_kwargs,
                )
                episodes.extend(on_episodes)

            # 4) Mixed questions run K-1 fresh and merge 1 replay.
            if final_mix_pairs:
                mix_indices = [idx for idx, _ in final_mix_pairs]
                exp_batch = _slice_batch_by_indices(rollout_batch, mix_indices)
                replay_episodes = []
                for _, q_id in final_mix_pairs:
                    _, replay_ep = replay_lookup[q_id]
                    replay_episodes.append(replay_ep)
                fresh_exp = _rollout_batch(
                    exp_batch,
                    num_answers_per_question=max(exgrpo_k - 1, 1),
                    rewrite_prefix=False,
                    **rollout_kwargs,
                )
                episodes.extend(
                    merge_replay_and_fresh_rollouts(replay_episodes, fresh_exp)
                )
        else:
            episodes = _rollout_batch(
                rollout_batch,
                num_answers_per_question=exgrpo_k,
                rewrite_prefix=False,
                **rollout_kwargs,
            )

        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        pass_at_1 = compute_pass_at_1(episodes)
        if exgrpo_cfg.enabled and not exgrpo_mgr.activated:
            activated_now = exgrpo_mgr.should_activate_exgrpo(pass_at_1)
            if activated_now:
                print(
                    f"Step {step}: ExGRPO activated (pass@1={pass_at_1:.3f} "
                    f">= {exgrpo_cfg.activation_threshold})"
                )
        exgrpo_stats["exgrpo_pass_at_1"] = pass_at_1
        exgrpo_stats["exgrpo_activated"] = float(exgrpo_mgr.activated)

        if exgrpo_cfg.enabled:
            collect_stats = exgrpo_mgr.collect_from_rollouts(episodes)
            exgrpo_stats.update({f"exgrpo_{k}": v for k, v in collect_stats.items()})
            exgrpo_stats["exgrpo_buffer_size"] = float(len(exgrpo_mgr.buffer))
            exgrpo_stats["exgrpo_buffer_traj"] = float(exgrpo_mgr.buffer.num_trajectories)
            exgrpo_stats["exgrpo_retired"] = float(len(exgrpo_mgr.buffer.retired_set))

        replenish_stats: dict[str, float] = {}
        if adv_zero_drop_cfg.enabled:

            def rollout_more():
                extra_batch, extra_indices, extra_exhausted = fetch_next_batch()
                if extra_exhausted or extra_batch is None:
                    return []
                extra_rollout_batch = hooks.rewrite_batch_prefix(
                    extra_batch, tokenizer=tokenizer, dataset_kind=dataset_kind
                )
                for idx, prefix in zip(extra_indices, extra_rollout_batch.prefix):
                    index_to_qid[idx] = question_id_from_prefix(prefix)
                return _rollout_batch(
                    extra_rollout_batch,
                    rewrite_prefix=False,
                    **rollout_kwargs,
                )

            episodes, replenish_stats = collect_episodes_with_adv_zero_drop_replenish(
                episodes,
                replenish_cfg=adv_zero_drop_cfg,
                rollout_more=rollout_more,
            )
            if replenish_stats.get("adv_zero_drop_replenish_rounds", 0) > 0:
                print(
                    f"Step {step}: adv-zero-drop replenish "
                    f"{int(replenish_stats['adv_zero_drop_initial_usable'])} -> "
                    f"{int(replenish_stats['adv_zero_drop_final_usable'])} "
                    f"(target={adv_zero_drop_cfg.target_batch_size}, "
                    f"rounds={int(replenish_stats['adv_zero_drop_replenish_rounds'])})"
                )
            if device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        if not episodes:
            n_total = len(rollout_batch.prefix) * exgrpo_k if rollout_batch else 0
            print(
                f"Step {step}: all episodes filtered out, skip update. "
                f"(skip_unfinished={skip_unfinished_episodes}, "
                f"treat_max_len_finished={hooks.treat_max_length_as_finished})"
            )
            continue

        use_exgrpo_loss = exgrpo_mgr.activated and any(
            ep.is_exp_group for ep in episodes
        )
        step_scale_by_std = (
            scale_advantages_by_std if not use_exgrpo_loss else False
        )

        results = update_policy(
            model=model,
            optimizer=optimizer,
            episodes=episodes,
            micro_batch_size=config["training"]["micro_batch_size"],
            pad_token_id=tokenizer.pad_token_id,
            max_grad_norm=config["training"]["max_grad_norm"],
            device=device,
            dtype=dtype,
            clip_eps=clip_eps,
            clip_ratio_low=clip_low,
            clip_ratio_high=clip_high,
            use_ppo_clip=use_ppo_clip,
            ppo_epochs=ppo_epochs,
            advantage_std_threshold=advantage_std_threshold,
            center_advantages=center_advantages,
            scale_advantages_by_std=step_scale_by_std,
            advantage_std_epsilon=advantage_std_epsilon,
            drop_zero_adv_groups=drop_zero_adv_groups,
            beta=beta,
            advantage_mode=advantage_mode,
            gspo_clip_len_scaling=gspo_clip_len_scaling,
            use_exgrpo_loss=use_exgrpo_loss,
            exgrpo_shaping_beta=exgrpo_cfg.beta,
            exgrpo_rho=exgrpo_cfg.rho,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        torch.cuda.synchronize()
        end_time = time.time()
        duration = end_time - start_time
        start_time = end_time

        reward = [ep.reward for ep in episodes]
        on_policy_episodes = [ep for ep in episodes if not ep.is_exp_group]
        exp_group_episodes = [ep for ep in episodes if ep.is_exp_group]
        replay_episodes = [ep for ep in episodes if ep.is_replay]
        fresh_exp_episodes = [ep for ep in episodes if ep.is_exp_group and not ep.is_replay]
        format_r = [ep.reward_info.get("format_reward", 0.0) for ep in episodes]
        num_finished = sum(ep.is_finished for ep in episodes)
        mean_reward = float(np.mean(reward))
        std_reward = float(np.std(reward))
        success_rate = _success_rate_from_episodes(episodes) or 0.0
        success_rate_on_policy = _success_rate_from_episodes(on_policy_episodes)
        success_rate_exp_group = _success_rate_from_episodes(exp_group_episodes)
        success_rate_replay = _success_rate_from_episodes(replay_episodes)
        success_rate_fresh_exp = _success_rate_from_episodes(fresh_exp_episodes)
        mean_format = float(np.mean(format_r))
        grad_norm = results["grad_norm"]
        entropy = results["entropy"]
        clip_fraction = results["clip_fraction"]
        approx_kl = results["approx_kl"]
        kl_loss = results.get("kl_loss", 0.0)
        ppo_loss = results.get("ppo_loss", 0.0)
        ratio_mean = results["ratio_mean"]
        num_responses = results.get("num_responses", 0.0)
        nonzero_adv_frac = results.get("nonzero_adv_frac", 1.0)
        group_reward_std_mean = results.get("group_reward_std_mean", 0.0)
        kept_group_reward_std_mean = results.get("kept_group_reward_std_mean", 0.0)
        group_below_threshold_frac = results.get("group_below_threshold_frac", 0.0)
        advantage_std = results.get("advantage_std", 0.0)
        replenish_rounds = replenish_stats.get("adv_zero_drop_replenish_rounds", 0.0)
        adv_drop_final_usable = replenish_stats.get(
            "adv_zero_drop_final_usable", float(len(episodes))
        )
        adv_drop_reached_target = replenish_stats.get(
            "adv_zero_drop_reached_target", 1.0 if adv_zero_drop_cfg.enabled else 0.0
        )
        ppo_epochs_ran = results.get("ppo_epochs_ran", 0)
        num_target_tokens = results.get("num_target_tokens", 0.0)
        loss = results["loss"]
        mean_resp_len = float(np.mean([len(ep.generated_token_ids) for ep in episodes]))

        num_replay = len(replay_episodes)
        exp_group_frac = float(len(exp_group_episodes) / max(len(episodes), 1))
        replay_frac = float(num_replay / max(len(episodes), 1))
        exgrpo_stats["exgrpo_num_replay"] = float(num_replay)
        exgrpo_stats["exgrpo_exp_group_frac"] = exp_group_frac
        exgrpo_stats["exgrpo_replay_frac"] = replay_frac

        print(
            f"Step {step}, mean_reward: {mean_reward:.3f}, "
            "accuracy(all/on/exp/replay/fresh): "
            f"{success_rate:.3f}/"
            f"{_fmt_optional_metric(success_rate_on_policy)}/"
            f"{_fmt_optional_metric(success_rate_exp_group)}/"
            f"{_fmt_optional_metric(success_rate_replay)}/"
            f"{_fmt_optional_metric(success_rate_fresh_exp)}, "
            f"format: {mean_format:.3f}, "
            f"grad_norm: {grad_norm:.2f}, duration: {duration:.2f}, "
            f"mean_response_len: {mean_resp_len:.0f}, entropy: {entropy:.2f}, "
            f"clip_frac: {clip_fraction:.3f}, approx_kl: {approx_kl:.4f}, "
            f"nonzero_adv: {nonzero_adv_frac:.2f}, "
            f"group_std: {group_reward_std_mean:.4f}, "
            f"kept_group_std: {kept_group_reward_std_mean:.4f}, "
            f"adv_std: {advantage_std:.3f}, "
            f"usable_eps: {int(adv_drop_final_usable)}, "
            f"repl_rounds: {int(replenish_rounds)}, epochs: {ppo_epochs_ran}"
            + (
                f", exgrpo: on={int(exgrpo_stats['exgrpo_n_on'])}/"
                f"exp={int(exgrpo_stats['exgrpo_n_exp'])}, "
                f"buf={int(exgrpo_stats['exgrpo_buffer_size'])}, "
                f"pass@1={exgrpo_stats['exgrpo_pass_at_1']:.3f}"
                if exgrpo_cfg.enabled
                else ""
            )
        )

        if step % config["training"]["eval_interval"] == 0:
            eval_sr = evaluate_accuracy(
                model=model,
                tokenizer=tokenizer,
                device=device,
                dtype=dtype,
                config=config,
                collate_fn=collate_fn,
                dataset_kind=dataset_kind,
                training_hooks=hooks,
                grader_mode=reward_cfg.grader_mode,
            )
            print(f"Eval accuracy: {eval_sr:.3f}")
            tb_writer.add_scalar("success_rate/eval", eval_sr, step)

        core_scalar_metrics = {
            "loss": loss,
            "mean_reward": mean_reward,
            "success_rate/train": success_rate,
            "grad_norm": grad_norm,
            "duration": duration,
            "mean_response_len": mean_resp_len,
            "entropy": entropy,
            "clip_fraction": clip_fraction,
            "approx_kl": approx_kl,
            "ppo_loss": ppo_loss,
            "kl_loss": kl_loss,
            "ratio_mean": ratio_mean,
            "num_target_tokens": num_target_tokens,
        }
        for key, value in core_scalar_metrics.items():
            tb_writer.add_scalar(key, value, step)

        if success_rate_on_policy is not None:
            tb_writer.add_scalar("success_rate/train_on_policy", success_rate_on_policy, step)
        if success_rate_exp_group is not None:
            tb_writer.add_scalar("success_rate/train_exp_group", success_rate_exp_group, step)
        if success_rate_replay is not None:
            tb_writer.add_scalar("success_rate/train_exp_replay", success_rate_replay, step)
        if success_rate_fresh_exp is not None:
            tb_writer.add_scalar("success_rate/train_exp_fresh", success_rate_fresh_exp, step)

        if tb_verbose_metrics:
            verbose_scalar_metrics = {
                "std_reward": std_reward,
                "format_reward": mean_format,
                "num_finished_episodes": float(num_finished),
                "num_responses": num_responses,
                "nonzero_adv_frac": nonzero_adv_frac,
                "group_reward_std_mean": group_reward_std_mean,
                "kept_group_reward_std_mean": kept_group_reward_std_mean,
                "group_below_threshold_frac": group_below_threshold_frac,
                "advantage_std": advantage_std,
                "adv_zero_drop_replenish_rounds": replenish_rounds,
                "adv_zero_drop_final_usable": adv_drop_final_usable,
                "adv_zero_drop_reached_target": adv_drop_reached_target,
                "ppo_epochs_ran": float(ppo_epochs_ran),
            }
            for key, value in verbose_scalar_metrics.items():
                tb_writer.add_scalar(key, value, step)

        if exgrpo_cfg.enabled:
            core_exgrpo_keys = {
                "exgrpo_n_exp_target",
                "exgrpo_n_exp_candidate",
                "exgrpo_n_exp",
                "exgrpo_n_on",
                "exgrpo_buffer_size",
                "exgrpo_activated",
                "exgrpo_pass_at_1",
                "exgrpo_num_replay",
                "exgrpo_exp_group_frac",
                "exgrpo_replay_frac",
            }
            for key, value in exgrpo_stats.items():
                if tb_verbose_metrics or key in core_exgrpo_keys:
                    tb_writer.add_scalar(key.replace("exgrpo_", "exgrpo/"), value, step)

        if tb_text_samples > 0:
            for i, ep in enumerate(episodes[:tb_text_samples]):
                text = html.escape(ep.text)
                tb_writer.add_text(f"text_{i}", f"<pre>{text}</pre>", step)

        if step % config["training"]["ckpt_save_interval"] == 0:
            output_file = save_training_checkpoint(
                ckpt_dir,
                step,
                use_lora=use_lora,
                model=model,
                optimizer=optimizer,
                base_model_path=pretrained_model_path if use_lora else None,
                lora_config=lora_cfg if use_lora else None,
                exgrpo_manager=exgrpo_mgr if exgrpo_cfg.enabled else None,
            )
            exgrpo_msg = (
                f", {format_exgrpo_storage_summary(manager=exgrpo_mgr)}"
                if exgrpo_cfg.enabled
                else ""
            )
            print(f"Saved checkpoint to {output_file}{exgrpo_msg}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config_exgrpo.yaml")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=sorted(MODEL_PRESETS),
        help="Override model.preset from config (e.g. qwen2.5-math-7b-instruct)",
    )
    add_resume_arguments(parser)
    args = parser.parse_args()
    main(args.config, args.resume_lora_ckpt, args.resume_log_dir, args.model)
