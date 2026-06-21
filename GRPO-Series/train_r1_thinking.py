"""
独立训练入口：DeepSeek-R1 风格 thinking 奖励 + 显存优化 rollout。

与 train_unified_readme.py 的区别：
  1. 奖励 = accuracy + format + 温和 length bonus（mean_reward ≠ success_rate）
  2. prefix 预填 <think>，引导模型进入推理模式
  3. 分块 rollout 降低 KV cache 峰值
  4. 日志单独记录 accuracy / format / length / thinking_chars
"""

import dataclasses
import html
import shutil
import time
from argparse import ArgumentParser
from datetime import datetime
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
    build_lora_checkpoint_payload,
    load_lora_training_checkpoint,
    resolve_resume_paths,
)
from countdown_task import CountdownTasksDataset
from dapo_math_task import DAPOMathDataset
from deepscaler_task import DeepScalerDataset
from grpo import update_policy
from grpo_efficient import rollout_chunked
from lora import (
    LoRAConfig,
    apply_lora_to_model,
    count_parameters,
    freeze_non_lora_parameters,
    get_trainable_parameters,
)
from optimizer import MemoryEfficientAdamW
from qwen2_model import Transformer
from r1_thinking_reward import build_r1_reward_function, r1_reward_config_from_dict
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
    raise ValueError(
        "dataset.name must be one of countdown, dapo_math_17k, deepscaler. "
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


def evaluate_accuracy(
    model: Transformer,
    tokenizer: Tokenizer,
    device: torch.device,
    dtype: torch.dtype,
    config: dict[str, Any],
    collate_fn: Callable,
    dataset_kind: str,
) -> float:
    """评测仅用答案正确率，与训练 reward 解耦。"""
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
        batch = rewrite_batch_with_r1_prefix(batch, tokenizer=tokenizer, dataset_kind=dataset_kind)
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
                pred = extract_pred_answer(response, dataset_name="math500")
                acc = 1.0 if answer_matches(pred, batch.ground_truth[q_idx], dataset_name="math500") else 0.0
            correct.append(acc)
    return float(np.mean(correct)) if correct else 0.0


def main(
    config_path: str,
    resume_lora_ckpt: str | None = None,
    resume_log_dir: str | None = None,
):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    pretrained_model_path = Path(config["model"]["pretrained_model_path"])
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
    if resumed_log_dir is not None:
        run_log_dir = resumed_log_dir
        print(f"Resuming into log dir: {run_log_dir}")
    else:
        current_time = datetime.now().strftime(r"%Y%m%d-%H%M%S")
        run_log_dir = Path(config["training"]["log_dir"]) / f"{current_time}-r1-thinking"
    run_log_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, run_log_dir / Path(config_path).name)
    tb_writer = SummaryWriter(log_dir=str(run_log_dir))

    tokenizer = Tokenizer(str(pretrained_model_path / "tokenizer.json"))
    train_dataset, collate_fn, dataset_kind = build_dataset_and_collate(
        config, tokenizer=tokenizer, split="train"
    )
    r1_cfg = r1_reward_config_from_dict(config["training"].get("r1_reward"))
    reward_function = build_r1_reward_function(dataset_kind=dataset_kind, cfg=r1_cfg)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        generator=torch.Generator(device=device),
        batch_size=num_questions_per_batch,
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

    resumed_step = 0
    if resume_ckpt_path is not None:
        resumed_step = load_lora_training_checkpoint(
            resume_ckpt_path,
            model,
            optimizer,
            use_lora=use_lora,
        )

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
    beta = float(config["training"].get("beta", 0.0))
    advantage_mode = str(config["training"].get("advantage_mode", "grpo")).lower()
    gspo_clip_len_scaling = str(config["training"].get("gspo_clip_len_scaling", "linear")).lower()

    print(
        "R1 thinking reward: "
        f"w_acc={r1_cfg.w_accuracy}, w_fmt={r1_cfg.w_format}, w_len={r1_cfg.w_length}, "
        f"target_chars={r1_cfg.target_thinking_chars}"
    )
    print(f"Rollout chunk size: {rollout_chunk_size}")
    print(f"max_gen_len: {config['training']['max_gen_len']}")
    print(f"Checkpoints will be saved under: {ckpt_dir}")
    start_time = time.time()

    for local_step, batch in enumerate(train_dataloader, start=1):
        step = resumed_step + local_step
        batch = rewrite_batch_with_r1_prefix(batch, tokenizer=tokenizer, dataset_kind=dataset_kind)
        episodes = rollout_chunked(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            max_gen_len=config["training"]["max_gen_len"],
            num_answer_per_question=num_answers_per_question,
            reward_function=reward_function,
            device=device,
            dtype=dtype,
            sampling=sampling,
            rollout_chunk_size=rollout_chunk_size,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        if config["training"]["skip_unfinished_episodes"]:
            episodes = [ep for ep in episodes if ep.is_finished]
        if not episodes:
            print(f"Step {step}: all episodes filtered out, skip update.")
            continue

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
            scale_advantages_by_std=scale_advantages_by_std,
            advantage_std_epsilon=advantage_std_epsilon,
            drop_zero_adv_groups=drop_zero_adv_groups,
            beta=beta,
            advantage_mode=advantage_mode,
            gspo_clip_len_scaling=gspo_clip_len_scaling,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        torch.cuda.synchronize()
        end_time = time.time()
        duration = end_time - start_time
        start_time = end_time

        reward = [ep.reward for ep in episodes]
        accuracy = [ep.reward_info.get("accuracy_reward", ep.reward_info.get("answer_reward", 0.0)) for ep in episodes]
        format_r = [ep.reward_info.get("format_reward", 0.0) for ep in episodes]
        length_b = [ep.reward_info.get("length_bonus", 0.0) for ep in episodes]
        thinking_chars = [ep.reward_info.get("thinking_chars", 0.0) for ep in episodes]
        num_finished = sum(ep.is_finished for ep in episodes)
        mean_reward = float(np.mean(reward))
        std_reward = float(np.std(reward))
        success_rate = float(np.mean(accuracy))
        mean_format = float(np.mean(format_r))
        mean_length_bonus = float(np.mean(length_b))
        mean_thinking_chars = float(np.mean(thinking_chars))
        grad_norm = results["grad_norm"]
        entropy = results["entropy"]
        clip_fraction = results["clip_fraction"]
        approx_kl = results["approx_kl"]
        kl_loss = results.get("kl_loss", 0.0)
        ppo_loss = results.get("ppo_loss", 0.0)
        ratio_mean = results["ratio_mean"]
        num_responses = results.get("num_responses", 0.0)
        nonzero_adv_frac = results.get("nonzero_adv_frac", 1.0)
        ppo_epochs_ran = results.get("ppo_epochs_ran", 0)
        num_target_tokens = results.get("num_target_tokens", 0.0)
        loss = results["loss"]
        mean_resp_len = float(np.mean([len(ep.generated_token_ids) for ep in episodes]))

        print(
            f"Step {step}, mean_reward: {mean_reward:.3f}, "
            f"accuracy: {success_rate:.3f}, format: {mean_format:.3f}, "
            f"len_bonus: {mean_length_bonus:.3f}, think_chars: {mean_thinking_chars:.0f}, "
            f"grad_norm: {grad_norm:.2f}, duration: {duration:.2f}, "
            f"mean_response_len: {mean_resp_len:.0f}, entropy: {entropy:.2f}, "
            f"clip_frac: {clip_fraction:.3f}, approx_kl: {approx_kl:.4f}, "
            f"nonzero_adv: {nonzero_adv_frac:.2f}, epochs: {ppo_epochs_ran}"
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
            )
            print(f"Eval accuracy: {eval_sr:.3f}")
            tb_writer.add_scalar("success_rate/eval", eval_sr, step)

        tb_writer.add_scalar("loss", loss, step)
        tb_writer.add_scalar("mean_reward", mean_reward, step)
        tb_writer.add_scalar("std_reward", std_reward, step)
        tb_writer.add_scalar("success_rate/train", success_rate, step)
        tb_writer.add_scalar("format_reward", mean_format, step)
        tb_writer.add_scalar("length_bonus", mean_length_bonus, step)
        tb_writer.add_scalar("thinking_chars", mean_thinking_chars, step)
        tb_writer.add_scalar("grad_norm", grad_norm, step)
        tb_writer.add_scalar("duration", duration, step)
        tb_writer.add_scalar("num_finished_episodes", num_finished, step)
        tb_writer.add_scalar("mean_response_len", mean_resp_len, step)
        tb_writer.add_scalar("entropy", entropy, step)
        tb_writer.add_scalar("clip_fraction", clip_fraction, step)
        tb_writer.add_scalar("approx_kl", approx_kl, step)
        tb_writer.add_scalar("ppo_loss", ppo_loss, step)
        tb_writer.add_scalar("kl_loss", kl_loss, step)
        tb_writer.add_scalar("ratio_mean", ratio_mean, step)
        tb_writer.add_scalar("num_target_tokens", num_target_tokens, step)
        tb_writer.add_scalar("num_responses", num_responses, step)
        tb_writer.add_scalar("nonzero_adv_frac", nonzero_adv_frac, step)
        tb_writer.add_scalar("ppo_epochs_ran", ppo_epochs_ran, step)
        for i, ep in enumerate(episodes[:4]):
            text = html.escape(ep.text)
            tb_writer.add_text(f"text_{i}", f"<pre>{text}</pre>", step)

        if step % config["training"]["ckpt_save_interval"] == 0:
            output_file = ckpt_dir / f"ckpt_{step:06d}.pt"
            if use_lora:
                checkpoint = build_lora_checkpoint_payload(
                    step,
                    model,
                    optimizer,
                    base_model_path=pretrained_model_path,
                    lora_config=lora_cfg,
                )
                torch.save(checkpoint, output_file)
            else:
                torch.save(model.state_dict(), output_file)
            print(f"Saved checkpoint to {output_file}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config_r1_thinking.yaml")
    add_resume_arguments(parser)
    args = parser.parse_args()
    main(args.config, args.resume_lora_ckpt, args.resume_log_dir)
