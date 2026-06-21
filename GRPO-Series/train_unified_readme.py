import dataclasses
"""
消融实验：Qwen README 风格 prompt + 纯准确率奖励。

与 train_unified.py 对比 format+answer 复合奖励的效果。
用法: uv run python train_unified_readme.py --config configs/train_unified.yaml
"""

import html
import re
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
from grpo import rollout, update_policy
from lora import (
    LoRAConfig,
    apply_lora_to_model,
    count_parameters,
    freeze_non_lora_parameters,
    get_trainable_parameters,
)
from optimizer import MemoryEfficientAdamW
from qwen2_model import Transformer
from sampling import sampling_config_from_dict
from tokenizer import Tokenizer


README_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.You first think about the reasoning processin your mind and then provide the user with the answer."


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


def rewrite_batch_with_readme_prefix(batch: Any, tokenizer: Tokenizer, dataset_kind: str):
    questions = _extract_questions_from_batch(batch, dataset_kind)
    prefixes: list[str] = []
    prefix_tokens: list[list[str]] = []
    prefix_token_ids: list[list[int]] = []
    for q in questions:
        prefix = tokenizer.encode_chat(
            [
                {"role": "system", "content": README_SYSTEM_PROMPT},
                {"role": "user", "content": q},
            ]
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


def _extract_countdown_expression(response: str) -> str | None:
    tag_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if tag_match:
        expr = tag_match.group(1).strip()
        if expr:
            return expr
    lines = [ln.strip() for ln in response.splitlines() if ln.strip()]
    if not lines:
        return None
    return lines[-1]


def _countdown_answer_correct(response: str, numbers: list[int], target: int) -> bool:
    expr = _extract_countdown_expression(response)
    if not expr:
        return False
    if not re.match(r"^[0-9+\-*/() ]+$", expr):
        return False
    used_numbers = [int(n) for n in re.findall(r"\d+", expr)]
    if sorted(used_numbers) != sorted(numbers):
        return False
    try:
        result = eval(expr, {"__builtins__": None}, {})
        return abs(float(result) - float(target)) < 1e-5
    except Exception:
        return False


def build_readme_reward_function(dataset_kind: str) -> Callable[..., dict[str, Any]]:
    def reward_function(
        response: str,
        ground_truth: str = "",
        numbers: list[int] | None = None,
        target: int | None = None,
        end_token: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        if end_token and response.endswith(end_token):
            response = response[: -len(end_token)]
        if dataset_kind == "countdown":
            is_correct = bool(
                numbers is not None
                and target is not None
                and _countdown_answer_correct(response, numbers, target)
            )
        else:
            pred = extract_pred_answer(response, dataset_name="math500")
            is_correct = answer_matches(pred, str(ground_truth), dataset_name="math500")
        answer_reward = 1.0 if is_correct else 0.0
        return {
            "reward": answer_reward,
            "reward_info": {
                "format_reward": 0.0,
                "answer_reward": answer_reward,
            },
        }

    return reward_function


def evaluate(
    model: Transformer,
    tokenizer: Tokenizer,
    device: torch.device,
    dtype: torch.dtype,
    config: dict[str, Any],
    reward_function: Callable,
    collate_fn: Callable,
    dataset_kind: str,
) -> float:
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
    success = []
    for batch in dataloader:
        batch = rewrite_batch_with_readme_prefix(batch, tokenizer=tokenizer, dataset_kind=dataset_kind)
        episodes = rollout(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            max_gen_len=config["training"]["max_gen_len"],
            num_answer_per_question=1,
            reward_function=reward_function,
            device=device,
            dtype=dtype,
            sampling=sampling,
        )
        success.extend([ep.reward_info["answer_reward"] for ep in episodes])
    return float(np.mean(success)) if success else 0.0


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
        run_log_dir = Path(config["training"]["log_dir"]) / f"{current_time}-readme-reward"
    run_log_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, run_log_dir / Path(config_path).name)
    tb_writer = SummaryWriter(log_dir=str(run_log_dir))

    tokenizer = Tokenizer(str(pretrained_model_path / "tokenizer.json"))
    train_dataset, collate_fn, dataset_kind = build_dataset_and_collate(
        config, tokenizer=tokenizer, split="train"
    )
    reward_function = build_readme_reward_function(dataset_kind=dataset_kind)
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
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sampling = sampling_config_from_dict(config["training"].get("sampling"))
    clip_eps = config["training"].get("clip_eps", 0.2)
    clip_low = config["training"].get("clip_ratio_low")
    clip_high = config["training"].get("clip_ratio_high")
    use_ppo_clip = config["training"].get("use_ppo_clip", True)
    ppo_epochs = int(config["training"].get("ppo_epochs", 1))
    advantage_std_threshold = float(
        config["training"].get("advantage_std_threshold", 1e-6)
    )
    center_advantages = bool(config["training"].get("center_advantages", True))
    scale_advantages_by_std = bool(
        config["training"].get("scale_advantages_by_std", True)
    )
    advantage_std_epsilon = float(config["training"].get("advantage_std_epsilon", 1e-4))
    drop_zero_adv_groups = bool(config["training"].get("drop_zero_adv_groups", True))
    beta = float(config["training"].get("beta", 0.0))
    advantage_mode = str(config["training"].get("advantage_mode", "grpo")).lower()
    gspo_clip_len_scaling = str(
        config["training"].get("gspo_clip_len_scaling", "linear")
    ).lower()
    print(
        "Sampling config: "
        f"do_sample={sampling.do_sample}, temperature={sampling.temperature}, "
        f"top_p={sampling.top_p}, top_k={sampling.top_k}"
    )
    print(
        "PPO clip config: "
        f"use_ppo_clip={use_ppo_clip}, clip_eps={clip_eps}, "
        f"clip_ratio_low={clip_low}, clip_ratio_high={clip_high}, "
        f"ppo_epochs={ppo_epochs}, advantage_std_threshold={advantage_std_threshold}"
    )
    print(
        "Advantage norm config: "
        f"center_advantages={center_advantages}, "
        f"scale_advantages_by_std={scale_advantages_by_std}, "
        f"advantage_std_epsilon={advantage_std_epsilon}, "
        f"drop_zero_adv_groups={drop_zero_adv_groups}"
    )
    print(
        "Advantage objective config: "
        f"advantage_mode={advantage_mode}, "
        f"gspo_clip_len_scaling={gspo_clip_len_scaling}"
    )
    print(f"KL penalty config: beta={beta}")
    print(f"Reward mode: readme_answer_only, dataset_kind={dataset_kind}")
    print(f"Checkpoints will be saved under: {ckpt_dir}")
    start_time = time.time()

    for local_step, batch in enumerate(train_dataloader, start=1):
        step = resumed_step + local_step
        batch = rewrite_batch_with_readme_prefix(batch, tokenizer=tokenizer, dataset_kind=dataset_kind)
        episodes = rollout(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            max_gen_len=config["training"]["max_gen_len"],
            num_answer_per_question=num_answers_per_question,
            reward_function=reward_function,
            device=device,
            dtype=dtype,
            sampling=sampling,
        )
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
            clip_eps=config["training"].get("clip_eps", 0.2),
            clip_ratio_low=config["training"].get("clip_ratio_low"),
            clip_ratio_high=config["training"].get("clip_ratio_high"),
            use_ppo_clip=config["training"].get("use_ppo_clip", True),
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
        torch.cuda.synchronize()
        end_time = time.time()
        duration = end_time - start_time
        start_time = end_time

        reward = [ep.reward for ep in episodes]
        answer_reward = [ep.reward_info["answer_reward"] for ep in episodes]
        num_finished = sum(ep.is_finished for ep in episodes)
        mean_reward = float(np.mean(reward))
        std_reward = float(np.std(reward))
        success_rate = float(np.mean(answer_reward))
        grad_norm = results["grad_norm"]
        entropy = results["entropy"]
        clip_fraction = results["clip_fraction"]
        approx_kl = results["approx_kl"]
        kl_loss = results.get("kl_loss", 0.0)
        ppo_loss = results.get("ppo_loss", 0.0)
        ratio_mean = results["ratio_mean"]
        num_responses = results.get("num_responses", 0.0)
        nonzero_adv_frac = results.get("nonzero_adv_frac", 1.0)
        num_zero_adv_groups = results.get("num_zero_adv_groups", 0.0)
        ppo_epochs_ran = results.get("ppo_epochs_ran", 0)
        num_target_tokens = results.get("num_target_tokens", 0.0)
        lr = optimizer.param_groups[0]["lr"]
        loss = results["loss"]
        mean_resp_len = float(np.mean([len(ep.generated_token_ids) for ep in episodes]))

        print(
            f"Step {step}, mean_reward: {mean_reward:.2f}, "
            f"train success_rate: {success_rate:.2f}, grad_norm: {grad_norm:.2f}, "
            f"duration: {duration:.2f}, num_finished: {num_finished}, "
            f"mean_response_len: {mean_resp_len:.2f}, entropy: {entropy:.2f}, "
            f"clip_frac: {clip_fraction:.3f}, approx_kl: {approx_kl:.4f}, "
            f"ppo_loss: {ppo_loss:.4f}, kl_loss: {kl_loss:.4f}, "
            f"ratio_mean: {ratio_mean:.3f}, nonzero_adv: {nonzero_adv_frac:.2f}, "
            f"epochs: {ppo_epochs_ran}, tgt_toks: {num_target_tokens:.0f}, "
            f"responses: {num_responses:.0f}"
        )

        if step % config["training"]["eval_interval"] == 0:
            eval_sr = evaluate(
                model=model,
                tokenizer=tokenizer,
                device=device,
                dtype=dtype,
                config=config,
                reward_function=reward_function,
                collate_fn=collate_fn,
                dataset_kind=dataset_kind,
            )
            print(f"Eval success rate: {eval_sr:.2f}")
            tb_writer.add_scalar("success_rate/eval", eval_sr, step)

        tb_writer.add_scalar("loss", loss, step)
        tb_writer.add_scalar("mean_reward", mean_reward, step)
        tb_writer.add_scalar("std_reward", std_reward, step)
        tb_writer.add_scalar("success_rate/train", success_rate, step)
        tb_writer.add_scalar("format_reward", 0.0, step)
        tb_writer.add_scalar("grad_norm", grad_norm, step)
        tb_writer.add_scalar("duration", duration, step)
        tb_writer.add_scalar("num_finished_episodes", num_finished, step)
        tb_writer.add_scalar("learning_rate", lr, step)
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
        tb_writer.add_scalar("num_zero_adv_groups", num_zero_adv_groups, step)
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
    parser.add_argument("--config", type=str, default="configs/config_unified.yaml")
    add_resume_arguments(parser)
    args = parser.parse_args()
    main(args.config, args.resume_lora_ckpt, args.resume_log_dir)
