import html
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from checkpoint import (
    add_resume_arguments,
    format_run_timestamp,
    load_lora_training_checkpoint,
    resolve_resume_paths,
    save_run_config_snapshot,
    save_training_checkpoint,
)
from competition_math_task import CompetitionMathDataset, reward_function as competition_math_reward
from countdown_task import CountdownTasksDataset, reward_function as countdown_reward
from dapo_math_task import DAPOMathDataset, reward_function as dapo_reward
from deepscaler_task import DeepScalerDataset, reward_function as deepscaler_reward
from grpo import rollout, update_policy
from sampling import sampling_config_from_dict
from lora import (
    LoRAConfig,
    apply_lora_to_model,
    count_parameters,
    freeze_non_lora_parameters,
    get_trainable_parameters,
)
from optimizer import MemoryEfficientAdamW
from qwen2_model import Transformer
from tokenizer import Tokenizer


def build_dataset_and_reward(config: dict[str, Any], tokenizer: Tokenizer, split: str):
    dataset_name = config["dataset"]["name"].lower()
    dcfg = config["dataset"]
    if dataset_name == "countdown":
        dataset = CountdownTasksDataset(
            tokenizer=tokenizer,
            data_path=dcfg["countdown_path"],
            split=split,
            test_size=dcfg["test_size"],
        )
        return dataset, CountdownTasksDataset.collate_fn, countdown_reward
    if dataset_name == "dapo_math_17k":
        dataset = DAPOMathDataset(
            tokenizer=tokenizer,
            parquet_path=dcfg["dapo_parquet_path"],
            split=split,
            test_size=dcfg["test_size"],
        )
        return dataset, DAPOMathDataset.collate_fn, dapo_reward
    if dataset_name == "deepscaler":
        dataset = DeepScalerDataset(
            tokenizer=tokenizer,
            json_path=dcfg["deepscaler_json_path"],
            split=split,
            test_size=dcfg["test_size"],
        )
        return dataset, DeepScalerDataset.collate_fn, deepscaler_reward
    if dataset_name in ("competition_math", "math"):
        dataset = CompetitionMathDataset(
            tokenizer=tokenizer,
            parquet_path=dcfg["competition_math_parquet_path"],
            split=split,
            test_size=dcfg["test_size"],
        )
        return dataset, CompetitionMathDataset.collate_fn, competition_math_reward
    raise ValueError(
        "dataset.name must be one of countdown, dapo_math_17k, deepscaler, "
        "competition_math. "
        f"Got: {config['dataset']['name']}"
    )


def evaluate(
    model: Transformer,
    tokenizer: Tokenizer,
    device: torch.device,
    dtype: torch.dtype,
    config: dict[str, Any],
    reward_function: Callable,
    collate_fn: Callable,
) -> float:
    test_dataset, _, _ = build_dataset_and_reward(config, tokenizer=tokenizer, split="test")
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
    session_timestamp = format_run_timestamp()
    if resumed_log_dir is not None:
        run_log_dir = resumed_log_dir
        print(f"Resuming in-place into log dir: {run_log_dir}")
    else:
        run_log_dir = Path(config["training"]["log_dir"]) / session_timestamp
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
    train_dataset, collate_fn, reward_function = build_dataset_and_reward(
        config, tokenizer=tokenizer, split="train"
    )
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
    tb_writer_kwargs: dict[str, Any] = {"log_dir": str(run_log_dir)}
    if resumed_log_dir is not None and resumed_step > 0:
        # Reusing one log dir across resumed sessions requires purging stale steps.
        tb_writer_kwargs["purge_step"] = resumed_step + 1
    tb_writer = SummaryWriter(**tb_writer_kwargs)

    # Each run keeps its own checkpoints under its log directory so that
    # parallel/sequential runs never overwrite each other's ckpt_<step>.pt.
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
        config["training"].get("gspo_clip_len_scaling", "none")
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
    print(f"Checkpoints will be saved under: {ckpt_dir}")
    start_time = time.time()

    for local_step, batch in enumerate(train_dataloader, start=1):
        step = resumed_step + local_step
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
        formatted_reward = [ep.reward_info["format_reward"] for ep in episodes]
        answer_reward = [ep.reward_info["answer_reward"] for ep in episodes]
        num_finished = sum(ep.is_finished for ep in episodes)
        mean_reward = float(np.mean(reward))
        std_reward = float(np.std(reward))
        success_rate = float(np.mean(answer_reward))
        format_reward_mean = float(np.mean(formatted_reward))
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
        group_reward_std_mean = results.get("group_reward_std_mean", 0.0)
        kept_group_reward_std_mean = results.get("kept_group_reward_std_mean", 0.0)
        group_below_threshold_frac = results.get("group_below_threshold_frac", 0.0)
        advantage_std = results.get("advantage_std", 0.0)
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
            f"group_std: {group_reward_std_mean:.4f}, adv_std: {advantage_std:.3f}, "
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
            )
            print(f"Eval success rate: {eval_sr:.2f}")
            tb_writer.add_scalar("success_rate/eval", eval_sr, step)

        tb_writer.add_scalar("loss", loss, step)
        tb_writer.add_scalar("mean_reward", mean_reward, step)
        tb_writer.add_scalar("std_reward", std_reward, step)
        tb_writer.add_scalar("success_rate/train", success_rate, step)
        tb_writer.add_scalar("format_reward", format_reward_mean, step)
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
        tb_writer.add_scalar("group_reward_std_mean", group_reward_std_mean, step)
        tb_writer.add_scalar("kept_group_reward_std_mean", kept_group_reward_std_mean, step)
        tb_writer.add_scalar("group_below_threshold_frac", group_below_threshold_frac, step)
        tb_writer.add_scalar("advantage_std", advantage_std, step)
        tb_writer.add_scalar("num_zero_adv_groups", num_zero_adv_groups, step)
        tb_writer.add_scalar("ppo_epochs_ran", ppo_epochs_ran, step)
        for i, ep in enumerate(episodes[:4]):
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
            )
            print(f"Saved checkpoint to {output_file}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config_unified.yaml")
    add_resume_arguments(parser)
    args = parser.parse_args()
    main(args.config, args.resume_lora_ckpt, args.resume_log_dir)
