# train.py

import html
import shutil
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from deepscaler_task import DeepScalerDataset, reward_function

from grpo import rollout, update_policy
from lora import (
    LoRAConfig,
    apply_lora_to_model,
    count_parameters,
    freeze_non_lora_parameters,
    get_lora_state_dict,
    get_trainable_parameters,
    load_lora_state_dict,
)
from optimizer import MemoryEfficientAdamW
from qwen2_model import Transformer
from tokenizer import Tokenizer


def evaluate(model, tokenizer, device, dtype, config):
    test_dataset = DeepScalerDataset(
        tokenizer=tokenizer,
        json_path=config["data"]["json_path"],
        split="test",
        test_size=config["data"]["test_size"],
    )
    generator = torch.Generator(device=device)
    dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=DeepScalerDataset.collate_fn,
        generator=generator,
        batch_size=config["training"]["batch_size"] // 2,
        drop_last=False,
    )
    success = []
    for batch_idx, batch in enumerate(dataloader):
        print(f"\n[Eval batch {batch_idx + 1}/{len(dataloader)}]", flush=True)
        episodes = rollout(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            max_gen_len=config["training"]["max_gen_len"],
            num_answer_per_question=1,
            reward_function=reward_function,
            device=device,
            dtype=dtype,
        )
        success.extend([episode.reward_info["answer_reward"] for episode in episodes])
    return np.mean(success)


def main(config_path: str, resume_lora_ckpt: str | None = None):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    pretrained_model_path = Path(config["model"]["pretrained_model_path"])
    device = torch.device(config["model"]["device"])
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }
    dtype = dtype_map.get(config["model"]["dtype"], torch.bfloat16)
    torch.set_default_device(device)
    torch.random.manual_seed(config["training"]["random_seed"])

    BATCH_SIZE               = config["training"]["batch_size"]
    NUM_QUESTIONS_PER_BATCH  = config["training"]["num_questions_per_batch"]
    NUM_ANSWERS_PER_QUESTION = BATCH_SIZE // NUM_QUESTIONS_PER_BATCH

    current_time = datetime.now().strftime(r"%Y%m%d-%H%M%S")
    run_log_dir  = Path(config["training"]["log_dir"]) / current_time
    run_log_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, run_log_dir / Path(config_path).name)
    tb_writer = SummaryWriter(log_dir=str(run_log_dir))

    tokenizer = Tokenizer(str(pretrained_model_path / "tokenizer.json"))

    train_dataset = DeepScalerDataset(
        tokenizer=tokenizer,
        json_path=config["data"]["json_path"],
        split="train",
        test_size=config["data"]["test_size"],
    )
    generator = torch.Generator(device=device)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=DeepScalerDataset.collate_fn,
        generator=generator,
        batch_size=NUM_QUESTIONS_PER_BATCH,
    )


    model = Transformer.from_pretrained(pretrained_model_path, device=device).train()

    lora_cfg = config["training"].get("lora", {})
    use_lora = lora_cfg.get("enabled", False)
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
            f"LoRA enabled: replaced {len(replaced)} layers, "
            f"trainable {trainable_params}/{total_params} parameters."
        )
    else:
        total_params, trainable_params = count_parameters(model)
        print(
            f"LoRA disabled: full finetuning with "
            f"{trainable_params}/{total_params} trainable parameters."
        )

    optimizer = MemoryEfficientAdamW(
        get_trainable_parameters(model),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        betas=config["training"]["betas"],
        enabled=config["training"]["memory_efficient_adamw"],
    )
    resumed_step = 0
    resume_path = resume_lora_ckpt or config["training"].get("resume_lora_ckpt")
    if resume_path:
        if not use_lora:
            raise ValueError("resume_lora_ckpt is set but training.lora.enabled is false.")
        resume_file = Path(resume_path)
        checkpoint = torch.load(resume_file, map_location="cpu")
        lora_state_dict = checkpoint.get("lora_state_dict")
        if lora_state_dict is None:
            raise KeyError(f"No 'lora_state_dict' found in checkpoint: {resume_file}")
        load_lora_state_dict(model, lora_state_dict, strict=True)
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        resumed_step = int(checkpoint.get("step", 0))
        print(f"Resumed LoRA checkpoint from {resume_file}, step={resumed_step}")

    start_time = time.time()
    ckpt_dir   = Path(config["training"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for local_step, batch in enumerate(train_dataloader, start=1):
        step = resumed_step + local_step
        episodes = rollout(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            max_gen_len=config["training"]["max_gen_len"],
            num_answer_per_question=NUM_ANSWERS_PER_QUESTION,
            reward_function=reward_function,
            device=device,
            dtype=dtype,
        )
        if config["training"]["skip_unfinished_episodes"]:
            episodes = [ep for ep in episodes if ep.is_finished]

        # 优化建议 1：保护空 episodes 边界，避免 update_policy 内除零或断言失败
        if len(episodes) == 0:
            print(f"Step {step}: all episodes filtered out, skipping policy update.")
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
        )
        torch.cuda.synchronize()
        end_time   = time.time()
        duration   = end_time - start_time
        start_time = end_time

        reward           = [ep.reward for ep in episodes]
        formatted_reward = [ep.reward_info["format_reward"] for ep in episodes]
        answer_reward    = [ep.reward_info["answer_reward"]  for ep in episodes]
        num_finished     = sum(ep.is_finished for ep in episodes)
        mean_reward      = np.mean(reward)
        std_reward       = np.std(reward)
        success_rate     = np.mean(answer_reward)
        format_reward_m  = np.mean(formatted_reward)
        grad_norm        = results["grad_norm"]
        entropy          = results["entropy"]
        lr               = optimizer.param_groups[0]["lr"]
        loss             = results["loss"]
        mean_resp_len    = np.mean([len(ep.generated_token_ids) for ep in episodes])

        print(
            f"Step {step}, mean_reward: {mean_reward:.2f}, "
            f"train success_rate: {success_rate:.2f}, "
            f"grad_norm: {grad_norm:.2f}, duration: {duration:.2f}, "
            f"num_finished: {num_finished}, "
            f"mean_response_len: {mean_resp_len:.2f}, "
            f"entropy: {entropy:.2f}"
        )

        if step % config["training"]["eval_interval"] == 0:
            eval_sr = evaluate(model, tokenizer, device, dtype, config)
            print(f"Eval success rate: {eval_sr:.2f}" + " " * 100)
            tb_writer.add_scalar("success_rate/eval", eval_sr, step)

        tb_writer.add_scalar("loss",                  loss,            step)
        tb_writer.add_scalar("mean_reward",           mean_reward,     step)
        tb_writer.add_scalar("std_reward",            std_reward,      step)
        tb_writer.add_scalar("success_rate/train",    success_rate,    step)
        tb_writer.add_scalar("format_reward",         format_reward_m, step)
        tb_writer.add_scalar("grad_norm",             grad_norm,       step)
        tb_writer.add_scalar("duration",              duration,        step)
        tb_writer.add_scalar("num_finished_episodes", num_finished,    step)
        tb_writer.add_scalar("learning_rate",         lr,              step)
        tb_writer.add_scalar("mean_response_len",     mean_resp_len,   step)
        tb_writer.add_scalar("entropy",               entropy,         step)

        # 优化建议 2：仅记录前若干条 episode 文本，避免 TensorBoard 日志膨胀
        for i, ep in enumerate(episodes[:4]):
            text = html.escape(ep.text)
            tb_writer.add_text(f"text_{i}", f"<pre>{text}</pre>", step)

        if step % config["training"]["ckpt_save_interval"] == 0:
            output_file = ckpt_dir / f"ckpt_{step:06d}.pt"
            if use_lora:
                checkpoint = {
                    "checkpoint_type": "lora_adapter",
                    "step": step,
                    "base_model_path": str(pretrained_model_path),
                    "lora_config": lora_cfg,
                    "lora_state_dict": get_lora_state_dict(model),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                torch.save(checkpoint, output_file)
            else:
                torch.save(model.state_dict(), output_file)
            print(f"Saved checkpoint to {output_file}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--resume_lora_ckpt",
        type=str,
        default=None,
        help="Path to LoRA-only checkpoint for resuming training.",
    )
    args = parser.parse_args()
    main(args.config, args.resume_lora_ckpt)
