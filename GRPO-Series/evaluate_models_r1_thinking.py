"""
R1 thinking 风格评测：与 train_r1_thinking.py / benchmark_task.py 的 prefix 对齐。

与 evaluate_models_readme.py 的区别：
  - system prompt 使用 SYSTEM_MESSAGE（含 thinking 引导）
  - 在 chat template 后追加 RESPONSE_PROMPT（预填 <think> 开头）
  - 答案解析仍走 benchmark_task.extract_pred_answer（优先 <answer> 标签）

用法：
  uv run python evaluate_models_r1_thinking.py --config configs/config_eval.yaml --types lora
"""

from __future__ import annotations

import json
import shutil
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark_task import RESPONSE_PROMPT, SYSTEM_MESSAGE
from eval_metrics import (
    build_benchmark_result,
    format_pass_at_k_summary,
    is_pass_at_k_solved,
    pass_at_k_simple,
    resolve_eval_benchmarks,
    resolve_pass_at_k_config,
)
from evaluate_models_readme import (
    EvalProgress,
    _dtype_from_config,
    _lora_signature,
    _run_generate_with_auto_batch_fallback,
    batched,
    build_eval_targets,
    extract_ground_truth,
    extract_question,
    filter_eval_targets,
    load_model_for_target,
    load_rows,
    resolve_benchmark_eval_params,
)
from lora import LoRAConfig, apply_lora_to_model, load_lora_state_dict
from r1_thinking_reward import (
    _normalize_for_format_check,
    extract_r1_pred_answer,
    extract_thinking_text,
    is_r1_benchmark_answer_correct,
)


def build_r1_thinking_prompt(tokenizer, question: str) -> str:
    """与训练时 encode_chat_with_response_prompt 等价（transformers 口径）。"""
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": question},
    ]
    chat_prefix = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return chat_prefix + RESPONSE_PROMPT


@torch.no_grad()
def evaluate_one_benchmark(
    model,
    tokenizer,
    benchmark_name: str,
    benchmark_path: str,
    batch_size: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    progress_desc: str,
    show_progress: bool = True,
    pass_at_k: list[int] | None = None,
    num_samples: int = 1,
    collect_format_stats: bool = False,
) -> dict[str, Any]:
    pass_at_k = pass_at_k or [1]
    num_samples = max(int(num_samples), 1)
    question_batch_size = max(1, batch_size // num_samples)
    display_k = max(pass_at_k)

    rows = load_rows(Path(benchmark_path))
    questions = [extract_question(r) for r in rows]
    ground_truth = [extract_ground_truth(r, benchmark_name) for r in rows]

    total = len(rows)
    sample_results: list[list[bool]] = []
    first_sample_correct: list[bool] = []
    format_ok = 0
    thinking_char_sum = 0
    progress = (
        EvalProgress(total=total, desc=progress_desc, display_k=display_k)
        if show_progress
        else None
    )
    try:
        for q_batch, gt_batch in zip(
            batched(questions, question_batch_size),
            batched(ground_truth, question_batch_size),
        ):
            num_q = len(q_batch)
            expanded_prompts: list[str] = []
            expanded_gt: list[str] = []
            for q, gt in zip(q_batch, gt_batch):
                prompt = build_r1_thinking_prompt(tokenizer, q)
                for _ in range(num_samples):
                    expanded_prompts.append(prompt)
                    expanded_gt.append(gt)

            all_responses: list[str] = []
            if progress is not None:
                progress.begin_generate(len(expanded_prompts))
            for chunk_start in range(0, len(expanded_prompts), batch_size):
                chunk_prompts = expanded_prompts[chunk_start : chunk_start + batch_size]
                all_responses.extend(
                    _run_generate_with_auto_batch_fallback(
                        model=model,
                        tokenizer=tokenizer,
                        prompts=chunk_prompts,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        benchmark_name=benchmark_name,
                    )
                )
                if progress is not None:
                    progress.update_generate(len(chunk_prompts))

            batch_samples = [[False] * num_samples for _ in range(num_q)]
            batch_first_correct = [False] * num_q
            for idx, (response, gt) in enumerate(zip(all_responses, expanded_gt)):
                qi = idx // num_samples
                si = idx % num_samples
                # 与训练一致：仅对 prefix 之后的「生成段」做格式归一化
                #（训练 rollout 传入 reward 的也是 ep.text[len(ep.prefix):]）
                full_response = _normalize_for_format_check(response, end_token=None)
                is_correct = is_r1_benchmark_answer_correct(
                    full_response,
                    gt,
                    dataset_name=benchmark_name,
                )
                batch_samples[qi][si] = is_correct
                if si == 0 and is_correct:
                    batch_first_correct[qi] = True
                if collect_format_stats:
                    thinking_text = extract_thinking_text(full_response)
                    if thinking_text:
                        thinking_char_sum += len(thinking_text)
                    if extract_r1_pred_answer(
                        full_response, dataset_name=benchmark_name
                    ) is not None:
                        format_ok += 1

            sample_results.extend(batch_samples)
            first_sample_correct.extend(batch_first_correct)
            if progress is not None:
                batch_pass_rate = sum(
                    pass_at_k_simple(s, display_k) for s in batch_samples
                )
                batch_solved = sum(
                    1 for s in batch_samples if is_pass_at_k_solved(s, display_k)
                )
                progress.update(
                    num_q,
                    pass_rate_delta=batch_pass_rate,
                    solved_delta=batch_solved,
                )
    finally:
        if progress is not None:
            progress.close()

    result = build_benchmark_result(
        sample_results=sample_results,
        num_samples=num_samples,
        pass_at_k=pass_at_k,
        first_sample_correct=first_sample_correct,
    )
    if collect_format_stats and total > 0:
        result["format_stats"] = {
            "answer_extracted_rate": format_ok / (total * num_samples),
            "mean_thinking_chars": thinking_char_sum / (total * num_samples),
        }
    return result


def evaluate_target_benchmarks(
    *,
    model,
    tokenizer,
    target_name: str,
    config: dict[str, Any],
    default_batch_size: int,
    default_max_gen_len: int,
    default_do_sample: bool,
    default_temperature: float,
    default_top_p: float,
    default_top_k: int,
    show_progress: bool,
    collect_format_stats: bool,
) -> dict[str, Any]:
    per_target: dict[str, Any] = {}
    eval_cfg = config.get("eval", {})
    for benchmark_name, benchmark_cfg in resolve_eval_benchmarks(
        config["benchmarks"], eval_cfg
    ):
        (
            benchmark_batch_size,
            benchmark_max_gen_len,
            benchmark_do_sample,
            benchmark_temperature,
            benchmark_top_p,
            benchmark_top_k,
        ) = resolve_benchmark_eval_params(
            config=config,
            benchmark_name=benchmark_name,
            default_batch_size=default_batch_size,
            default_max_gen_len=default_max_gen_len,
            default_do_sample=default_do_sample,
            default_temperature=default_temperature,
            default_top_p=default_top_p,
            default_top_k=default_top_k,
        )
        pass_at_k, num_samples = resolve_pass_at_k_config(
            config["eval"], benchmark_name=benchmark_name
        )
        print(f"Benchmark: {benchmark_name}")
        print(
            "  params: "
            f"batch_size={benchmark_batch_size}, "
            f"max_gen_len={benchmark_max_gen_len}, "
            f"num_samples={num_samples}, pass_at_k={pass_at_k}, "
            f"do_sample={benchmark_do_sample}, "
            f"temperature={benchmark_temperature}, "
            f"top_p={benchmark_top_p}, top_k={benchmark_top_k}"
        )
        print(f"  prefix: RESPONSE_PROMPT={RESPONSE_PROMPT!r}")
        if num_samples > 1 and not benchmark_do_sample:
            print(
                "  warning: num_samples>1 with do_sample=false yields identical "
                "samples; enable sampling for meaningful pass@k>1."
            )
        one = evaluate_one_benchmark(
            model=model,
            tokenizer=tokenizer,
            benchmark_name=benchmark_name,
            benchmark_path=benchmark_cfg["path"],
            batch_size=benchmark_batch_size,
            max_new_tokens=benchmark_max_gen_len,
            do_sample=benchmark_do_sample,
            temperature=benchmark_temperature,
            top_p=benchmark_top_p,
            top_k=benchmark_top_k,
            progress_desc=f"r1-thinking/{target_name}/{benchmark_name}",
            show_progress=show_progress,
            pass_at_k=pass_at_k,
            num_samples=num_samples,
            collect_format_stats=collect_format_stats,
        )
        print(
            f"  accuracy={one['accuracy']:.4f} ({one['correct']}/{one['total']}), "
            f"{format_pass_at_k_summary(one['pass_at_k'], num_samples=num_samples)}"
        )
        if "format_stats" in one:
            stats = one["format_stats"]
            print(
                "  format_stats: "
                f"answer_extracted_rate={stats['answer_extracted_rate']:.4f}, "
                f"mean_thinking_chars={stats['mean_thinking_chars']:.0f}"
            )
        per_target[benchmark_name] = one
    return per_target


def evaluate_lora_targets_r1(
    *,
    lora_targets: list[dict[str, Any]],
    base_model_path: Path,
    dtype: torch.dtype | str,
    tokenizer,
    config: dict[str, Any],
    default_batch_size: int,
    default_max_gen_len: int,
    default_do_sample: bool,
    default_temperature: float,
    default_top_p: float,
    default_top_k: int,
    show_progress: bool,
    collect_format_stats: bool,
) -> dict[str, Any]:
    """LoRA 评测：复用 readme 版的 backbone 复用逻辑，替换 benchmark 循环。"""
    results: dict[str, Any] = {}
    current_model = None
    current_sig: tuple[Any, ...] | None = None

    for target in lora_targets:
        ckpt_path = Path(target["ckpt_path"])
        if not ckpt_path.exists():
            raise FileNotFoundError(f"LoRA checkpoint not found: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise ValueError(f"LoRA checkpoint must be a dict: {ckpt_path}")

        lora_cfg = checkpoint.get("lora_config", target.get("lora_config", {}))
        sig = _lora_signature(lora_cfg)
        if current_model is None or sig != current_sig:
            if current_model is not None:
                del current_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            current_model = AutoModelForCausalLM.from_pretrained(
                str(base_model_path),
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True,
            ).eval()
            lora_config = LoRAConfig(
                r=int(lora_cfg.get("r", 8)),
                alpha=float(lora_cfg.get("alpha", 16.0)),
                dropout=float(lora_cfg.get("dropout", 0.0)),
                target_modules=tuple(
                    lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
                ),
            )
            replaced = apply_lora_to_model(current_model, lora_config)
            print(f"  prepared LoRA backbone, modules={len(replaced)}, signature={sig}")
            current_model.generation_config.pad_token_id = tokenizer.pad_token_id
            current_sig = sig

        lora_state_dict = checkpoint.get("lora_state_dict")
        if not isinstance(lora_state_dict, dict):
            raise KeyError(f"LoRA checkpoint missing lora_state_dict: {ckpt_path}")
        model_state_keys = set(current_model.state_dict().keys())
        mapped_lora_state: dict[str, torch.Tensor] = {}
        for key, value in lora_state_dict.items():
            if key in model_state_keys:
                mapped_lora_state[key] = value
                continue
            prefixed = f"model.{key}"
            if prefixed in model_state_keys:
                mapped_lora_state[prefixed] = value
                continue
            if key.startswith("model.") and key[6:] in model_state_keys:
                mapped_lora_state[key[6:]] = value
                continue
            mapped_lora_state[key] = value
        load_lora_state_dict(current_model, mapped_lora_state, strict=True)

        target_name = target["name"]
        print(f"Evaluating target={target_name} type=lora ckpt={ckpt_path}")
        results[target_name] = evaluate_target_benchmarks(
            model=current_model,
            tokenizer=tokenizer,
            target_name=target_name,
            config=config,
            default_batch_size=default_batch_size,
            default_max_gen_len=default_max_gen_len,
            default_do_sample=default_do_sample,
            default_temperature=default_temperature,
            default_top_p=default_top_p,
            default_top_k=default_top_k,
            show_progress=show_progress,
            collect_format_stats=collect_format_stats,
        )
    return results


def save_eval_results(
    results: dict[str, Any],
    config: dict[str, Any],
    config_path: str,
    eval_targets: list[dict[str, Any]],
    output_dir: str,
) -> tuple[Path, Path]:
    run_time = datetime.now().strftime(r"%Y%m%d-%H%M%S")
    run_dir = Path(output_dir) / run_time
    run_dir.mkdir(parents=True, exist_ok=True)

    config_src = Path(config_path)
    config_dst = run_dir / config_src.name
    shutil.copy2(config_src, config_dst)

    payload = {
        "timestamp": run_time,
        "config_path": str(config_src.resolve()),
        "model": config.get("model", {}),
        "models": config.get("models", {}),
        "benchmarks": config.get("benchmarks", {}),
        "eval": config.get("eval", {}),
        "eval_targets": eval_targets,
        "mode": "transformers_r1_thinking_method",
        "r1_prefix": {
            "system_message": SYSTEM_MESSAGE,
            "response_prompt": RESPONSE_PROMPT,
        },
        "results": results,
    }

    json_path = run_dir / "results.json"
    yaml_path = run_dir / "results.yaml"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)
    return json_path, yaml_path


def main(config_path: str, include_types: list[str] | None = None):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    base_model_path = config["model"]["base_model_path"]
    eval_cfg = config["eval"]
    sampling = eval_cfg.get("sampling", {})
    do_sample = bool(sampling.get("do_sample", False))
    temperature = float(sampling.get("temperature", 1.0))
    top_p = float(sampling.get("top_p", 1.0))
    top_k = int(sampling.get("top_k", 0))
    batch_size = int(eval_cfg["batch_size"])
    max_new_tokens = int(eval_cfg["max_gen_len"])
    show_progress = bool(eval_cfg.get("show_progress", True))
    collect_format_stats = bool(eval_cfg.get("collect_format_stats", True))
    model_path = Path(base_model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            "base_model_path must be a local model directory for R1 thinking eval: "
            f"{model_path}"
        )

    eval_targets = build_eval_targets(config)
    if include_types is None:
        include_types = eval_cfg.get("include_types")
    eval_targets = filter_eval_targets(eval_targets, include_types=include_types)
    if not eval_targets:
        raise ValueError("No eval targets left after type filtering.")

    dtype = _dtype_from_config(config)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("R1 thinking eval prefix:")
    print(f"  SYSTEM_MESSAGE={SYSTEM_MESSAGE!r}")
    print(f"  RESPONSE_PROMPT={RESPONSE_PROMPT!r}")

    results: dict[str, Any] = {"benchmarks": {}}
    lora_targets = [t for t in eval_targets if t["type"] == "lora"]
    other_targets = [t for t in eval_targets if t["type"] != "lora"]

    for target in other_targets:
        target_name = target["name"]
        print(f"Evaluating target={target_name} type={target['type']}")
        model = load_model_for_target(
            base_model_path=model_path,
            target=target,
            dtype=dtype,
        )
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        results["benchmarks"][target_name] = evaluate_target_benchmarks(
            model=model,
            tokenizer=tokenizer,
            target_name=target_name,
            config=config,
            default_batch_size=batch_size,
            default_max_gen_len=max_new_tokens,
            default_do_sample=do_sample,
            default_temperature=temperature,
            default_top_p=top_p,
            default_top_k=top_k,
            show_progress=show_progress,
            collect_format_stats=collect_format_stats,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if lora_targets:
        lora_results = evaluate_lora_targets_r1(
            lora_targets=lora_targets,
            base_model_path=model_path,
            dtype=dtype,
            tokenizer=tokenizer,
            config=config,
            default_batch_size=batch_size,
            default_max_gen_len=max_new_tokens,
            default_do_sample=do_sample,
            default_temperature=temperature,
            default_top_p=top_p,
            default_top_k=top_k,
            show_progress=show_progress,
            collect_format_stats=collect_format_stats,
        )
        results["benchmarks"].update(lora_results)

    output_dir = eval_cfg.get("r1_output_dir", eval_cfg.get("output_dir", "logs/eval_r1"))
    json_path, yaml_path = save_eval_results(
        results=results,
        config=config,
        config_path=config_path,
        eval_targets=eval_targets,
        output_dir=output_dir,
    )
    print(f"Saved evaluation results to {json_path.parent}/")
    print(f"  - {json_path.name}")
    print(f"  - {yaml_path.name}")
    print(f"  - {Path(config_path).name}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate models with R1 thinking training prefix.")
    parser.add_argument("--config", type=str, default="configs/config_eval.yaml")
    parser.add_argument(
        "--types",
        type=str,
        default="",
        help="Comma-separated target types to evaluate, e.g. 'lora' or 'base,full'.",
    )
    args = parser.parse_args()
    include_types = [x.strip() for x in args.types.split(",")] if args.types else None
    main(args.config, include_types=include_types)
