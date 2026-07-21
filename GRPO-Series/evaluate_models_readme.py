import json
import shutil
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark_task import (
    answer_matches,
    extract_pred_answer,
    is_aime_benchmark,
    normalize_aime_answer,
    normalize_math_answer,
)
from eval_metrics import (
    build_benchmark_result,
    format_pass_at_k_summary,
    is_pass_at_k_solved,
    pass_at_k_simple,
    resolve_eval_benchmarks,
    resolve_pass_at_k_config,
)
from lora import LoRAConfig, apply_lora_to_model, load_lora_state_dict


README_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


class EvalProgress:
    """Simple in-place progress counter for evaluation loops."""

    def __init__(self, total: int, desc: str, *, display_k: int = 1):
        self.total = max(total, 1)
        self.desc = desc
        self.display_k = display_k
        self.done = 0
        self.pass_rate_sum = 0.0
        self.solved_sum = 0
        self.gen_done = 0
        self.gen_total = 0

    def begin_generate(self, gen_total: int) -> None:
        self.gen_total = max(gen_total, 0)
        self.gen_done = 0
        self._render()

    def update_generate(self, n: int) -> None:
        if self.gen_total <= 0:
            return
        self.gen_done = min(self.gen_done + n, self.gen_total)
        self._render()

    def update(self, n: int, *, pass_rate_delta: float = 0.0, solved_delta: int = 0) -> None:
        self.done += n
        self.pass_rate_sum += pass_rate_delta
        self.solved_sum += solved_delta
        self.gen_done = 0
        self.gen_total = 0
        self._render()

    def _render(self) -> None:
        pct = 100.0 * self.done / self.total
        bar_width = 24
        filled = int(bar_width * self.done / self.total)
        bar = "=" * filled + "-" * (bar_width - filled)
        msg = (
            f"\r{self.desc} [{bar}] {self.done}/{self.total} ({pct:5.1f}%)"
        )
        if self.done > 0:
            rate = self.pass_rate_sum / self.done
            msg += (
                f" pass@{self.display_k}={rate:.4f} "
                f"({self.solved_sum}/{self.done})"
            )
        if self.gen_total > 0:
            gen_pct = 100.0 * self.gen_done / self.gen_total
            msg += f" | gen {self.gen_done}/{self.gen_total} ({gen_pct:5.1f}%)"
        print(msg, end="", flush=True)

    def close(self) -> None:
        print()


def build_eval_targets(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Build evaluation targets for README-style inference.

    Supported target types:
      - base
      - full
      - lora
    """
    eval_cfg = config.get("eval", {})
    model_targets = eval_cfg.get("model_targets")
    targets: list[dict[str, Any]] = []
    if model_targets:
        for item in model_targets:
            model_type = str(item["type"]).lower()
            name = str(item.get("name", model_type))
            if model_type == "base":
                targets.append({"name": name, "type": "base", "ckpt_path": None})
                continue
            if model_type not in {"full", "lora"}:
                print(f"Skip unsupported target type in README mode: {model_type}")
                continue
            ckpt_paths: list[str] = []
            if "ckpt_paths" in item and item["ckpt_paths"]:
                ckpt_paths = [str(p) for p in item["ckpt_paths"] if p]
            elif item.get("ckpt_path"):
                ckpt_paths = [str(item["ckpt_path"])]
            if not ckpt_paths:
                print(f"Skip {model_type} target without ckpt_path(s): {name}")
                continue
            for idx, ckpt_path in enumerate(ckpt_paths):
                if len(ckpt_paths) == 1:
                    final_name = name
                else:
                    final_name = f"{name}[{idx}]"
                targets.append(
                    {
                        "name": final_name,
                        "type": model_type,
                        "ckpt_path": ckpt_path,
                        "lora_config": item.get("lora_config"),
                    }
                )
    else:
        targets.append({"name": "base", "type": "base", "ckpt_path": None})
    if not targets:
        raise ValueError("No valid eval targets for README mode.")
    return targets


def filter_eval_targets(
    targets: list[dict[str, Any]],
    include_types: list[str] | None = None,
) -> list[dict[str, Any]]:
    if not include_types:
        return targets
    allow = {item.strip().lower() for item in include_types if item.strip()}
    if not allow:
        return targets
    valid_types = {"base", "full", "lora"}
    unknown = sorted(allow - valid_types)
    if unknown:
        raise ValueError(f"Unknown target types: {unknown}. Valid: {sorted(valid_types)}")
    return [t for t in targets if t["type"] in allow]


def _lora_signature(lora_cfg: dict[str, Any]) -> tuple[Any, ...]:
    target_modules = tuple(lora_cfg.get("target_modules", []))
    return (
        int(lora_cfg.get("r", 8)),
        float(lora_cfg.get("alpha", 16.0)),
        float(lora_cfg.get("dropout", 0.0)),
        target_modules,
    )


def resolve_benchmark_eval_params(
    *,
    config: dict[str, Any],
    benchmark_name: str,
    default_batch_size: int,
    default_max_gen_len: int,
    default_do_sample: bool,
    default_temperature: float,
    default_top_p: float,
    default_top_k: int,
) -> tuple[int, int, bool, float, float, int]:
    eval_cfg = config.get("eval", {})
    override_cfg = (eval_cfg.get("benchmark_overrides") or {}).get(benchmark_name, {})
    legacy_batch_size_map = eval_cfg.get("batch_size_by_benchmark", {}) or {}

    batch_size = int(
        override_cfg.get(
            "batch_size",
            legacy_batch_size_map.get(benchmark_name, default_batch_size),
        )
    )
    max_gen_len = int(override_cfg.get("max_gen_len", default_max_gen_len))
    sampling_cfg = override_cfg.get("sampling") or {}
    do_sample = bool(sampling_cfg.get("do_sample", default_do_sample))
    temperature = float(sampling_cfg.get("temperature", default_temperature))
    top_p = float(sampling_cfg.get("top_p", default_top_p))
    top_k = int(sampling_cfg.get("top_k", default_top_k))
    return batch_size, max_gen_len, do_sample, temperature, top_p, top_k


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".parquet":
        return pd.read_parquet(path).to_dict(orient="records")
    if path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and isinstance(payload.get("data"), list):
            return payload["data"]
    raise ValueError(f"Unsupported benchmark file type: {path}")


def extract_question(row: dict[str, Any]) -> str:
    for key in ("question", "problem", "query", "input"):
        if key in row and row[key] is not None:
            return str(row[key])
    raise KeyError("Cannot find question field in row.")


def extract_ground_truth(row: dict[str, Any], benchmark_name: str) -> str:
    name = benchmark_name.lower().replace("_", "-")
    if name == "gsm8k":
        answer = str(row.get("answer", ""))
        marker = "####"
        if marker in answer:
            return answer.split(marker)[-1].strip().replace(",", "")
        return answer.strip()

    if name.replace("-", "") == "math500":
        if row.get("answer") is not None:
            return normalize_math_answer(str(row["answer"]))
        return normalize_math_answer(str(row.get("solution", "")).strip())

    if is_aime_benchmark(name):
        if row.get("answer") is not None:
            return normalize_aime_answer(str(row["answer"]))
        raise KeyError("AIME row is missing required field 'answer'.")

    for key in ("answer", "final_answer", "ground_truth", "target"):
        if key in row and row[key] is not None:
            return str(row[key]).strip()
    raise KeyError("Cannot find ground truth field in row.")


def batched(iterable: list[Any], batch_size: int):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]


def _is_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return ("out of memory" in msg and "cuda" in msg) or isinstance(
        exc, torch.OutOfMemoryError
    )


def _run_generate_with_auto_batch_fallback(
    *,
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    benchmark_name: str,
) -> list[str]:
    """Generate with adaptive batch-size fallback on CUDA OOM."""
    responses: list[str] = []
    start = 0
    current_bs = len(prompts)
    while start < len(prompts):
        current_bs = min(current_bs, len(prompts) - start)
        chunk = prompts[start : start + current_bs]
        try:
            model_inputs = tokenizer(chunk, return_tensors="pt", padding=True).to(model.device)
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if do_sample:
                gen_kwargs.update(
                    {
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": top_k,
                    }
                )
            generated_ids = model.generate(**model_inputs, **gen_kwargs)
            trimmed_ids = []
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids):
                trimmed_ids.append(output_ids[len(input_ids) :])
            responses.extend(tokenizer.batch_decode(trimmed_ids, skip_special_tokens=True))
            start += current_bs
        except RuntimeError as exc:
            if not _is_oom_error(exc):
                raise
            if current_bs <= 1:
                raise RuntimeError(
                    f"OOM even at batch_size=1 for benchmark={benchmark_name}. "
                    "Please reduce eval.max_gen_len or use a larger GPU."
                ) from exc
            next_bs = max(1, current_bs // 2)
            print(
                f"\n[OOM] benchmark={benchmark_name}, chunk_bs={current_bs} -> retry with {next_bs}",
                flush=True,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            current_bs = next_bs
            continue
    return responses


def _dtype_from_config(config: dict[str, Any]) -> torch.dtype | str:
    dtype_name = str(config.get("model", {}).get("dtype", "auto")).lower()
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    return "auto"


def load_model_for_target(
    *,
    base_model_path: Path,
    target: dict[str, Any],
    dtype: torch.dtype | str,
):
    model = AutoModelForCausalLM.from_pretrained(
        str(base_model_path),
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    ).eval()

    if target["type"] == "full":
        ckpt_path = Path(target["ckpt_path"])
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Full checkpoint not found: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location="cpu")
        if not isinstance(state_dict, dict):
            raise ValueError(f"Full checkpoint must be a state_dict dict: {ckpt_path}")

        mapped_state_dict: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if not isinstance(value, torch.Tensor):
                continue
            if key.startswith("model.") or key == "lm_head.weight":
                mapped_key = key
            else:
                mapped_key = f"model.{key}"
            mapped_state_dict[mapped_key] = value

        load_res = model.load_state_dict(mapped_state_dict, strict=False)
        print(
            f"  loaded full ckpt={ckpt_path} "
            f"(missing={len(load_res.missing_keys)}, unexpected={len(load_res.unexpected_keys)})"
        )
        return model

    if target["type"] == "lora":
        ckpt_path = Path(target["ckpt_path"])
        if not ckpt_path.exists():
            raise FileNotFoundError(f"LoRA checkpoint not found: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise ValueError(f"LoRA checkpoint must be a dict: {ckpt_path}")

        lora_cfg = checkpoint.get("lora_config", target.get("lora_config", {}))
        lora_config = LoRAConfig(
            r=int(lora_cfg.get("r", 8)),
            alpha=float(lora_cfg.get("alpha", 16.0)),
            dropout=float(lora_cfg.get("dropout", 0.0)),
            target_modules=tuple(
                lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
            ),
        )
        replaced = apply_lora_to_model(model, lora_config)
        print(f"  applied LoRA modules={len(replaced)} from ckpt={ckpt_path}")

        lora_state_dict = checkpoint.get("lora_state_dict")
        if not isinstance(lora_state_dict, dict):
            raise KeyError(f"LoRA checkpoint missing lora_state_dict: {ckpt_path}")

        # Map custom-project keys to HF-style keys when needed.
        model_state_keys = set(model.state_dict().keys())
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
        load_lora_state_dict(model, mapped_lora_state, strict=True)
        return model
    return model


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
            progress_desc=f"readme/{target_name}/{benchmark_name}",
            show_progress=show_progress,
            pass_at_k=pass_at_k,
            num_samples=num_samples,
        )
        print(
            f"  accuracy={one['accuracy']:.4f} ({one['correct']}/{one['total']}), "
            f"{format_pass_at_k_summary(one['pass_at_k'], num_samples=num_samples)}"
        )
        per_target[benchmark_name] = one
    return per_target


def evaluate_lora_targets_with_reuse(
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
) -> dict[str, Any]:
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
        )
    return results


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
                messages = [
                    {"role": "system", "content": README_SYSTEM_PROMPT},
                    {"role": "user", "content": q},
                ]
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
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
                pred = extract_pred_answer(response, dataset_name=benchmark_name)
                is_correct = answer_matches(pred, gt, dataset_name=benchmark_name)
                batch_samples[qi][si] = is_correct
                if si == 0 and is_correct:
                    batch_first_correct[qi] = True
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

    return build_benchmark_result(
        sample_results=sample_results,
        num_samples=num_samples,
        pass_at_k=pass_at_k,
        first_sample_correct=first_sample_correct,
    )


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
        "mode": "transformers_readme_method",
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
    sampling = config["eval"].get("sampling", {})
    do_sample = bool(sampling.get("do_sample", False))
    temperature = float(sampling.get("temperature", 1.0))
    top_p = float(sampling.get("top_p", 1.0))
    top_k = int(sampling.get("top_k", 0))
    batch_size = int(config["eval"]["batch_size"])
    max_new_tokens = int(config["eval"]["max_gen_len"])
    show_progress = bool(config["eval"].get("show_progress", True))
    model_path = Path(base_model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            "base_model_path must be a local model directory for readme eval: "
            f"{model_path}"
        )
    eval_targets = build_eval_targets(config)
    if include_types is None:
        include_types = config["eval"].get("include_types")
    eval_targets = filter_eval_targets(eval_targets, include_types=include_types)
    if not eval_targets:
        raise ValueError("No eval targets left after type filtering.")
    dtype = _dtype_from_config(config)

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
    )
    # Decoder-only models (Qwen) should use left padding for batched generation.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
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
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if lora_targets:
        lora_results = evaluate_lora_targets_with_reuse(
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
        )
        results["benchmarks"].update(lora_results)

    output_dir = config["eval"].get("output_dir", "logs/eval")
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
    parser = ArgumentParser()
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
