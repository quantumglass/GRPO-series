import json
import shutil
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from benchmark_task import (
    GenericMathBenchmarkDataset,
    answer_matches,
    extract_pred_answer,
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
from qwen2_model import Transformer
from sampling import SamplingConfig, sample_next_token, sampling_config_from_dict
from tokenizer import Tokenizer


def _default_target_name(model_type: str, ckpt_path: str | None) -> str:
    if ckpt_path is None:
        return model_type
    stem = Path(ckpt_path).stem
    return f"{model_type}:{stem}"


def build_eval_targets(config: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Build concrete evaluation targets from config.

    Supports two interfaces:
    1) New: eval.model_targets (list of dict)
    2) Legacy: eval.model_types + models.{type}.ckpt_path/ckpt_paths

    Targets without checkpoint paths are skipped (with a log line), so
    --types / include_types can filter to lora-only even when full has no ckpt.
    """
    eval_cfg = config.get("eval", {})
    targets: list[dict[str, Any]] = []

    model_targets = eval_cfg.get("model_targets")
    if model_targets:
        for item in model_targets:
            if not isinstance(item, dict):
                raise ValueError("eval.model_targets items must be dict objects.")
            model_type = str(item["type"]).lower()
            target_name = str(item.get("name", model_type))
            if model_type == "base":
                targets.append(
                    {
                        "name": target_name,
                        "type": "base",
                        "ckpt_path": None,
                        "lora_config": item.get("lora_config"),
                    }
                )
                continue
            if model_type not in {"full", "lora"}:
                print(f"Skip unsupported target type: {model_type}")
                continue

            ckpt_paths: list[str] = []
            if "ckpt_paths" in item and item["ckpt_paths"]:
                ckpt_paths = [str(p) for p in item["ckpt_paths"] if p]
            elif item.get("ckpt_path"):
                ckpt_paths = [str(item["ckpt_path"])]
            if not ckpt_paths:
                print(f"Skip {model_type} target without ckpt_path(s): {target_name}")
                continue

            for idx, ckpt_path in enumerate(ckpt_paths):
                if item.get("name") and len(ckpt_paths) == 1:
                    name = str(item["name"])
                elif item.get("name"):
                    name = f"{item['name']}[{idx}]"
                else:
                    name = _default_target_name(model_type, ckpt_path)
                targets.append(
                    {
                        "name": name,
                        "type": model_type,
                        "ckpt_path": ckpt_path,
                        "lora_config": item.get("lora_config"),
                    }
                )
        if not targets:
            raise ValueError("No valid eval targets.")
        return targets

    # Legacy fallback
    model_types = eval_cfg.get("model_types", [])
    if not model_types:
        raise ValueError("Please set eval.model_targets or eval.model_types.")
    models_cfg = config.get("models", {})
    for model_type_raw in model_types:
        model_type = str(model_type_raw).lower()
        if model_type == "base":
            targets.append(
                {
                    "name": "base",
                    "type": "base",
                    "ckpt_path": None,
                    "lora_config": None,
                }
            )
            continue
        if model_type not in {"full", "lora"}:
            print(f"Skip unsupported target type: {model_type}")
            continue

        type_cfg = models_cfg.get(model_type, {})
        if "ckpt_paths" in type_cfg and type_cfg["ckpt_paths"]:
            ckpt_paths = [str(p) for p in type_cfg["ckpt_paths"] if p]
        elif type_cfg.get("ckpt_path"):
            ckpt_paths = [str(type_cfg["ckpt_path"])]
        else:
            ckpt_paths = []
        if not ckpt_paths:
            print(f"Skip legacy {model_type} target without ckpt_path(s)")
            continue
        for ckpt_path in ckpt_paths:
            targets.append(
                {
                    "name": _default_target_name(model_type, ckpt_path),
                    "type": model_type,
                    "ckpt_path": ckpt_path,
                    "lora_config": type_cfg.get("lora_config"),
                }
            )
    if not targets:
        raise ValueError("No valid eval targets.")
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


def resolve_benchmark_eval_params(
    *,
    config: dict[str, Any],
    benchmark_name: str,
    default_batch_size: int,
    default_max_gen_len: int,
    default_sampling: SamplingConfig,
) -> tuple[int, int, SamplingConfig]:
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
    if "sampling" in override_cfg and override_cfg["sampling"] is not None:
        sampling = sampling_config_from_dict(override_cfg["sampling"])
    else:
        sampling = default_sampling
    return batch_size, max_gen_len, sampling


class EvalProgress:
    """Simple in-place progress counter for evaluation loops."""

    def __init__(self, total: int, desc: str, *, display_k: int = 1):
        self.total = max(total, 1)
        self.desc = desc
        self.display_k = display_k
        self.done = 0
        self.pass_rate_sum = 0.0
        self.solved_sum = 0

    def update(self, n: int, *, pass_rate_delta: float = 0.0, solved_delta: int = 0) -> None:
        self.done += n
        self.pass_rate_sum += pass_rate_delta
        self.solved_sum += solved_delta
        rate = self.pass_rate_sum / self.done if self.done else 0.0
        pct = 100.0 * self.done / self.total
        bar_width = 24
        filled = int(bar_width * self.done / self.total)
        bar = "=" * filled + "-" * (bar_width - filled)
        print(
            f"\r{self.desc} [{bar}] {self.done}/{self.total} "
            f"({pct:5.1f}%) pass@{self.display_k}={rate:.4f} "
            f"({self.solved_sum}/{self.done})",
            end="",
            flush=True,
        )

    def close(self) -> None:
        print()


@torch.no_grad()
def generate(
    model: Transformer,
    tokenizer: Tokenizer,
    prefix_token_ids: list[list[int]],
    max_gen_len: int,
    device: torch.device,
    dtype: torch.dtype,
    sampling: SamplingConfig | None = None,
) -> list[str]:
    sampling = sampling or SamplingConfig(do_sample=False)
    end_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    bsz = len(prefix_token_ids)
    min_prompt_len = min(len(t) for t in prefix_token_ids)
    max_prompt_len = max(len(t) for t in prefix_token_ids)
    total_len = max_prompt_len + max_gen_len

    model.init_kv_cache(max_batch_size=bsz, max_seq_len=total_len, device=device, dtype=dtype)
    tokens = torch.full((bsz, total_len), pad_token_id, dtype=torch.long, device=device)
    for i, t in enumerate(prefix_token_ids):
        tokens[i, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)

    prev_pos = 0
    input_text_mask = tokens != pad_token_id
    is_finished = torch.zeros((bsz,), dtype=torch.bool, device=device)

    for cur_pos in range(min_prompt_len, total_len):
        with torch.autocast(device_type=device.type, dtype=dtype):
            logits = model.inference(tokens[:, prev_pos:cur_pos], prev_pos)
        next_token = sample_next_token(logits[:, -1], sampling)
        next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        next_token = torch.where(is_finished, torch.tensor(pad_token_id, device=device), next_token)
        is_generated = ~input_text_mask[:, cur_pos]
        tokens[:, cur_pos] = next_token
        if end_token_id is not None:
            is_finished = is_finished | ((next_token == end_token_id) & is_generated)
        prev_pos = cur_pos
        if is_finished.all():
            break

    model.del_kv_cache()
    token_list = tokens.tolist()
    outputs = []
    for i in range(bsz):
        prompt_len = len(prefix_token_ids[i])
        gen_token_ids = token_list[i][prompt_len:]
        if pad_token_id in gen_token_ids:
            gen_token_ids = gen_token_ids[: gen_token_ids.index(pad_token_id)]
        outputs.append(tokenizer.detokenize(gen_token_ids))
    return outputs


def load_model_for_eval(config: dict[str, Any], target: dict[str, Any]) -> Transformer:
    base_path = Path(config["model"]["base_model_path"])
    device = torch.device(config["model"]["device"])
    model = Transformer.from_pretrained(base_path, device=device).eval()
    model_type = target["type"]

    if model_type == "base":
        return model

    ckpt_path = Path(target["ckpt_path"])
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    if model_type == "full":
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Full checkpoint must be a state_dict dict: {ckpt_path}")
        model.load_state_dict(checkpoint, strict=True)
        return model.eval()

    if model_type == "lora":
        lora_cfg = checkpoint.get("lora_config", target.get("lora_config", {}))
        lora_config = LoRAConfig(
            r=lora_cfg.get("r", 8),
            alpha=lora_cfg.get("alpha", 16.0),
            dropout=lora_cfg.get("dropout", 0.0),
            target_modules=tuple(
                lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
            ),
        )
        apply_lora_to_model(model, lora_config)
        lora_state_dict = checkpoint.get("lora_state_dict")
        if lora_state_dict is None:
            raise KeyError(f"LoRA checkpoint missing lora_state_dict: {ckpt_path}")
        load_lora_state_dict(model, lora_state_dict, strict=True)
        return model.to(device).eval()

    raise ValueError(f"Unknown model_type: {model_type}")


def evaluate_target(
    *,
    target: dict[str, Any],
    model: Transformer,
    tokenizer: Tokenizer,
    config: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    show_progress: bool,
    sampling: SamplingConfig,
) -> dict[str, Any]:
    target_name = target["name"]
    model_result = {}
    eval_cfg = config.get("eval", {})
    for benchmark_name, benchmark_cfg in resolve_eval_benchmarks(
        config["benchmarks"], eval_cfg
    ):
        benchmark_batch_size, benchmark_max_gen_len, benchmark_sampling = (
            resolve_benchmark_eval_params(
                config=config,
                benchmark_name=benchmark_name,
                default_batch_size=config["eval"]["batch_size"],
                default_max_gen_len=config["eval"]["max_gen_len"],
                default_sampling=sampling,
            )
        )
        pass_at_k, num_samples = resolve_pass_at_k_config(
            config["eval"], benchmark_name=benchmark_name
        )
        print(f"  Benchmark: {benchmark_name}")
        print(
            "    params: "
            f"batch_size={benchmark_batch_size}, "
            f"max_gen_len={benchmark_max_gen_len}, "
            f"num_samples={num_samples}, pass_at_k={pass_at_k}, "
            f"do_sample={benchmark_sampling.do_sample}, "
            f"temperature={benchmark_sampling.temperature}, "
            f"top_p={benchmark_sampling.top_p}, top_k={benchmark_sampling.top_k}"
        )
        if num_samples > 1 and not benchmark_sampling.do_sample:
            print(
                "    warning: num_samples>1 with do_sample=false yields identical "
                "samples; enable sampling for meaningful pass@k>1."
            )
        one = evaluate_one_benchmark(
            model=model,
            tokenizer=tokenizer,
            benchmark_name=benchmark_name,
            benchmark_path=benchmark_cfg["path"],
            eval_batch_size=benchmark_batch_size,
            max_gen_len=benchmark_max_gen_len,
            device=device,
            dtype=dtype,
            progress_desc=f"{target_name}/{benchmark_name}",
            show_progress=show_progress,
            sampling=benchmark_sampling,
            pass_at_k=pass_at_k,
            num_samples=num_samples,
        )
        print(
            f"    accuracy={one['accuracy']:.4f} "
            f"({one['correct']}/{one['total']}), "
            f"{format_pass_at_k_summary(one['pass_at_k'], num_samples=num_samples)}"
        )
        model_result[benchmark_name] = one
    return model_result


def _lora_signature(lora_cfg: dict[str, Any]) -> tuple[Any, ...]:
    target_modules = tuple(lora_cfg.get("target_modules", []))
    return (
        int(lora_cfg.get("r", 8)),
        float(lora_cfg.get("alpha", 16.0)),
        float(lora_cfg.get("dropout", 0.0)),
        target_modules,
    )


def evaluate_lora_targets(
    *,
    lora_targets: list[dict[str, Any]],
    config: dict[str, Any],
    tokenizer: Tokenizer,
    device: torch.device,
    dtype: torch.dtype,
    show_progress: bool,
    sampling: SamplingConfig,
) -> dict[str, Any]:
    """
    Evaluate multiple LoRA checkpoints with lightweight reuse.

    If consecutive checkpoints share the same LoRA structure, we reuse a single
    already-initialized LoRA model and only swap lora_state_dict.
    """
    base_path = Path(config["model"]["base_model_path"])
    results: dict[str, Any] = {}
    current_model: Transformer | None = None
    current_sig: tuple[Any, ...] | None = None

    for target in lora_targets:
        ckpt_path = Path(target["ckpt_path"])
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        lora_cfg = checkpoint.get("lora_config", target.get("lora_config", {}))
        sig = _lora_signature(lora_cfg)
        if current_model is None or sig != current_sig:
            if current_model is not None:
                del current_model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            current_model = Transformer.from_pretrained(base_path, device=device).eval()
            lora_config = LoRAConfig(
                r=lora_cfg.get("r", 8),
                alpha=lora_cfg.get("alpha", 16.0),
                dropout=lora_cfg.get("dropout", 0.0),
                target_modules=tuple(
                    lora_cfg.get(
                        "target_modules",
                        ["q_proj", "k_proj", "v_proj", "o_proj"],
                    )
                ),
            )
            apply_lora_to_model(current_model, lora_config)
            current_model = current_model.to(device).eval()
            current_sig = sig
            print(f"Prepared LoRA backbone for signature={current_sig}")

        lora_state_dict = checkpoint.get("lora_state_dict")
        if lora_state_dict is None:
            raise KeyError(f"LoRA checkpoint missing lora_state_dict: {ckpt_path}")
        load_lora_state_dict(current_model, lora_state_dict, strict=True)
        print(f"Evaluating target={target['name']} ckpt={ckpt_path}")
        results[target["name"]] = evaluate_target(
            target=target,
            model=current_model,
            tokenizer=tokenizer,
            config=config,
            device=device,
            dtype=dtype,
            show_progress=show_progress,
            sampling=sampling,
        )

    return results


def evaluate_one_benchmark(
    model: Transformer,
    tokenizer: Tokenizer,
    benchmark_name: str,
    benchmark_path: str,
    eval_batch_size: int,
    max_gen_len: int,
    device: torch.device,
    dtype: torch.dtype,
    progress_desc: str | None = None,
    show_progress: bool = True,
    sampling: SamplingConfig | None = None,
    pass_at_k: list[int] | None = None,
    num_samples: int = 1,
) -> dict[str, Any]:
    pass_at_k = pass_at_k or [1]
    num_samples = max(int(num_samples), 1)
    question_batch_size = max(1, eval_batch_size // num_samples)
    display_k = max(pass_at_k)

    dataset = GenericMathBenchmarkDataset(
        tokenizer=tokenizer,
        dataset_name=benchmark_name,
        path=benchmark_path,
    )
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        collate_fn=GenericMathBenchmarkDataset.collate_fn,
        batch_size=question_batch_size,
    )

    total = len(dataset)
    sample_results: list[list[bool]] = []
    first_sample_correct: list[bool] = []
    desc = progress_desc or benchmark_name
    progress = (
        EvalProgress(total=total, desc=desc, display_k=display_k)
        if show_progress
        else None
    )
    try:
        for batch in dataloader:
            num_q = len(batch.ground_truth)
            expanded_prefix: list[list[int]] = []
            expanded_gt: list[str] = []
            for prefix_ids, gt in zip(batch.prefix_token_ids, batch.ground_truth):
                for _ in range(num_samples):
                    expanded_prefix.append(prefix_ids)
                    expanded_gt.append(gt)

            all_responses: list[str] = []
            for chunk_start in range(0, len(expanded_prefix), eval_batch_size):
                chunk_prefix = expanded_prefix[chunk_start : chunk_start + eval_batch_size]
                all_responses.extend(
                    generate(
                        model=model,
                        tokenizer=tokenizer,
                        prefix_token_ids=chunk_prefix,
                        max_gen_len=max_gen_len,
                        device=device,
                        dtype=dtype,
                        sampling=sampling,
                    )
                )

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
) -> tuple[Path, Path]:
    output_root = Path(config["eval"].get("output_dir", "logs/eval"))
    run_time = datetime.now().strftime(r"%Y%m%d-%H%M%S")
    run_dir = output_root / run_time
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

    device = torch.device(config["model"]["device"])
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(config["model"]["dtype"], torch.bfloat16)
    tokenizer = Tokenizer(str(Path(config["model"]["base_model_path"]) / "tokenizer.json"))

    show_progress = config["eval"].get("show_progress", True)
    sampling = sampling_config_from_dict(config["eval"].get("sampling"))
    eval_targets = build_eval_targets(config)
    if include_types is None:
        include_types = config["eval"].get("include_types")
    eval_targets = filter_eval_targets(eval_targets, include_types=include_types)
    if not eval_targets:
        raise ValueError("No eval targets left after type filtering.")
    print(
        "Eval sampling: "
        f"do_sample={sampling.do_sample}, temperature={sampling.temperature}, "
        f"top_p={sampling.top_p}, top_k={sampling.top_k}"
    )
    print("Eval targets:")
    for target in eval_targets:
        if target["ckpt_path"] is None:
            print(f"  - {target['name']} (type={target['type']})")
        else:
            print(
                f"  - {target['name']} (type={target['type']}, ckpt={target['ckpt_path']})"
            )
    results: dict[str, Any] = {"benchmarks": {}}
    lora_targets = [t for t in eval_targets if t["type"] == "lora"]
    other_targets = [t for t in eval_targets if t["type"] != "lora"]

    for target in other_targets:
        print(f"Evaluating target={target['name']} (type={target['type']})")
        model = load_model_for_eval(config, target)
        results["benchmarks"][target["name"]] = evaluate_target(
            target=target,
            model=model,
            tokenizer=tokenizer,
            config=config,
            device=device,
            dtype=dtype,
            show_progress=show_progress,
            sampling=sampling,
        )

    if lora_targets:
        lora_results = evaluate_lora_targets(
            lora_targets=lora_targets,
            config=config,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
            show_progress=show_progress,
            sampling=sampling,
        )
        results["benchmarks"].update(lora_results)

    json_path, yaml_path = save_eval_results(results, config, config_path, eval_targets)
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
        help="Comma-separated target types to evaluate, e.g. 'full,lora'.",
    )
    args = parser.parse_args()
    include_types = [x.strip() for x in args.types.split(",")] if args.types else None
    main(args.config, include_types=include_types)
