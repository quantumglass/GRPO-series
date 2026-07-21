# 评测自检流程

[English](evaluation.md) | **简体中文**

本文档说明如何运行基准评测并排查分数异常。训练算法与超参见 [training_zh-CN.md](training_zh-CN.md)。

## 1. 环境与依赖

```bash
uv sync
```

- `evaluate_models.py` — 使用项目自研模型栈（`qwen2_model.py` + `tokenizer.py`）
- `evaluate_models_readme.py` — 使用 `transformers` 口径（对齐 Qwen 官方 README）
- `evaluate_models_r1_thinking.py` — R1 thinking 格式评测
- `evaluate_models_math7b.py` — Qwen2.5-Math-7B base 评测

## 2. 数据集完整性检查

```bash
uv run python - <<'PY'
import json
import pandas as pd
from pathlib import Path

gsm = Path("data/gsm8k/test-00000-of-00001.parquet")
math = Path("data/MATH-500/test.jsonl")

df = pd.read_parquet(gsm)
print("gsm8k rows =", len(df), "cols =", list(df.columns))

with open(math, encoding="utf-8") as f:
    first = json.loads(f.readline())
    print("math500 first keys =", list(first.keys()))
    n = sum(1 for _ in f) + 1
print("math500 rows =", n)
PY
```

预期：GSM8K 约 1319 条（字段含 `question`、`answer`）；MATH-500 500 条（字段含 `problem`、`answer`）。

## 3. 运行评测

```bash
uv run python evaluate_models.py --config configs/eval.yaml
uv run python evaluate_models_readme.py --config configs/eval.yaml --types base
```

评测 LoRA 时，在 `configs/eval.yaml` 的 `eval.model_targets` 中填写 checkpoint 路径：

```yaml
- name: "lora_ckpt"
  type: "lora"
  ckpt_paths:
    - "logs/<run_dir>/ckpt_000100.pt"
```

## 4. 结果对比

```bash
python - <<'PY'
import json, glob

def latest(pattern):
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None

ours = latest("logs/eval/*/results.json")
readme = latest("logs/eval_readme/*/results.json")
print("ours =", ours)
print("readme =", readme)
if ours and readme:
    a = json.load(open(ours, encoding="utf-8"))
    b = json.load(open(readme, encoding="utf-8"))
    print("ours =", a.get("results", a).get("benchmarks"))
    print("readme =", b.get("results", b).get("benchmarks"))
PY
```

## 5. 分数异常排查

1. **Prompt 口径**：system prompt 与 chat template 是否一致
2. **解码口径**：`do_sample` / `temperature` / `max_gen_len` 是否对齐
3. **答案匹配**：是否要求 `<think>` / `<answer>` / `\boxed{}`，数学归一化是否一致
