# Evaluation Self-Check

**English** | [简体中文](evaluation_zh-CN.md)

How to run benchmark evaluation and debug score anomalies. For training algorithms and hyperparameters, see [training.md](training.md).

## 1. Environment

```bash
uv sync
```

- `evaluate_models.py` — native model stack (`qwen2_model.py` + `tokenizer.py`)
- `evaluate_models_readme.py` — `transformers` API (aligned with Qwen official README)
- `evaluate_models_r1_thinking.py` — R1 thinking-format evaluation
- `evaluate_models_math7b.py` — Qwen2.5-Math-7B base evaluation

## 2. Data integrity check

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

Expected: GSM8K ~1319 rows (`question`, `answer`); MATH-500 500 rows (`problem`, `answer`).

## 3. Run evaluation

```bash
uv run python evaluate_models.py --config configs/eval.yaml
uv run python evaluate_models_readme.py --config configs/eval.yaml --types base
```

For LoRA checkpoints, set paths in `configs/eval.yaml`:

```yaml
- name: "lora_ckpt"
  type: "lora"
  ckpt_paths:
    - "logs/<run_dir>/ckpt_000100.pt"
```

## 4. Compare results

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

## 5. Debugging low scores

1. **Prompt**: system prompt and chat template consistency
2. **Decoding**: `do_sample` / `temperature` / `max_gen_len` alignment
3. **Answer matching**: `<think>` / `<answer>` / `\boxed{}` requirements and math normalization
