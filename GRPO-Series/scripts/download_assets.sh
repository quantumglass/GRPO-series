#!/usr/bin/env bash
# 下载训练与评测所需的模型与数据集
# 用法: bash scripts/download_assets.sh [model|model7b|countdown|dapo|math500|gsm8k|deepscaler|competition_math|all]

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

download_model() {
  if [ -d "models/Qwen2.5-3B-Instruct" ]; then
    echo "[skip] models/Qwen2.5-3B-Instruct 已存在"
    return
  fi
  echo "[download] Qwen2.5-3B-Instruct ..."
  git lfs install
  mkdir -p models
  git clone https://huggingface.co/Qwen/Qwen2.5-3B-Instruct models/Qwen2.5-3B-Instruct
}

download_model7b() {
  if [ -d "models/Qwen2.5-Math-7B" ]; then
    echo "[skip] models/Qwen2.5-Math-7B 已存在"
    return
  fi
  echo "[download] Qwen2.5-Math-7B ..."
  git lfs install
  mkdir -p models
  git clone https://huggingface.co/Qwen/Qwen2.5-Math-7B models/Qwen2.5-Math-7B
}

download_countdown() {
  if [ -d "data/Countdown-Tasks-3to4" ]; then
    echo "[skip] data/Countdown-Tasks-3to4 已存在"
    return
  fi
  echo "[download] Countdown-Tasks-3to4 ..."
  git lfs install
  git clone https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4 data/Countdown-Tasks-3to4
}

download_dapo() {
  if [ -d "data/DAPO-Math-17k" ]; then
    echo "[skip] data/DAPO-Math-17k 已存在"
    return
  fi
  echo "[download] DAPO-Math-17k ..."
  git lfs install
  git clone https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k data/DAPO-Math-17k
}

download_math500() {
  if [ -f "data/MATH-500/test.jsonl" ]; then
    echo "[skip] data/MATH-500/test.jsonl 已存在"
    return
  fi
  echo "[download] MATH-500 ..."
  mkdir -p data/MATH-500
  git lfs install
  git clone --depth 1 https://huggingface.co/datasets/HuggingFaceH4/MATH-500 data/MATH-500-tmp
  cp data/MATH-500-tmp/test.jsonl data/MATH-500/
  rm -rf data/MATH-500-tmp
}

download_gsm8k() {
  if [ -f "data/gsm8k/test-00000-of-00001.parquet" ]; then
    echo "[skip] data/gsm8k 已存在"
    return
  fi
  echo "[download] GSM8K test split ..."
  mkdir -p data/gsm8k
  uv run python - <<'PY'
from datasets import load_dataset
ds = load_dataset("gsm8k", "main", split="test")
ds.to_parquet("data/gsm8k/test-00000-of-00001.parquet")
print(f"Saved {len(ds)} rows")
PY
}

download_deepscaler() {
  if [ -f "data/deepscaler/deepscaler.json" ]; then
    echo "[skip] data/deepscaler/deepscaler.json 已存在"
    return
  fi
  echo "[download] DeepScaleR dataset ..."
  mkdir -p data/deepscaler
  uv run python - <<'PY'
import json
from pathlib import Path
from datasets import load_dataset

ds = load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split="train")
records = [
    {"problem": r["problem"], "answer": r["answer"], "solution": r.get("solution", "")}
    for r in ds
]
out = Path("data/deepscaler/deepscaler.json")
out.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
print(f"Saved {len(records)} records to {out}")
PY
}

download_competition_math() {
  out="data/competition_math/data/train-00000-of-00001.parquet"
  if [ -f "$out" ]; then
    echo "[skip] $out 已存在"
    return
  fi
  echo "[download] competition_math (Hendrycks MATH) ..."
  mkdir -p data/competition_math/data
  uv run python - <<'PY'
from pathlib import Path
from datasets import load_dataset

# Prefer parquet-ready mirrors; fall back to original HF dataset.
candidates = [
    ("Maxwell-Jia/MATH", "train"),
    ("qwedsacf/competition_math", "train"),
    ("hendrycks/competition_math", "train"),
]
last_err = None
ds = None
for name, split in candidates:
    try:
        print(f"trying {name} ...")
        ds = load_dataset(name, split=split)
        break
    except Exception as e:
        last_err = e
        print(f"  failed: {e}")
if ds is None:
    raise RuntimeError(f"Failed to download competition_math: {last_err}")

out = Path("data/competition_math/data/train-00000-of-00001.parquet")
ds.to_parquet(out)
print(f"Saved {len(ds)} rows to {out}")
PY
}

TARGET="${1:-all}"
case "$TARGET" in
  model)             download_model ;;
  model7b)           download_model7b ;;
  countdown)         download_countdown ;;
  dapo)              download_dapo ;;
  math500)           download_math500 ;;
  gsm8k)             download_gsm8k ;;
  deepscaler)        download_deepscaler ;;
  competition_math)  download_competition_math ;;
  all)
    download_model
    download_countdown
    download_dapo
    download_math500
    download_gsm8k
    download_deepscaler
    download_competition_math
    ;;
  *)
    echo "未知目标: $TARGET"
    echo "可选: model model7b countdown dapo math500 gsm8k deepscaler competition_math all"
    exit 1
    ;;
esac

echo "完成。"
