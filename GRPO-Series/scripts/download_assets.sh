#!/usr/bin/env bash
# 下载训练与评测所需的模型与数据集
# 用法: bash scripts/download_assets.sh [model|countdown|dapo|math500|gsm8k|deepscaler|all]

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
  git clone https://huggingface.co/Qwen/Qwen2.5-3B-Instruct models/Qwen2.5-3B-Instruct
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
  echo "[download] GSM8K test split (via huggingface-cli) ..."
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
records = [{"problem": r["problem"], "answer": r["answer"], "solution": r.get("solution", "")} for r in ds]
out = Path("data/deepscaler/deepscaler.json")
out.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
print(f"Saved {len(records)} records to {out}")
PY
}

TARGET="${1:-all}"
case "$TARGET" in
  model)       download_model ;;
  countdown)   download_countdown ;;
  dapo)        download_dapo ;;
  math500)     download_math500 ;;
  gsm8k)       download_gsm8k ;;
  deepscaler)  download_deepscaler ;;
  all)
    download_model
    download_countdown
    download_dapo
    download_math500
    download_gsm8k
    download_deepscaler
    ;;
  *) echo "未知目标: $TARGET"; echo "可选: model countdown dapo math500 gsm8k deepscaler all"; exit 1 ;;
esac

echo "完成。"
