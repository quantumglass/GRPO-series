# Datasets

**English** | [简体中文](README_zh-CN.md)

Training and evaluation data live in this directory. Large files are not tracked in Git — use `scripts/download_assets.sh` to download them.

## Layout

```
data/
├── deepscaler/deepscaler.json        # DeepScaleR math training set (~40k)
├── Countdown-Tasks-3to4/             # Countdown arithmetic task
├── DAPO-Math-17k/                    # DAPO math training set
├── gsm8k/test-00000-of-00001.parquet # GSM8K eval set
└── MATH-500/test.jsonl               # MATH-500 eval set
```

## Download

```bash
bash scripts/download_assets.sh all        # everything
bash scripts/download_assets.sh deepscaler # DeepScaler only
bash scripts/download_assets.sh math500    # MATH-500 only
```

## Sources

| Dataset | HuggingFace | Use |
|---------|-------------|-----|
| DeepScaleR | [agentica-org/DeepScaleR-Preview-Dataset](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset) | Math RL training |
| Countdown | [Jiayi-Pan/Countdown-Tasks-3to4](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4) | Arithmetic RL training |
| DAPO-Math-17k | [BytedTsinghua-SIA/DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) | Math RL training |
| GSM8K | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | Evaluation |
| MATH-500 | [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | Evaluation |

## DeepScaler JSON schema

```json
[
  {
    "problem": "math problem text",
    "answer": "ground truth (may contain LaTeX)",
    "solution": "full solution (unused during RL training)"
  }
]
```
