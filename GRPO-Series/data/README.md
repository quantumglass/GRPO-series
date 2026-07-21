# Datasets

**English** | [简体中文](README_zh-CN.md)

Large data files are **not** tracked in Git. Download with:

```bash
bash scripts/download_assets.sh all
# or: model | model7b | countdown | dapo | math500 | gsm8k | deepscaler | competition_math
```

## Layout

```
data/
├── deepscaler/deepscaler.json
├── Countdown-Tasks-3to4/
├── DAPO-Math-17k/
├── competition_math/data/train-00000-of-00001.parquet
├── gsm8k/test-00000-of-00001.parquet
└── MATH-500/test.jsonl
```

## Sources

| Dataset | HuggingFace | Use |
|---------|-------------|-----|
| DeepScaleR | [agentica-org/DeepScaleR-Preview-Dataset](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset) | Math RL train |
| Countdown | [Jiayi-Pan/Countdown-Tasks-3to4](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4) | Arithmetic RL |
| DAPO-Math-17k | [BytedTsinghua-SIA/DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) | Math RL train |
| competition_math | [hendrycks/competition_math](https://huggingface.co/datasets/hendrycks/competition_math) (or parquet mirrors) | MATH train |
| GSM8K | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | Eval |
| MATH-500 | [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | Eval |

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
