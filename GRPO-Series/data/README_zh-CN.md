# 数据集

[English](README.md) | **简体中文**

大数据文件 **不** 纳入 Git。请用：

```bash
bash scripts/download_assets.sh all
# 或: model | model7b | countdown | dapo | math500 | gsm8k | deepscaler | competition_math
```

## 目录结构

```
data/
├── deepscaler/deepscaler.json
├── Countdown-Tasks-3to4/
├── DAPO-Math-17k/
├── competition_math/data/train-00000-of-00001.parquet
├── gsm8k/test-00000-of-00001.parquet
└── MATH-500/test.jsonl
```

## 来源

| 数据集 | HuggingFace | 用途 |
|--------|-------------|------|
| DeepScaleR | [agentica-org/DeepScaleR-Preview-Dataset](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset) | 数学 RL 训练 |
| Countdown | [Jiayi-Pan/Countdown-Tasks-3to4](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4) | 算术 RL |
| DAPO-Math-17k | [BytedTsinghua-SIA/DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) | 数学 RL |
| competition_math | [hendrycks/competition_math](https://huggingface.co/datasets/hendrycks/competition_math) | MATH 训练 |
| GSM8K | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | 评测 |
| MATH-500 | [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | 评测 |
