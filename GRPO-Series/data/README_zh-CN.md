# 数据集说明

[English](README.md) | **简体中文**

本目录存放训练与评测数据。大文件不纳入 Git，请通过 `scripts/download_assets.sh` 获取。

## 目录结构

```
data/
├── deepscaler/deepscaler.json       # DeepScaleR 数学训练集（~40k 条）
├── Countdown-Tasks-3to4/            # Countdown 算术任务
├── DAPO-Math-17k/                   # DAPO 数学训练集
├── gsm8k/test-00000-of-00001.parquet # GSM8K 评测集
└── MATH-500/test.jsonl              # MATH-500 评测集
```

## 下载方式

```bash
bash scripts/download_assets.sh all
bash scripts/download_assets.sh deepscaler
bash scripts/download_assets.sh math500
```

## 数据来源

| 数据集 | HuggingFace 链接 | 用途 |
|--------|------------------|------|
| DeepScaleR | [agentica-org/DeepScaleR-Preview-Dataset](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset) | 数学 RL 训练 |
| Countdown | [Jiayi-Pan/Countdown-Tasks-3to4](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4) | 算术 RL 训练 |
| DAPO-Math-17k | [BytedTsinghua-SIA/DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) | 数学 RL 训练 |
| GSM8K | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | 评测 |
| MATH-500 | [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | 评测 |

## DeepScaler JSON 格式

```json
[
  {
    "problem": "数学题文本",
    "answer": "标准答案（可含 LaTeX）",
    "solution": "完整解析（训练时不使用）"
  }
]
```
