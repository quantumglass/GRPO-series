# GRPO-Series

[English](README.md) | **简体中文**

面向 **GRPO 算法族** 的单卡友好强化学习代码库，基于 [policy-gradient/GRPO-Zero](https://github.com/policy-gradient/GRPO-Zero) 扩展而来。在保留上游「从零实现、低依赖」理念的同时，增加了数学推理任务、LoRA 微调、多种算法变体，以及完整的评测管线。

---

## 代码性质说明

**GRPO-Series** 是一个面向 **学习与实验** 的训练框架 —— 不是生产级 RL 系统，也不是论文级精确复现代码库。

| 属性 | 说明 |
|------|------|
| **定位** | 帮助新手在 **单张 GPU** 上跑通并调参 GRPO 系列算法 |
| **实现方式** | 核心模型（`qwen2_model.py`）、分词器、RL 训练循环均用 **纯 PyTorch 从零编写**；训练阶段 **不依赖** `transformers` 或 `vLLM` |
| **算法范围** | 提供 GRPO / GSPO / PPO 裁剪等 **主干实现**，足以建立直觉与调参经验，但 **未包含** 各论文中的全部 trick |
| **代码来源** | 基本在 GRPO-Zero 基础上由 AI 辅助完成开发；优先考虑可读性与可改动性，而非刷榜分数 |
| **硬件目标** | 单卡 NVIDIA GPU（24–48 GB 显存）；内置 LoRA、分块 rollout、优化器 CPU offload 等省显存手段 |

> ⚠️ **请注意**：本代码库基本由 AI 完成，旨在满足新手在单卡上测试 GRPO 系列算法的需求。其间方法均为主干内容的实现，用于形成基本方法与调参经验，**未必包含**对应论文中的所有 trick。若需精细复现，还请参考各方法的官方代码库。

### 它 **不是** 什么

- ❌ 官方 DAPO / DeepSeek-R1 / verl / OpenRLHF 等框架的平替
- ❌ 分布式多机多卡 RL 训练系统
- ❌ 论文结果的精确复现或榜单成绩保证
- ❌ 经过大量工程打磨、覆盖所有边界情况的生产管线

如果你需要 **忠实复现** 某篇论文，请使用该方法的官方仓库。GRPO-Series 的定位是 **垫脚石**：先理解训练循环、尝试超参数，再迁移到更重的框架。

---

## 与 GRPO-Zero 的关系

本项目 fork 自 [GRPO-Zero](https://github.com/policy-gradient/GRPO-Zero)，保留其核心设计并做如下扩展：

| 维度 | GRPO-Zero（上游） | GRPO-Series（本仓库） |
|------|-------------------|----------------------|
| 任务 | Countdown 算术 | + DeepScaler、DAPO-Math-17k；GSM8K / MATH-500 评测 |
| 训练 | 全参数微调 | + LoRA（支持续训） |
| 算法 | Token 级 GRPO | + PPO 裁剪、GSPO、多 epoch 更新、KL 惩罚、零优势组过滤 |
| 奖励 | format + answer | + R1 复合奖励（准确率 + 格式 + 长度） |
| 显存 | MemoryEfficientAdamW | + 分块 rollout（`grpo_efficient.py`） |
| 评测 | 训练内简单 eval | 独立评测脚本 + pass@k |

核心模型实现（`qwen2_model.py`、`tokenizer.py`、`grpo.py`）仍保持从零编写；`evaluate_models_readme.py` 可选使用 `transformers` 做基线对比。

---

## 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│  入口脚本                                                    │
│  train_unified.py  train_r1_thinking.py  train.py           │
│  evaluate_models.py  evaluate_models_readme.py                │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  算法层                                                      │
│  grpo.py（rollout + update_policy）                          │
│  grpo_efficient.py（分块 rollout 降显存）                     │
│  sampling.py  data_types.py                                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  模型层（从零实现）                                           │
│  qwen2_model.py  tokenizer.py  lora.py  optimizer.py          │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  任务层（数据集 + 奖励）                                      │
│  countdown_task.py  deepscaler_task.py  dapo_math_task.py     │
│  benchmark_task.py  r1_thinking_reward.py                     │
└─────────────────────────────────────────────────────────────┘
```

**训练循环（简化）：**

1. 采样一批问题 → 每题自回归生成 M 个回答（rollout）
2. 计算奖励（格式 + 正确性，或 R1 复合奖励）
3. 组内优势归一化：\(A = (r - \mu) / \sigma\)
4. 用裁剪代理损失更新策略（GRPO token 级或 GSPO 响应级）
5. 写入 TensorBoard；保存 LoRA checkpoint

---

## 支持的算法变体

| 模式 | 配置项 | 说明 |
|------|--------|------|
| **GRPO** | `advantage_mode: grpo` | Token 级策略梯度 + 组内相对优势 |
| **GSPO** | `advantage_mode: gspo` | 响应级 importance ratio；可选长度缩放裁剪 |
| **PPO 裁剪** | `use_ppo_clip: true` | 对 importance ratio 做裁剪；`ppo_epochs > 1` 时生效 |
| **KL 惩罚** | `beta > 0` | 相对参考策略的 KL 正则项 |
| **零优势过滤** | `drop_zero_adv_groups: true` | 跳过组内奖励完全相同（无梯度信号）的样本组 |

---

## 项目结构

```
GRPO-Series/
├── train_unified.py           # ★ 主入口：多数据集 + LoRA + GSPO/PPO
├── train_r1_thinking.py       # R1 风格长思考训练
├── train.py                   # 上游兼容：Countdown 任务
├── train_deepscaler_legacy.py # 早期简化 DeepScaler 训练
├── evaluate_models.py         # 基准评测（自研模型栈）
├── evaluate_models_readme.py  # 基线评测（transformers 口径）
├── grpo.py                    # GRPO/GSPO/PPO 核心逻辑
├── grpo_efficient.py          # 分块 rollout（降低显存峰值）
├── qwen2_model.py             # Qwen2 Transformer 从零实现
├── tokenizer.py               # Tokenizer + Chat Template
├── lora.py / checkpoint.py    # LoRA 注入与续训
├── *_task.py                  # 各任务数据集与奖励函数
├── configs/                   # 训练与评测配置
├── scripts/download_assets.sh # 一键下载模型与数据集
└── docs/evaluation.md         # 评测自检流程
```

---

## 环境安装

要求 Python >= 3.11，推荐 NVIDIA GPU（数学 LoRA 训练建议 32 GB+ 显存）。

```bash
pip install uv
uv sync
```

## 数据与模型准备

```bash
# 一键下载
bash scripts/download_assets.sh all

# 或手动下载
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-3B-Instruct models/Qwen2.5-3B-Instruct
git clone https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4 data/Countdown-Tasks-3to4
git clone https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k data/DAPO-Math-17k
```

DeepScaler 数据可通过 `scripts/download_assets.sh deepscaler` 自动导出。GSM8K 与 MATH-500 说明见 `data/README.md`。

---

## 快速开始

### 1. 统一训练（推荐）

```bash
uv run python train_unified.py --config configs/train_unified.yaml
```

修改 `configs/train_unified.yaml` 中 `dataset.name` 切换数据集：`countdown` / `dapo_math_17k` / `deepscaler`。

### 2. R1 风格长思考训练

```bash
uv run python train_r1_thinking.py --config configs/train_r1_thinking.yaml
```

奖励为 `accuracy + format + length`，prefix 预填 `<think>` 引导推理模式。

### 3. Countdown 任务（上游兼容）

```bash
uv run python train.py --config configs/train_countdown.yaml          # 48GB
uv run python train.py --config configs/train_countdown_24gb.yaml     # 24GB
```

### 4. LoRA 续训

```bash
uv run python train_unified.py \
  --config configs/train_unified.yaml \
  --resume_lora_ckpt logs/<run_dir>/ckpt_000100.pt
```

### 5. 基准评测

```bash
uv run python evaluate_models.py --config configs/eval.yaml
uv run python evaluate_models_readme.py --config configs/eval.yaml --types base
```

评测前在 `configs/eval.yaml` 中填写 LoRA checkpoint 路径。详细流程见 [docs/evaluation.md](docs/evaluation.md)（[English](docs/evaluation.md) | [中文](docs/evaluation_zh-CN.md)）。

---

## 配置说明

| 配置文件 | 用途 |
|----------|------|
| `configs/train_unified.yaml` | 主训练：DeepScaler + LoRA + GSPO |
| `configs/train_r1_thinking.yaml` | R1 风格思考训练 |
| `configs/train_countdown.yaml` | Countdown 全参数微调 |
| `configs/train_countdown_24gb.yaml` | Countdown 低显存配置 |
| `configs/train_deepscaler_legacy.yaml` | 早期简化 DeepScaler 训练 |
| `configs/eval.yaml` | GSM8K + MATH-500 评测 |

所有路径均为相对于项目根目录的相对路径。模型放在 `models/`，数据放在 `data/`，日志与 checkpoint 输出到 `logs/`。

---

## 致谢

- [GRPO-Zero](https://github.com/policy-gradient/GRPO-Zero) — 本项目的基础实现
- [DeepSeekMath](https://arxiv.org/abs/2402.03300) — GRPO 算法
- [DAPO](https://arxiv.org/abs/2503.14476) — GRPO 算法增强
- [TinyZero](https://github.com/Jiayi-Pan/TinyZero) — Countdown 数据集
- [nano-aha-moment](https://github.com/McGill-NLP/nano-aha-moment) — GRPO 教程实现
- [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) — 基座模型

## License

Apache License 2.0 — 详见 [LICENSE](LICENSE)。
