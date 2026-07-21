# GRPO-Series

[English](README.md) | **简体中文**

面向 **GRPO 算法族** 的单卡友好强化学习代码库，基于 [policy-gradient/GRPO-Zero](https://github.com/policy-gradient/GRPO-Zero)。保留上游「从零实现、低依赖」设计，并扩展数学推理任务、LoRA、GSPO、**ExGRPO** 经验回放与独立评测管线。

> **v0.3** 同步最新 GRPO-Zero 实验栈（ExGRPO + GSPO + SimKO 判题）。方法与超参见 [docs/training_zh-CN.md](docs/training_zh-CN.md)。

---

## 代码性质

**GRPO-Series** 面向 **学习与实验** —— 不是生产级 RL，也不是论文级精确复现。

| 属性 | 说明 |
|------|------|
| **定位** | 在 **单卡 GPU** 上跑通并调参 GRPO 系列算法 |
| **实现** | 核心模型 / tokenizer / RL 循环为纯 PyTorch；训练 **不依赖** `vLLM` |
| **算法** | GRPO / GSPO / PPO 裁剪 / **ExGRPO** — 主干实现，非论文全部 trick |
| **硬件** | 单卡 24–48 GB；LoRA、分块 rollout、优化器 CPU offload |

### 它 **不是**

- ❌ verl / OpenRLHF / 官方 DAPO·ExGRPO 仓库的平替  
- ❌ 多机分布式 RL  
- ❌ 论文或榜单成绩保证  

---

## 与 GRPO-Zero 的关系

| 维度 | GRPO-Zero | GRPO-Series |
|------|-----------|-------------|
| 任务 | Countdown | + DeepScaler、DAPO、competition_math；GSM8K / MATH-500 评测 |
| 训练 | 全参微调 | + LoRA 续训 |
| 算法 | Token GRPO | + PPO 裁剪、GSPO、ExGRPO、零优势过滤 |
| 奖励 | format + answer | + R1 复合奖励；SimKO 判题 |
| 显存 | MemoryEfficientAdamW | + 分块 rollout |
| 评测 | 训练内 | 独立评测 + pass@k |

---

## 快速开始

```bash
pip install uv
uv sync
bash scripts/download_assets.sh all
```

### 推荐训练入口

```bash
uv run python train_exgrpo.py --config configs/train_exgrpo.yaml
```

### 其他入口

```bash
uv run python train_unified.py --config configs/train_unified.yaml
uv run python train_r1_thinking.py --config configs/train_r1_thinking.yaml

bash scripts/download_assets.sh model7b
uv run python train_exgrpo_math7b.py --config configs/train_exgrpo_7b.yaml

uv run python train.py --config configs/train_countdown.yaml
```

### 续训 / 评测

```bash
uv run python train_exgrpo.py \
  --config configs/train_exgrpo.yaml \
  --resume_lora_ckpt logs/<run_dir>/ckpt_000100.pt

uv run python evaluate_models.py --config configs/eval.yaml
```

路径相对项目根：`models/`、`data/`、`logs/`。

---

## 配置一览

| 配置 | 用途 |
|------|------|
| `configs/train_exgrpo.yaml` | **主线** ExGRPO + GSPO（3B） |
| `configs/train_exgrpo_7b.yaml` | ExGRPO Math-7B base |
| `configs/train_r1_thinking.yaml` | R1 奖励 + ExGRPO |
| `configs/train_unified.yaml` | 纯 on-policy GSPO |
| `configs/train_countdown*.yaml` | Countdown 全参 |
| `configs/eval.yaml` | GSM8K + MATH-500 |

详情：[docs/training_zh-CN.md](docs/training_zh-CN.md) · [docs/evaluation_zh-CN.md](docs/evaluation_zh-CN.md)

---

## 单元测试

```bash
uv run python -m unittest test_exgrpo test_eval_metrics test_simko_grader test_competition_math -v
```

---

## 致谢

- [GRPO-Zero](https://github.com/policy-gradient/GRPO-Zero)  
- [DeepSeekMath](https://arxiv.org/abs/2402.03300) · [GSPO](https://arxiv.org/abs/2507.18071) · [ExGRPO](https://arxiv.org/abs/2510.02245) · [DAPO](https://arxiv.org/abs/2503.14476)  
- [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) / [Qwen2.5-Math](https://huggingface.co/Qwen/Qwen2.5-Math-7B)

## License

Apache License 2.0 — 见 [LICENSE](LICENSE)。
