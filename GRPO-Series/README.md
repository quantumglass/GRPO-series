# GRPO-Series

**English** | [简体中文](README_zh-CN.md)

A single-GPU-friendly reinforcement learning codebase for the **GRPO algorithm family**, built on top of [policy-gradient/GRPO-Zero](https://github.com/policy-gradient/GRPO-Zero). It keeps the upstream philosophy of *minimal dependencies* and *from-scratch implementation*, while extending support to math reasoning tasks, LoRA fine-tuning, multiple algorithm variants, and a full evaluation pipeline.

---

## What This Codebase Is

**GRPO-Series** is an **educational and experimental** training framework — not a production RL system and not a paper-level reproduction codebase.

| Property | Description |
|----------|-------------|
| **Purpose** | Help beginners run and tune GRPO-family algorithms on a **single GPU** |
| **Implementation style** | Core model (`qwen2_model.py`), tokenizer, and RL loop are written **from scratch** in pure PyTorch; training does **not** depend on `transformers` or `vLLM` |
| **Algorithm scope** | **Mainline implementations** of GRPO / GSPO / PPO-style clipped updates — enough to build intuition and tuning experience, but **not every trick** from the original papers |
| **Code origin** | Largely AI-assisted development on top of GRPO-Zero; optimized for readability and hackability over SOTA scores |
| **Hardware target** | Single NVIDIA GPU (24–48 GB VRAM); includes memory tricks (LoRA, chunked rollout, optimizer CPU offload) |

### What It Is **Not**

- ❌ A drop-in replacement for official DAPO / DeepSeek-R1 / verl / OpenRLHF codebases
- ❌ A distributed multi-node RL framework
- ❌ A guarantee of exact paper reproduction or leaderboard numbers
- ❌ A heavily engineered production pipeline with exhaustive edge-case handling

If you need **faithful reproduction** of a specific paper, please refer to that method's official repository. GRPO-Series is meant to be a **stepping stone**: understand the loop, try hyperparameters, then graduate to heavier frameworks.

---

## Relationship to GRPO-Zero

This project is a fork of [GRPO-Zero](https://github.com/policy-gradient/GRPO-Zero). We preserve its core design and extend it as follows:

| Dimension | GRPO-Zero (upstream) | GRPO-Series (this repo) |
|-----------|----------------------|-------------------------|
| Tasks | Countdown arithmetic | + DeepScaler, DAPO-Math-17k; GSM8K / MATH-500 eval |
| Training | Full fine-tuning | + LoRA with checkpoint resume |
| Algorithm | Token-level GRPO | + PPO clip, GSPO, multi-epoch updates, KL penalty, zero-advantage group filtering |
| Reward | format + answer | + R1-style composite reward (accuracy + format + length) |
| Memory | MemoryEfficientAdamW | + chunked rollout (`grpo_efficient.py`) |
| Evaluation | In-training eval only | Standalone eval scripts + pass@k |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  Entry scripts                                              │
│  train_unified.py  train_r1_thinking.py  train.py           │
│  evaluate_models.py  evaluate_models_readme.py                │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  Algorithm layer                                            │
│  grpo.py (rollout + update_policy)                          │
│  grpo_efficient.py (chunked rollout for memory)             │
│  sampling.py  data_types.py                                 │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  Model layer (from scratch)                                 │
│  qwen2_model.py  tokenizer.py  lora.py  optimizer.py        │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  Task layer (dataset + reward)                              │
│  countdown_task.py  deepscaler_task.py  dapo_math_task.py    │
│  benchmark_task.py  r1_thinking_reward.py                     │
└─────────────────────────────────────────────────────────────┘
```

**Training loop (simplified):**

1. Sample a batch of questions → generate M answers each via autoregressive rollout
2. Compute rewards (format + correctness, or R1 composite)
3. Normalize advantages within each question group: \(A = (r - \mu) / \sigma\)
4. Update policy with clipped surrogate loss (GRPO token-level or GSPO response-level)
5. Log to TensorBoard; save LoRA checkpoints

---

## Supported Algorithms

| Mode | Config key | Description |
|------|------------|-------------|
| **GRPO** | `advantage_mode: grpo` | Token-level policy gradient with group-relative advantages |
| **GSPO** | `advantage_mode: gspo` | Response-level importance ratio; optional length-scaled clipping |
| **PPO clip** | `use_ppo_clip: true` | Clipped surrogate on importance ratio; effective when `ppo_epochs > 1` |
| **KL penalty** | `beta > 0` | Adds KL term against reference policy |
| **Zero-adv filter** | `drop_zero_adv_groups: true` | Skips groups where all rewards are identical (no gradient signal) |

---

## Project Structure

```
GRPO-Series/
├── train_unified.py           # ★ Main entry: multi-dataset + LoRA + GSPO/PPO
├── train_r1_thinking.py       # R1-style long-thinking training
├── train.py                   # Upstream-compatible Countdown task
├── train_deepscaler_legacy.py # Early simplified DeepScaler trainer
├── evaluate_models.py         # Benchmark eval (native model stack)
├── evaluate_models_readme.py  # Baseline eval (transformers API)
├── grpo.py                    # Core GRPO/GSPO/PPO logic
├── grpo_efficient.py          # Chunked rollout (lower peak memory)
├── qwen2_model.py             # Qwen2 Transformer from scratch
├── tokenizer.py               # Tokenizer + chat template
├── lora.py / checkpoint.py    # LoRA injection & resume
├── *_task.py                  # Per-task datasets & reward functions
├── configs/                   # Training & eval YAML configs
├── scripts/download_assets.sh # One-click model & data download
└── docs/evaluation.md         # Eval self-check guide
```

---

## Quick Start

### Install

```bash
pip install uv
uv sync
```

Requires Python >= 3.11 and an NVIDIA GPU (32 GB+ recommended for math LoRA training).

### Download assets

```bash
bash scripts/download_assets.sh all
```

### Train (recommended)

```bash
uv run python train_unified.py --config configs/train_unified.yaml
```

Switch dataset via `dataset.name` in config: `countdown` | `dapo_math_17k` | `deepscaler`.

### R1-style thinking

```bash
uv run python train_r1_thinking.py --config configs/train_r1_thinking.yaml
```

### Countdown (upstream-compatible)

```bash
uv run python train.py --config configs/train_countdown.yaml          # 48 GB
uv run python train.py --config configs/train_countdown_24gb.yaml     # 24 GB
```

### Resume LoRA training

```bash
uv run python train_unified.py \
  --config configs/train_unified.yaml \
  --resume_lora_ckpt logs/<run_dir>/ckpt_000100.pt
```

### Evaluate

```bash
uv run python evaluate_models.py --config configs/eval.yaml
uv run python evaluate_models_readme.py --config configs/eval.yaml --types base
```

See [docs/evaluation.md](docs/evaluation.md) ([中文](docs/evaluation_zh-CN.md)) for the full eval checklist.

---

## Configuration Reference

| Config file | Purpose |
|-------------|---------|
| `configs/train_unified.yaml` | Main training: DeepScaler + LoRA + GSPO |
| `configs/train_r1_thinking.yaml` | R1-style thinking training |
| `configs/train_countdown.yaml` | Countdown full fine-tuning |
| `configs/train_countdown_24gb.yaml` | Countdown low-VRAM config |
| `configs/train_deepscaler_legacy.yaml` | Legacy simplified DeepScaler trainer |
| `configs/eval.yaml` | GSM8K + MATH-500 evaluation |

All paths are relative to the project root. Models go in `models/`, data in `data/`, logs and checkpoints in `logs/`.

---

## Acknowledgements

- [GRPO-Zero](https://github.com/policy-gradient/GRPO-Zero) — foundation of this project
- [DeepSeekMath](https://arxiv.org/abs/2402.03300) — GRPO algorithm
- [DAPO](https://arxiv.org/abs/2503.14476) — GRPO enhancements
- [TinyZero](https://github.com/Jiayi-Pan/TinyZero) — Countdown dataset
- [nano-aha-moment](https://github.com/McGill-NLP/nano-aha-moment) — GRPO tutorial
- [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) — base model

## License

Apache License 2.0 — see [LICENSE](LICENSE).
