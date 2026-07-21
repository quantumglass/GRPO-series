# GRPO-Series

**English** | [简体中文](README_zh-CN.md)

A single-GPU-friendly reinforcement learning codebase for the **GRPO algorithm family**, built on [policy-gradient/GRPO-Zero](https://github.com/policy-gradient/GRPO-Zero). It keeps the upstream *from-scratch / low-dependency* design, and adds math reasoning tasks, LoRA, GSPO, **ExGRPO** experience replay, and a standalone evaluation pipeline.

> **v0.3** syncs the latest GRPO-Zero experiment stack (ExGRPO + GSPO + SimKO grader). See [docs/training.md](docs/training.md) for methods and hyperparameters.

---

## What This Codebase Is

**GRPO-Series** is an **educational and experimental** framework — not a production RL system and not a paper-faithful reproduction codebase.

| Property | Description |
|----------|-------------|
| **Purpose** | Run and tune GRPO-family algorithms on a **single GPU** |
| **Implementation** | Core model (`qwen2_model.py`), tokenizer, and RL loop are pure PyTorch; training does **not** require `vLLM` |
| **Algorithms** | GRPO / GSPO / PPO clip / **ExGRPO** — mainline ideas, not every paper trick |
| **Hardware** | Single NVIDIA GPU (24–48 GB); LoRA, chunked rollout, optimizer CPU offload |

### What It Is **Not**

- ❌ A drop-in replacement for verl / OpenRLHF / official DAPO or ExGRPO repos  
- ❌ Multi-node distributed RL  
- ❌ A guarantee of paper or leaderboard numbers  

---

## Relationship to GRPO-Zero

| Dimension | GRPO-Zero (upstream) | GRPO-Series (this repo) |
|-----------|----------------------|-------------------------|
| Tasks | Countdown | + DeepScaler, DAPO-Math-17k, competition_math; GSM8K / MATH-500 eval |
| Training | Full FT | + LoRA resume |
| Algorithm | Token GRPO | + PPO clip, GSPO, ExGRPO replay, zero-adv filtering |
| Reward | format + answer | + R1 composite; SimKO math grader |
| Memory | MemoryEfficientAdamW | + chunked rollout |
| Evaluation | In-training only | Standalone eval + pass@k |

---

## Project Structure

```
GRPO-Series/
├── train_exgrpo.py            # ★ Main: ExGRPO + GSPO
├── train_exgrpo_math7b.py     # Math-7B base + ExGRPO
├── train_r1_thinking.py       # R1 thinking reward (+ optional ExGRPO)
├── train_unified.py           # On-policy GSPO/PPO (no ExGRPO)
├── train.py                   # Countdown (upstream-compatible)
├── evaluate_models*.py        # Benchmark evaluation
├── exgrpo.py / grpo.py        # Algorithm core
├── simko_grader/              # Math equivalence grader
├── configs/                   # Relative-path YAML configs
├── scripts/download_assets.sh
└── docs/training.md           # Methods & hyperparameters
```

---

## Quick Start

```bash
pip install uv
uv sync
bash scripts/download_assets.sh all
```

### Train (recommended)

```bash
uv run python train_exgrpo.py --config configs/train_exgrpo.yaml
```

### Other entrances

```bash
# On-policy GSPO
uv run python train_unified.py --config configs/train_unified.yaml

# R1 thinking + ExGRPO
uv run python train_r1_thinking.py --config configs/train_r1_thinking.yaml

# Math-7B base
bash scripts/download_assets.sh model7b
uv run python train_exgrpo_math7b.py --config configs/train_exgrpo_7b.yaml

# Countdown
uv run python train.py --config configs/train_countdown.yaml
```

### Resume / evaluate

```bash
uv run python train_exgrpo.py \
  --config configs/train_exgrpo.yaml \
  --resume_lora_ckpt logs/<run_dir>/ckpt_000100.pt

uv run python evaluate_models.py --config configs/eval.yaml
```

Paths are relative to the project root: models in `models/`, data in `data/`, logs in `logs/`.

---

## Configuration Reference

| Config | Purpose |
|--------|---------|
| `configs/train_exgrpo.yaml` | **Main** ExGRPO + GSPO (3B) |
| `configs/train_exgrpo_7b.yaml` | ExGRPO Math-7B base |
| `configs/train_r1_thinking.yaml` | R1 reward + ExGRPO |
| `configs/train_unified.yaml` | On-policy GSPO |
| `configs/train_countdown*.yaml` | Countdown full FT |
| `configs/eval.yaml` | GSM8K + MATH-500 |

Details: [docs/training.md](docs/training.md) · [docs/evaluation.md](docs/evaluation.md)

---

## Unit tests

```bash
uv run python -m unittest test_exgrpo test_eval_metrics test_simko_grader test_competition_math -v
```

---

## Acknowledgements

- [GRPO-Zero](https://github.com/policy-gradient/GRPO-Zero) — foundation  
- [DeepSeekMath](https://arxiv.org/abs/2402.03300) — GRPO  
- [GSPO](https://arxiv.org/abs/2507.18071) — sequence-level optimization  
- [ExGRPO](https://arxiv.org/abs/2510.02245) — experiential replay  
- [DAPO](https://arxiv.org/abs/2503.14476) — GRPO enhancements  
- [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) / [Qwen2.5-Math](https://huggingface.co/Qwen/Qwen2.5-Math-7B)

## License

Apache License 2.0 — see [LICENSE](LICENSE).
