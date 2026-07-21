# Training Methods & Hyperparameters

**English** | [简体中文](training_zh-CN.md)

This document describes the algorithms implemented in GRPO-Series, recommended hyperparameters, and how they map to YAML configs. It is written for educational / single-GPU experiments — not as a paper-faithful reproduction guide.

## 1. Algorithm Family

| Method | Config entry | Core idea | Paper |
|--------|--------------|-----------|-------|
| **GRPO** | `advantage_mode: grpo` | Group-relative advantages; **token-level** importance ratio | [DeepSeekMath](https://arxiv.org/abs/2402.03300) |
| **GSPO** | `advantage_mode: gspo` | Same group advantages; **sequence-level** geometric-mean ratio + clip | [GSPO](https://arxiv.org/abs/2507.18071) |
| **PPO clip** | `use_ppo_clip: true` | Clipped surrogate on the importance ratio | PPO / DAPO-style asymmetric ε |
| **ExGRPO** | `training.exgrpo.enabled` | Replay valuable past trajectories with mixed on/off-policy objective | [ExGRPO](https://arxiv.org/abs/2510.02245) |

### Training loop (shared)

1. Sample questions → rollout \(K\) completions per question  
2. Score with a task reward (format + accuracy, optionally length)  
3. Normalize advantages **within each question group**  
4. Update the policy with GRPO / GSPO / ExGRPO loss  
5. Log metrics; periodically save LoRA checkpoints and run held-out eval  

---

## 2. GRPO vs GSPO

**GRPO** uses a per-token ratio \(w_t = \pi_\theta / \pi_{\theta_{\mathrm{old}}}\) and typically clips in \([1-\varepsilon_L, 1+\varepsilon_H]\) with \(\varepsilon \approx 0.2\).

**GSPO** defines a **sequence** ratio (length-normalized geometric mean):

\[
s_i(\theta)=\exp\Big(\frac{1}{|y_i|}\sum_t\log\frac{\pi_\theta(y_{i,t}\mid\ldots)}{\pi_{\theta_{\mathrm{old}}}(y_{i,t}\mid\ldots)}\Big)
\]

and clips \(s_i\) in ratio space. The GSPO paper uses much smaller ε (≈ **3e-4 / 4e-4**) than token-level GRPO because the ratio is already length-normalized.

| Setting | GRPO (token) | GSPO (sequence) |
|---------|--------------|-----------------|
| `advantage_mode` | `grpo` | `gspo` |
| `gspo_clip_len_scaling` | ignored | `none` (paper GSPO) or `sqrt` (FSPO-style log band) |
| Typical `clip_ratio_*` | `0.2` / `0.28` | `3e-4` / `4e-4` when `none` |

`gspo_clip_len_scaling: sqrt` switches to an FSPO-style **log-space** band ≈ \(c\sqrt{L}\) with \(c\) taken from `clip_ratio_low/high` (often ≈ `0.03`).

---

## 3. ExGRPO (experience replay)

ExGRPO keeps a buffer of past correct / partially correct trajectories and mixes them with fresh on-policy rollouts.

### What this codebase implements

| Component | Behavior | Default |
|-----------|----------|---------|
| Delayed activation | Enable replay only after train-batch Pass@1 ≥ threshold | `activation_threshold: 0.35` |
| Question sampling | Prefer medium difficulty via Gaussian over historical accuracy | `mu: 0.5`, `sigma: 1.0` |
| Mix gate | Only mix a question if historical acc ≥ threshold and a correct traj exists | `mix_acc_threshold: 0.35` |
| Mixed group | 1 replay traj + \((K-1)\) fresh rollouts on the same question | `K: 8` |
| Objective mix | \((1-\rho)\,J_{\mathrm{on}} + \rho\,J_{\mathrm{exp}}\) via episode weights | `rho: 0.5` |
| Replay shaping | \(f(w)=w/(w+\beta)\) on replay importance weights (no PPO clip) | `exgrpo.beta: 0.1` |
| Advantage (when active) | Dr.GRPO-style: **center only**, no std scaling | forced in code |

Entrypoints:

- `train_exgrpo.py` + `configs/train_exgrpo.yaml` — recommended ExGRPO + GSPO  
- `train_exgrpo_math7b.py` + `configs/train_exgrpo_7b.yaml` — Math-7B base prompt  
- `train_r1_thinking.py` + `configs/train_r1_thinking.yaml` — R1 composite reward + optional ExGRPO  

Pure on-policy (no ExGRPO): `train_unified.py`.

---

## 4. Recommended configs (starting points)

### A. ExGRPO + GSPO (3B Instruct) — mainline

Config: `configs/train_exgrpo.yaml`

| Group | Key hyperparameters |
|-------|---------------------|
| Model | Qwen2.5-3B-Instruct, bf16, LoRA r=32 / α=64 |
| Batch | `batch_size=512`, `num_questions_per_batch=64` → **K=8** |
| Gen | `max_gen_len=3072`, `temperature=1.0` |
| Loss | GSPO (`none`), clip **3e-4 / 4e-4**, `ppo_epochs=2`, KL `beta=0` |
| Reward | accuracy 1.0 + format 0.2 (`signed`) |
| ExGRPO | `rho=0.5`, shaping `β=0.1`, activate @ Pass@1≥0.35 |
| Optim | AdamW lr=`1e-5`, wd=`0.1`, grad clip `1.1` |

```bash
uv run python train_exgrpo.py --config configs/train_exgrpo.yaml
```

### B. Math-7B base + ExGRPO

Config: `configs/train_exgrpo_7b.yaml`  
Aligned with Qwen2.5-Math RL notes (lr≈1e-5) and lower micro-batch for VRAM:

| Change vs 3B | Value |
|--------------|-------|
| Model | `models/Qwen2.5-Math-7B` |
| Batch | 256 / 32 questions (K=8) |
| `micro_batch_size` | 2 |
| `memory_efficient_adamw` | true |
| Prompt | native Math `\boxed{}` chat (no R1 think prefix) |

```bash
bash scripts/download_assets.sh model7b
uv run python train_exgrpo_math7b.py --config configs/train_exgrpo_7b.yaml
```

### C. Unified GSPO without ExGRPO

Config: `configs/train_unified.yaml` — FSPO-style `sqrt` scaling, KL `0.02`, larger batch 768.

### D. R1 thinking reward

Config: `configs/train_r1_thinking.yaml` — adds thinking-length shaping (`w_thinking`) and optional ExGRPO; uses **token-level GRPO** clip 0.2 / 0.28 by default.

---

## 5. Hyperparameter cheat sheet

| Key | Meaning | Practical tip |
|-----|---------|---------------|
| `batch_size` / `num_questions_per_batch` | Must satisfy `batch_size = Q × K` | Keep `exgrpo.K` equal to \(K\) |
| `rollout_chunk_size` | Caps parallel rollouts (KV peak) | Lower first if OOM during generate |
| `micro_batch_size` | Update micro-batch | Lower first if OOM during update |
| `ppo_epochs` | Reuse one rollout for multiple updates | Clip only bites when >1 |
| `drop_zero_adv_groups` | Skip all-correct / all-wrong groups | Helps stability; may reduce effective batch |
| `skip_unfinished_episodes` | Drop non-EOS trajectories | Prefer `true` for long generations |
| `learning_rate` | LoRA AdamW step size | 1e-5 is a safe math RL default |
| `temperature` | Rollout exploration | 0.8–1.0 for training; greedy / low-T for eval |

### Memory knobs (24–48 GB)

1. Enable LoRA (`mode: lora`)  
2. Reduce `rollout_chunk_size` and `micro_batch_size`  
3. Shorten `max_gen_len`  
4. Set `memory_efficient_adamw: true` (CPU optimizer states)  
5. Prefer ExGRPO/GSPO 3B config before scaling to 7B  

---

## 6. Rewards & graders

| Pipeline | Reward | Grader |
|----------|--------|--------|
| `train_exgrpo` | accuracy + format | SimKO / legacy math equivalence (`simko_grader`) |
| `train_r1_thinking` | accuracy + format + thinking length | same |
| `train_unified` / Countdown | task-native rewards | Countdown parser / dataset reward |
| Eval scripts | correctness on GSM8K / MATH-500 | boxed / numeric match |

Install grader deps (already in `pyproject.toml`): `sympy`, `regex`, `latex2sympy2`.

---

## 7. Monitoring

Useful TensorBoard signals:

- `reward` / Pass@1 — learning progress  
- `approx_kl` / clip fraction — update aggressiveness  
- `response_len` — length drift  
- ExGRPO: buffer size, n_exp / n_on groups, activation flag  

---

## 8. What this is / is not

**Is:** a readable single-GPU implementation of GRPO-family ideas for learning and hacking.  
**Is not:** a guarantee of paper numbers, multi-node RL infra, or every paper trick (routing replay, full DAPO dynamic sampling, etc.).

For faithful reproduction, use the official repositories of each paper. For concepts and single-GPU iteration, start from the configs above and read `docs/evaluation.md` for scoring.
