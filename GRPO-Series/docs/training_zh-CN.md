# 训练方法与超参数

[English](training.md) | **简体中文**

本文说明 GRPO-Series 中实现的算法、推荐超参，以及它们与 YAML 配置的对应关系。面向学习与单卡实验，**不是**论文级精确复现手册。

## 1. 算法族一览

| 方法 | 配置项 | 核心思想 | 论文 |
|------|--------|----------|------|
| **GRPO** | `advantage_mode: grpo` | 组内相对优势；**token 级** importance ratio | [DeepSeekMath](https://arxiv.org/abs/2402.03300) |
| **GSPO** | `advantage_mode: gspo` | 同样组内优势；**序列级**几何平均 ratio + 裁剪 | [GSPO](https://arxiv.org/abs/2507.18071) |
| **PPO 裁剪** | `use_ppo_clip: true` | 对 importance ratio 做裁剪代理目标 | PPO / DAPO 非对称 ε |
| **ExGRPO** | `training.exgrpo.enabled` | 回放高价值历史轨迹，混合 on/off-policy 目标 | [ExGRPO](https://arxiv.org/abs/2510.02245) |

### 共用训练循环

1. 采样题目 → 每题 rollout \(K\) 条回答  
2. 任务奖励打分（格式 + 正确性，可选长度）  
3. **按题目组内**归一化优势  
4. 用 GRPO / GSPO / ExGRPO 损失更新策略  
5. 写日志；定期存 LoRA ckpt、在 test 集评测  

---

## 2. GRPO 与 GSPO

**GRPO** 使用 token 级比率 \(w_t=\pi_\theta/\pi_{\theta_{\mathrm{old}}}\)，裁剪区间通常为 \([1-\varepsilon_L,1+\varepsilon_H]\)，\(\varepsilon\approx0.2\)。

**GSPO** 定义**序列级**比率（长度归一化的几何平均）：

\[
s_i(\theta)=\exp\Big(\frac{1}{|y_i|}\sum_t\log\frac{\pi_\theta(y_{i,t}\mid\ldots)}{\pi_{\theta_{\mathrm{old}}}(y_{i,t}\mid\ldots)}\Big)
\]

并在 ratio 空间裁剪。因已做长度归一化，GSPO 论文中的 ε 远小于 token 级 GRPO（约 **3e-4 / 4e-4**）。

| 设置 | GRPO（token） | GSPO（序列） |
|------|---------------|--------------|
| `advantage_mode` | `grpo` | `gspo` |
| `gspo_clip_len_scaling` | 忽略 | `none`（论文 GSPO）或 `sqrt`（FSPO 风格） |
| 典型 `clip_ratio_*` | `0.2` / `0.28` | `none` 时用 `3e-4` / `4e-4` |

`gspo_clip_len_scaling: sqrt` 时改为 FSPO 风格的 **log 空间**带宽 ≈ \(c\sqrt{L}\)，\(c\) 取自 `clip_ratio_low/high`（常用约 `0.03`）。

---

## 3. ExGRPO（经验回放）

ExGRPO 维护历史正确 / 部分正确轨迹的 buffer，并与当前 on-policy rollout 混合。

### 本仓库实现要点

| 组件 | 行为 | 默认 |
|------|------|------|
| 延迟激活 | 训练 batch Pass@1 ≥ 阈值后才启用 replay | `activation_threshold: 0.35` |
| 题目采样 | 按历史 acc 的高斯偏好中等难度 | `mu: 0.5`, `sigma: 1.0` |
| 混合门控 | 仅当历史 acc≥阈值且存在正解轨迹时混入 | `mix_acc_threshold: 0.35` |
| 混合组 | 1 条 replay + \((K-1)\) 条同题 fresh | `K: 8` |
| 目标混合 | 通过 episode 权重实现 \((1-\rho)J_{\mathrm{on}}+\rho J_{\mathrm{exp}}\) | `rho: 0.5` |
| Replay shaping | \(f(w)=w/(w+\beta)\)，replay 不做 PPO clip | `exgrpo.beta: 0.1` |
| Advantage | 激活后强制 Dr.GRPO：**只减均值、不除 std** | 代码内强制 |

入口：

- `train_exgrpo.py` + `configs/train_exgrpo.yaml` — 推荐 ExGRPO + GSPO  
- `train_exgrpo_math7b.py` + `configs/train_exgrpo_7b.yaml` — Math-7B base  
- `train_r1_thinking.py` + `configs/train_r1_thinking.yaml` — R1 复合奖励 + 可选 ExGRPO  

纯 on-policy（无 ExGRPO）：`train_unified.py`。

---

## 4. 推荐配置（起步）

### A. ExGRPO + GSPO（3B Instruct）— 主线

配置：`configs/train_exgrpo.yaml`

| 分组 | 关键超参 |
|------|----------|
| 模型 | Qwen2.5-3B-Instruct，bf16，LoRA r=32 / α=64 |
| Batch | `batch_size=512`，`num_questions_per_batch=64` → **K=8** |
| 生成 | `max_gen_len=3072`，`temperature=1.0` |
| 损失 | GSPO（`none`），clip **3e-4 / 4e-4**，`ppo_epochs=2`，KL=`0` |
| 奖励 | accuracy 1.0 + format 0.2（`signed`） |
| ExGRPO | `rho=0.5`，shaping `β=0.1`，Pass@1≥0.35 激活 |
| 优化 | AdamW lr=`1e-5`，wd=`0.1`，grad clip `1.1` |

```bash
uv run python train_exgrpo.py --config configs/train_exgrpo.yaml
```

### B. Math-7B base + ExGRPO

配置：`configs/train_exgrpo_7b.yaml`  
参考 Qwen2.5-Math RL（lr≈1e-5），并降低 micro-batch 以适配显存：

| 相对 3B 的调整 | 取值 |
|----------------|------|
| 模型 | `models/Qwen2.5-Math-7B` |
| Batch | 256 / 32 题（K=8） |
| `micro_batch_size` | 2 |
| `memory_efficient_adamw` | true |
| Prompt | Math 原生 `\boxed{}`（无 R1 think 预填） |

```bash
bash scripts/download_assets.sh model7b
uv run python train_exgrpo_math7b.py --config configs/train_exgrpo_7b.yaml
```

### C. 无 ExGRPO 的统一 GSPO

配置：`configs/train_unified.yaml` — FSPO 风格 `sqrt`，KL=`0.02`，batch=768。

### D. R1 思考奖励

配置：`configs/train_r1_thinking.yaml` — 增加 thinking 长度塑形（`w_thinking`），可选 ExGRPO；默认 **token 级 GRPO** clip 0.2 / 0.28。

---

## 5. 超参速查

| 键 | 含义 | 实践建议 |
|----|------|----------|
| `batch_size` / `num_questions_per_batch` | 须满足 `batch_size = Q × K` | `exgrpo.K` 必须等于 \(K\) |
| `rollout_chunk_size` | 限制并行 rollout（降 KV 峰值） | generate OOM 时优先下调 |
| `micro_batch_size` | update 阶段 micro-batch | update OOM 时优先下调 |
| `ppo_epochs` | 同一份 rollout 多轮更新 | >1 时 clip 才真正生效 |
| `drop_zero_adv_groups` | 跳过全对/全错组 | 有助稳定，但会减小有效 batch |
| `skip_unfinished_episodes` | 丢弃未生成 EOS 的轨迹 | 长生成建议 `true` |
| `learning_rate` | LoRA AdamW | 数学 RL 常用 `1e-5` |
| `temperature` | rollout 探索 | 训练 0.8–1.0；评测用 greedy / 低温 |

### 显存旋钮（24–48 GB）

1. 使用 LoRA（`mode: lora`）  
2. 降低 `rollout_chunk_size` 与 `micro_batch_size`  
3. 缩短 `max_gen_len`  
4. `memory_efficient_adamw: true`（优化器状态放 CPU）  
5. 先跑通 3B ExGRPO/GSPO，再上 7B  

---

## 6. 奖励与判题

| 管线 | 奖励 | 判题 |
|------|------|------|
| `train_exgrpo` | accuracy + format | SimKO / legacy（`simko_grader`） |
| `train_r1_thinking` | accuracy + format + thinking 长度 | 同上 |
| `train_unified` / Countdown | 任务原生奖励 | Countdown 解析器等 |
| 评测脚本 | GSM8K / MATH-500 正确性 | boxed / 数值匹配 |

判题依赖（已写入 `pyproject.toml`）：`sympy`、`regex`、`latex2sympy2`。

---

## 7. 监控

常用 TensorBoard 指标：

- `reward` / Pass@1 — 学习进度  
- `approx_kl` / clip fraction — 更新幅度  
- `response_len` — 长度漂移  
- ExGRPO：buffer 规模、n_exp / n_on、是否已激活  

---

## 8. 定位说明

**是：** 可读的单卡 GRPO 族实现，便于理解与改动。  
**不是：** 论文数字保证、多机 RL 基础设施，或论文全部 trick（如 Routing Replay、完整 DAPO 动态采样等）。

若需忠实复现，请使用各论文官方仓库；若以理解与单卡迭代为目标，请从上述配置起步，评测细节见 `docs/evaluation_zh-CN.md`。
