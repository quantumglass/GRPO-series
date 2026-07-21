# GRPO-series

Open-source mirror of **[GRPO-Series](./GRPO-Series/)**.

All code, configs, and docs live in the nested package directory:

```bash
cd GRPO-Series
uv sync
bash scripts/download_assets.sh all
uv run python train_exgrpo.py --config configs/train_exgrpo.yaml
```

- Methods & hyperparameters: [GRPO-Series/docs/training.md](GRPO-Series/docs/training.md) · [中文](GRPO-Series/docs/training_zh-CN.md)
- Full README: [GRPO-Series/README.md](GRPO-Series/README.md) · [中文](GRPO-Series/README_zh-CN.md)

Apache License 2.0 — see [LICENSE](LICENSE) and [GRPO-Series/LICENSE](GRPO-Series/LICENSE).
