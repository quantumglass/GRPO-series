# GRPO-series

开源仓库 **[GRPO-Series](./GRPO-Series/)** 的入口说明。

代码、配置与文档均在子目录中：

```bash
cd GRPO-Series
uv sync
bash scripts/download_assets.sh all
uv run python train_exgrpo.py --config configs/train_exgrpo.yaml
```

- 训练方法与超参：[GRPO-Series/docs/training_zh-CN.md](GRPO-Series/docs/training_zh-CN.md)
- 完整说明：[GRPO-Series/README_zh-CN.md](GRPO-Series/README_zh-CN.md)

Apache License 2.0 — 见 [LICENSE](LICENSE)。
