# 模型权重与 checkpoint

[English](../checkpoints.md) · [文档目录](README.md)

使用预训练模型前，先运行 `python3 scripts/resource_packages.py extract` 恢复资源。表中路径均相对于 `data/models/`。

| 文件 | 用途 |
| --- | --- |
| `esm2/model.safetensors` | ESM-2 编码器，需同目录配置及 tokenizer |
| `server_run/classifier.pt` | 服务器分类器，结构为 `server_tmonet_v1` |
| `server_run/pretrain.pt` | 服务器 TMO-Net 预训练权重 |
| `server_run/projection.pt` | 配套 320→256 序列投影 |
| `original/server_best_cls_model_fold_3.pt` | 原始服务器分类 checkpoint |
| `original/server_best_pretrain_fold3_epoch9.pt` | 原始服务器预训练 checkpoint |
| `original/initial_release_classifier.pt` | 早期发布的分类 checkpoint |

## 预测与兼容性

```bash
uv run deephvi predict --resources-dir data --pairs input.csv --out predictions.csv --device cpu
```

CLI 按 checkpoint 的 `model_name` 构建模型。服务器预测使用独立保存的投影，新训练的重构模型则包含自己的投影。服务器预训练结构与重构训练结构不同；`train-classifier --pretrain-checkpoint` 需要对应 `train-pretrain` 实现产生的权重。

重构模型 `deephvi`、`deephvi_pretrain` 要求 `model_revision: 2`，使用逆方差高斯融合 `exp(-logvar)`。旧重构 checkpoint 会被拒绝并要求重新训练；仅修改元数据无法修复已学习的权重。历史 `server_tmonet_v1` 使用独立兼容推理图，不受此修订影响。

## 转换原始 checkpoint

原始完整模块 pickle 在加载时可能执行代码，因此转换必须显式启用。仅转换可信来源文件：

```bash
uv run deephvi export-legacy \
  --checkpoint data/models/original/server_best_cls_model_fold_3.pt \
  --projection data/models/server_run/projection.pt \
  --cache data/cache/seq_emb_cache.pt --esm-dir data/models/esm2 \
  --out runs/server-export --allow-unsafe-pickle
```

转换器输出 state dict checkpoint、编码设置、资源指纹和数值对照报告。兼容类随 `deephvi.compat` 安装。早期 checkpoint 不提供 `--projection` 时输出 tensor 清单，不自动假定兼容服务器结构。正常预测使用 `weights_only=True` 加载。

## 新训练与精确续训

训练保存 `best_pretrain.pt` 或 `best.pt`、最新模型 `last.pt`，以及优化器和 RNG 状态 `last_train_state.pt`。元数据包括训练折、精确划分及数据指纹、种子、特征设置、epoch、验证指标与环境。

续训时，在同一目录保留 `last.pt`、匹配的 `last_train_state.pt` 和选中的最佳 checkpoint：

```bash
uv run deephvi train-classifier --config configs/default.yaml \
  --bundle runs/features --splits runs/splits --fold 0 \
  --resume PATH_TO_LAST_PT --epochs 10
```

将 `PATH_TO_LAST_PT` 替换为实际保存路径。`--epochs 10` 表示目标总轮数：第 5 轮后中断的运行继续第 6–10 轮。状态文件通过 epoch 与 SHA-256 绑定模型，并保存最佳分数、早停计数、历史记录、Python/NumPy/CPU/CUDA RNG 和 DataLoader generator 状态。恢复到新运行目录时会复制原最佳 checkpoint，后续较差 epoch 不会替换它。已触发早停的运行返回保存的最佳结果。

旧版不完整状态不能提供精确续训。测试覆盖相同输入、配置和环境下 CPU 续训参数完全一致；跨环境和 CUDA 一致性需另行验证。

## AAindex 参考表恢复

安装包包含 20×12 AAindex 表。可用保存的参考向量验证：

```bash
uv run python scripts/recover_aaindex_table.py \
  --source tests/fixtures/aaindex_legacy_examples.txt \
  --out dev-docs/reports/aaindex.csv --report dev-docs/reports/aaindex-recovery.json
```
