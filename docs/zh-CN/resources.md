# 资源说明

[English](../resources.md) · [文档目录](README.md)

资源根目录为 `data/`。Git 发布其根目录说明、清单和 `resources-*.tar.gz` 压缩包，解压子目录由 Git 忽略。压缩包合计约 328 MiB，解压后约 1.76 GiB；考虑依赖与工作文件，建议预留至少 8 GiB。

## 恢复与校验

在仓库根目录运行：

```bash
python3 scripts/resource_packages.py verify
python3 scripts/resource_packages.py extract
uv run deephvi verify-resources --resources-dir data
uv run python scripts/verify_resource_layout.py --resources-dir data --config-dir configs
```

前两个命令仅使用 Python 标准库。解压前校验压缩包和成员内容；已有同名文件内容不同时会报冲突，不直接覆盖。请使用新目录，或先移走冲突的本地文件再重试。布局校验还会检查配置副本及保存划分的有序输入。

`manifest.json` 记录路径、用途、大小和 SHA-256；`SHA256SUMS` 覆盖全部资源及 manifest。`PACKAGES.json` 将各包映射到资源成员并绑定 manifest 校验值；`PACKAGES.sha256` 覆盖压缩包与索引。唯一正式清单为 [data/manifest.json](../../data/manifest.json)。

## 内容

下列路径均相对于 `data/`：

| 路径 | 用途 |
| --- | --- |
| `training/` | 预训练 CSV（33,468 行）和分类 CSV（去重前 36,465 行） |
| `auxiliary/` | 部分特征表、有标签配对及候选序列表 |
| `models/esm2/` | ESM-2 权重、配置、tokenizer 和上游模型卡 |
| `models/server_run/classifier.pt` | 已转换的服务器分类器 |
| `models/server_run/pretrain.pt` | 服务器 TMO-Net 预训练权重 |
| `models/server_run/projection.pt` | 配套 320→256 投影 |
| `models/original/` | 供显式转换的原始 checkpoint |
| `cache/seq_emb_cache.pt` | 8,259 条历史缓存，其中 4,797 条可直接用于当前编码长度 |
| `examples/inputs/` | 缓存内、新序列及 CPU 训练样例 |
| `examples/expected_outputs/` | 参考预测输出 |
| `examples/historical/` | 供对照的历史训练 loader 预测 |
| `configs/`、`splits/cpu_smoke/` | 配套配置、固定划分及有序输入配对 |
| `provenance/` | 来源与训练数据统计 |
| `archives/data.7z` | 原始整理数据归档 |

来源及许可见[数据说明](data.md)和[许可说明](licensing.md)。原始分发副本：[百度网盘](https://pan.baidu.com/s/1Ygk06WfLfOzHvYW7N4y6qQ)，提取码 `g9cn`。

## 路径与预测

```bash
uv run deephvi predict --resources-dir data \
  --pairs data/examples/inputs/new_pairs.csv --out runs/predictions.csv --device cpu
```

路径优先级为显式命令行参数、显式配置、资源目录默认值。可用 `--checkpoint`、`--esm-dir`、`--projection`、`--cache` 覆盖；对应配置键为 `predict.checkpoint` 和 `resources.paths.esm/projection/cache`。CLI 自动识别当前目录的 `data/manifest.json`；从其他位置运行时，应传入绝对资源路径或相对于该工作目录的路径。

服务器编码使用 1024 token（含 BOS/EOS），按 attention mask 求均值且保留特殊 token，再使用保存的投影。历史 512-token 缓存中超过 510 个残基的条目首次使用时重算，磁盘原缓存保持不变。长度调整没有重新训练模型，因此长序列预测可能与历史输出不同。

## 公共参考资源

参考序列流程将下载和处理产物保存在 `data/reference/`。训练与推理在此共享原始 FASTA、accession 目录和按编码器区分的索引，已记录的索引应与对应 checkpoint 一起保留。完整库需要在上述模型归档之外安排额外磁盘、内存和编码时间。命令与版本管理见[参考序列流程](reference-sequences.md)。

## 构建压缩包

对已有且通过校验的资源目录运行：

```bash
python3 scripts/resource_packages.py build
```

多数文件按一级目录分组，模型文件按二级目录分组。每组拆成低于 95 MiB 的压缩包；若单个文件压缩后仍超限，需另外处理。同一清单的输出可确定重建。资源变更后，先将旧包集移入被忽略的本地目录，再统一更新 manifest、校验文件及压缩包并验证。

按来源清单收集新资源目录：

```bash
uv run python scripts/build_resource_bundle.py --manifest collection.json --out data
```

收集清单示例：

```json
{
  "version": "DeepHVI_resources_v1",
  "files": [
    {"source": "inputs/classifier.pt", "path": "models/server_run/classifier.pt", "role": "classifier"}
  ]
}
```

`source` 相对于收集清单，`path` 相对于目标目录。收集工具检查输入，拒绝有冲突的目标。收集清单、构建日志和工作副本统一放在被忽略的 `dev-docs/`，公开目录只放审查后的正式发布文件。
