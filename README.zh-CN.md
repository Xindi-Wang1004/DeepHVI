# DeepHVI

[English](README.md) · [简体中文](README.zh-CN.md)

DeepHVI 结合 ESM-2 序列嵌入与 AAindex 理化性质描述符，预测人类—病毒蛋白相互作用。本仓库提供预训练模型预测、特征处理、训练、逐折评估及蛋白相似性检索。

**DeepHVI Web 版已上线：** 访问 [deephvi.omicbox.com](https://deephvi.omicbox.com)，无需本地部署，即可在浏览器中预测人类—病毒蛋白相互作用。

论文：*DeepHVI: A multimodal deep learning framework for predicting human-virus protein-protein interactions using protein language models.* **Biosafety and Health**, 2025, 7:257–266。

## 安装

需要 Python 3.10 或更新版本。在终端运行：

```bash
git clone https://github.com/Xindi-Wang1004/DeepHVI.git
cd DeepHVI
python3 -m pip install uv
uv sync --frozen --extra dev --extra data
```

锁文件默认安装 CPU 版 PyTorch。CUDA 环境见[使用说明](docs/zh-CN/usage.md#cuda)。也可在虚拟环境中执行 `python -m pip install '.[data]'`，随后将命令中的 `uv run deephvi` 替换为 `deephvi`。

## CPU 快速体验

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run deephvi demo \
  --backend hash --n-pairs 30 --epochs 1 --device cpu --out runs/demo
```

此示例使用合成配对和确定性的 hash 编码器，在 CPU 上运行特征处理、训练、评估和检索。

## 恢复数据与预训练模型

资源随仓库以 `data/resources-*.tar.gz` 发布。克隆后，在仓库根目录运行：

```bash
python3 scripts/resource_packages.py verify
python3 scripts/resource_packages.py extract
uv run deephvi verify-resources --resources-dir data
```

压缩包合计约 328 MiB，解压后约 1.76 GiB；考虑依赖、Git 历史和输出，建议预留至少 8 GiB 空间。文件用途及校验方法见[资源说明](docs/zh-CN/resources.md)，完整参考库需要额外存储空间。

## 预测相互作用

输入 CSV 包含 `human_seq,viral_seq`，可选 `pair_id`；也支持 `S_H,S_V` 和 `seq_a,seq_b`。最小示例：

```csv
pair_id,human_seq,viral_seq
example_1,ACDEFGHIKLMNPQRSTVWY,MKTIIALSYIFCLVFAD
```

使用随附样例运行：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run deephvi predict \
  --resources-dir data --pairs data/examples/inputs/new_pairs.csv \
  --out runs/predictions.csv --device cpu
```

输出 CSV 包含样本 ID、序列、互作概率 `probability` 和预测标签 `prediction`，默认阈值为 0.5。结果按输入顺序排列，保留重复行。模型会自动编码新序列，并复用已有的兼容特征缓存。

随附编码器最多使用 1024 token，包含 BOS/EOS。池化、投影和缓存设置见[编码与复现约定](docs/zh-CN/reproducibility.md)。

## 训练与评估

```bash
uv run deephvi reproduce --resources-dir data --config configs/default.yaml \
  --pairs data/training/SH_SV_feature_with_vectors_0.5.csv \
  --reference-source both --reference-dir data/reference --out runs/experiment
```

该命令准备配对特征与数据划分、逐折训练，并评估选中的 checkpoint。默认 `paper` 协议保留 20% 测试集，在训练部分进行三折验证。按蛋白连接组分离数据的实验可使用 `group` 协议。

`--reference-source both` 同时准备人类与病毒参考库，供后续检索使用。来源版本和索引位置保存在 `references.json` 及各 checkpoint 中。只训练时可省略参考库参数；分阶段的 `train-pretrain` 和 `train-classifier` 支持相同设置。

分阶段训练与 CPU 示例见[使用说明](docs/zh-CN/usage.md)，编码与训练设置见[复现约定](docs/zh-CN/reproducibility.md)。

## 生成与检索

`retrieve` 和 `generate` 返回天然蛋白候选序列，按序列 embedding 的余弦相似度排序。完成上述训练后，可直接复用已记录的参考索引：

```bash
uv run deephvi generate --checkpoint runs/experiment/fold0/checkpoints/best.pt \
  --pairs data/examples/inputs/new_pairs.csv --target viral --out runs/candidates.json
```

checkpoint 提供编码设置和索引记录，`--target human` 切换到人类侧。直接从公共数据开始时，选择 `--reference-source human` 或 `--reference-source viral`，命令会自动下载、处理并建索引。也可用 `--reference` 提供 FASTA，或用 `--index` 指定已有索引。

人类参考库来自 NCBI RefSeq `GCF_000001405.40` 完整蛋白 FASTA，病毒库包含全部 UniProtKB `taxonomy_id:10239` 记录。两侧各自合并相同的完整序列，保留全部原始 accession。

运行命令见[生成与检索说明](docs/zh-CN/usage.md#生成与检索)。

## 文档

| 主题 | English | 简体中文 |
| --- | --- | --- |
| 安装与命令行流程 | [Usage](docs/usage.md) | [使用说明](docs/zh-CN/usage.md) |
| 参考库下载、训练记录与推理复用 | [Reference sequences](docs/reference-sequences.md) | [参考序列流程](docs/zh-CN/reference-sequences.md) |
| 数据来源、格式与划分 | [Data](docs/data.md) | [数据说明](docs/zh-CN/data.md) |
| 资源压缩包与校验 | [Resources](docs/resources.md) | [资源说明](docs/zh-CN/resources.md) |
| 模型与 checkpoint 兼容性 | [Checkpoints](docs/checkpoints.md) | [模型权重](docs/zh-CN/checkpoints.md) |
| 编码与实验设置 | [Reproducibility](docs/reproducibility.md) | [复现约定](docs/zh-CN/reproducibility.md) |
| 仓库结构与贡献流程 | [Repository layout](docs/repository-layout.md) | [仓库布局](docs/zh-CN/repository-layout.md) |
| 验证命令与范围 | [Validation](docs/validation.md) | [验证方法](docs/zh-CN/validation.md) |
| 许可与署名 | [Licensing](docs/licensing.md) | [许可说明](docs/zh-CN/licensing.md) |

## 许可、引用与反馈

自有代码和作者持有权利的 DeepHVI 权重采用 [MIT](LICENSE)。其他来源的模型与数据遵循[第三方声明](THIRD_PARTY_NOTICES.zh-CN.md)和[许可说明](docs/zh-CN/licensing.md)中的条款。相互作用记录来自 [HVIDB](https://zzdlab.com/hvidb/)，序列来源包括 [UniProt](https://www.uniprot.org/) 和 NCBI RefSeq。

使用实现时请引用上述 DeepHVI 论文；使用数据时同时引用[数据来源](docs/zh-CN/data.md)。反馈问题请提交 [GitHub Issue](https://github.com/Xindi-Wang1004/DeepHVI/issues)，附上运行命令、错误信息及相关 `deephvi doctor` 输出。
