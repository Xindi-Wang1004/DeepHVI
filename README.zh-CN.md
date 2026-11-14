# DeepHVI

[English](README.md) · [简体中文](README.zh-CN.md)

DeepHVI 结合 ESM-2 序列嵌入与 AAindex 理化性质描述符，预测人类—病毒蛋白相互作用。本仓库提供预训练模型预测、特征处理、训练、逐折评估及蛋白相似性检索。

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

## 无需模型资源的快速体验

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run deephvi demo \
  --backend hash --n-pairs 30 --epochs 1 --device cpu --out runs/demo
```

此命令使用合成配对，运行特征处理、训练、评估和检索。hash 编码器仅用于验证流程，其指标不代表生物学预测性能。

## 恢复数据与预训练模型

资源随仓库以 `data/resources-*.tar.gz` 发布。克隆后，在仓库根目录运行：

```bash
python3 scripts/resource_packages.py verify
python3 scripts/resource_packages.py extract
uv run deephvi verify-resources --resources-dir data
```

解压工具仅依赖 Python 标准库，会校验压缩包及解压内容。压缩包合计约 328 MiB，解压后约 1.76 GiB；考虑依赖、Git 历史和输出，建议预留至少 8 GiB 空间。文件用途及校验方法见[资源说明](docs/zh-CN/resources.md)。

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

输出包含样本 ID、序列、正类概率 `probability` 和标签 `prediction`，默认阈值为 0.5。输入顺序与重复行会保留。服务器模型使用配套 ESM-2、保存的 320→256 投影和兼容的序列缓存，新序列自动编码。

编码上限为 1024 token，包含 BOS/EOS。历史缓存中按 512 token 截断的序列会按需重算。与历史结果对照前，请阅读[编码与复现约定](docs/zh-CN/reproducibility.md)。

## 训练与评估

```bash
uv run deephvi reproduce --resources-dir data --config configs/default.yaml \
  --pairs data/training/SH_SV_feature_with_vectors_0.5.csv --out runs/experiment
```

该命令依次检查数据、编码特征、保存划分、逐折训练，并评估各折选中的 checkpoint。现代协议先按完整序列配对去重，标签冲突时报错。默认 `paper` 协议保留 20% 测试集，对训练部分执行三折验证；配对层面的划分仍可能共享蛋白，跨蛋白泛化实验应使用 `group` 协议。

分阶段训练与小型 CPU 示例见[使用说明](docs/zh-CN/usage.md)。实现中的重构假设已在文档中说明，小规模流程通过不代表论文完整指标已复现。

## 文档

| 主题 | English | 简体中文 |
| --- | --- | --- |
| 安装与命令行流程 | [Usage](docs/usage.md) | [使用说明](docs/zh-CN/usage.md) |
| 数据来源、格式与划分 | [Data](docs/data.md) | [数据说明](docs/zh-CN/data.md) |
| 资源压缩包与校验 | [Resources](docs/resources.md) | [资源说明](docs/zh-CN/resources.md) |
| 模型与 checkpoint 兼容性 | [Checkpoints](docs/checkpoints.md) | [模型权重](docs/zh-CN/checkpoints.md) |
| 编码与实验假设 | [Reproducibility](docs/reproducibility.md) | [复现约定](docs/zh-CN/reproducibility.md) |
| 仓库结构与贡献流程 | [Repository layout](docs/repository-layout.md) | [仓库布局](docs/zh-CN/repository-layout.md) |
| 验证命令与范围 | [Validation](docs/validation.md) | [验证方法](docs/zh-CN/validation.md) |
| 许可与署名 | [Licensing](docs/licensing.md) | [许可说明](docs/zh-CN/licensing.md) |

## 许可、引用与反馈

自有代码和作者持有权利的 DeepHVI 权重采用 [MIT](LICENSE)。第三方模型与数据库内容遵循上游条款，详见[第三方声明](THIRD_PARTY_NOTICES.zh-CN.md)及[许可范围](docs/zh-CN/licensing.md)。相互作用记录来自 [HVIDB](https://zzdlab.com/hvidb/)，蛋白序列来自 [UniProt](https://www.uniprot.org/)，数据不因随代码发布而改用 MIT 许可。

使用实现时请引用上述 DeepHVI 论文；使用数据时同时引用[数据来源](docs/zh-CN/data.md)。反馈问题请提交 [GitHub Issue](https://github.com/Xindi-Wang1004/DeepHVI/issues)，附上运行命令、错误信息及相关 `deephvi doctor` 输出。
