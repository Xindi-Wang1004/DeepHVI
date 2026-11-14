# 参考序列流程

[English](../reference-sequences.md) · [使用说明](usage.md)

参考库为 `retrieve` 和 `generate` 提供天然蛋白候选序列。两种命令均按序列 embedding 的余弦相似度排列候选。

| 参考库 | 来源 |
| --- | --- |
| 人类 | NCBI RefSeq `GCF_000001405.40`（GRCh38.p14）完整蛋白 FASTA，包含异构体、补丁和替代位点 |
| 病毒 | UniProtKB `taxonomy_id:10239` 全部查询结果，包含 reviewed、unreviewed 条目及分类后代 |

两侧分别合并相同的完整序列，保留全部原始 accession。下载与处理记录包含来源版本、序列统计和文件校验值，便于在不同运行之间复用同一份参考库。

安装依赖并[恢复模型资源](resources.md)后，在仓库根目录执行以下命令。

## 直接检索公共参考库

通过 `--reference-source` 选择参考库。首次运行会依次下载 FASTA、整理序列目录、计算 embedding、建立索引并检索：

```bash
uv run deephvi retrieve --resources-dir data \
  --pairs data/examples/inputs/new_pairs.csv \
  --reference-source viral --reference-dir data/reference \
  --top-k 5 --out runs/retrieval.json
```

检索人类库时改用 `--reference-source human`。`--target` 默认与所选参考库一致：`viral` 从输入的 `viral_seq` 列读取查询，`human` 从 `human_seq` 列读取查询。`generate` 接受相同参数，返回所选库内的候选序列。

后续运行会校验并复用已保存的参考库和索引，再编码查询序列。JSON 输出包含排序后的候选、索引位置，以及记录来源和 accession 映射路径的 `reference` 字段。CSV 每行对应一个命中结果，CLI 同时打印索引位置。

## 随训练准备参考库

在完整训练命令中加入参考库参数，即可一并准备两侧资源：

```bash
uv run deephvi reproduce --resources-dir data --config configs/default.yaml \
  --pairs data/training/SH_SV_feature_with_vectors_0.5.csv \
  --reference-source both --reference-dir data/reference --out runs/experiment
```

流程使用训练特征的序列编码器建立两侧参考索引，各折共享同一组索引。运行目录的 `references.json` 和每个 checkpoint 的 `reference_libraries` 字段保存来源版本、指纹和索引位置。模型训练使用传入的配对表，参考库为后续检索提供候选序列。

分阶段的 `train-pretrain`、`train-classifier` 支持相同参考库参数。分类训练未选择新来源时，会从 `--pretrain-checkpoint` 继承参考记录。分阶段命令见[使用说明](usage.md#训练)。

也可以将参考库设置写入 YAML：

```yaml
references:
  sources: [human, viral]
  dir: data/reference
  overwrite: false
```

CLI 参数优先于 YAML。默认的 `sources: []` 只准备训练所需的配对特征；命令行选择 `human`、`viral` 或 `both` 后，会加入对应参考资源的准备步骤。

分类推理时，可给 `predict` 添加 `--reference-source both --reference-dir data/reference`，将索引记录写入 `<预测文件>.references.json`。通过 CLI 或 YAML 选择参考库后才会执行这一步；普通分类直接读取配对输入和模型资源。

## 复用训练时的索引

传入训练产生的 checkpoint，并选择检索侧：

```bash
uv run deephvi generate --checkpoint runs/experiment/fold0/checkpoints/best.pt \
  --pairs data/examples/inputs/new_pairs.csv --target viral --out runs/candidates.json
```

checkpoint 提供序列编码设置和已记录的索引，`--target human` 切换到人类索引。检索将所选侧的查询 embedding 与参考 embedding 比较，并通过来源和编码指纹校验匹配关系。

迁移索引时，复制整个目录，并用 `--checkpoint ... --index NEW_DIR --target human|viral` 指定新位置。索引文件与 `pipeline.json` 一起保留，即可延续训练时的记录。checkpoint 中尚无参考记录时，可通过 `--reference-source`、`--reference` 或 `--index` 选择参考库。

## 单独下载和处理

独立脚本适合提前准备参考库，或离线处理已有 FASTA，仅依赖 Python 标准库。

下载两侧来源：

```bash
uv run --frozen python scripts/download_reference_sequences.py \
  --source both --output data/reference/raw
```

`--source human`、`--source viral` 分别只下载一侧。随后处理各参考库：

```bash
uv run --frozen python scripts/prepare_reference_sequences.py \
  --source human \
  --fasta data/reference/raw/human/GCF_000001405.40_GRCh38.p14_protein.faa.gz \
  --download-receipt data/reference/raw/human/download.json \
  --output data/reference/prepared/human

uv run --frozen python scripts/prepare_reference_sequences.py \
  --source viral \
  --fasta data/reference/raw/viral/uniprotkb_taxonomy_id_10239.fasta.gz \
  --download-receipt data/reference/raw/viral/download.json \
  --output data/reference/prepared/viral
```

处理过程流式读取 FASTA，通过 SQLite 按完整序列去重。序列统一转为大写，移除空白、比对间隙和末尾终止符，同时检查残基字符与 accession 映射。长蛋白在导出的 FASTA 中保留完整序列。

已有 `.faa`、`.fasta` 或 gzip FASTA 可直接通过 `--fasta` 指定。`--download-receipt` 将输入关联到本下载脚本的来源记录；其他本地输入可省略此参数，由清单中的 SHA-256 标识文件。

处理后的 FASTA 可直接用于检索：

```bash
uv run deephvi retrieve --resources-dir data \
  --pairs data/examples/inputs/new_pairs.csv \
  --reference data/reference/prepared/viral/reference.fasta \
  --index-out runs/viral-reference-index \
  --target viral --top-k 5 --out runs/retrieval.json
```

后续将 `--reference ... --index-out ...` 替换为 `--index runs/viral-reference-index` 即可复用索引。人类和病毒库各使用独立索引目录。

## 文件与存储

参考数据默认保存在 Git 忽略的 `data/reference/` 下，`--reference-dir` 可覆盖 YAML 的 `references.dir`。训练与推理共用以下缓存结构：

| 路径 | 内容 |
| --- | --- |
| `raw/<source>/` | 原始压缩 FASTA 和 `download.json`；人类下载还保存 NCBI 的 MD5 表 |
| `raw/viral/pages/` | UniProt 压缩分页，进度保存在 `raw/viral/download-progress.json` |
| `prepared/<source>/reference.fasta` | 每条完整序列对应稳定的 `<source>_<sequence_sha256>` 候选 ID |
| `prepared/<source>/accessions.tsv` | 候选 ID、全部原始 accession 和 FASTA 标题 |
| `prepared/<source>/catalog.sqlite3` | 序列、accession 和来源元数据表 |
| `prepared/<source>/manifest.json` | 处理策略、来源记录、序列统计和产物校验值 |
| `indices/<source>/<fingerprint>/` | 按编码器区分的索引、元数据及完成记录 `pipeline.json` |

通过结果中的候选 ID，可在 `accessions.tsv` 或 `catalog.sqlite3` 查询全部原始 accession；FASTA 描述中也附有一个代表 accession。

自动索引路径由来源、处理策略和编码设置决定，编码设置包括模型内容、池化、截断及适用的外部投影。兼容运行共享索引，更换编码器会使用新的目录。`--index-out DIR` 可指定索引位置。

SQLite 控制处理阶段的内存占用。编码和搜索会将参考序列与向量载入内存，应按完整库规模安排磁盘、内存和编码时间。随附 `server_esm2` 使用含 BOS/EOS 的 1024-token 上限，普通序列最多编码 1022 个残基；FASTA 保存完整序列。

## 版本与重跑

已完成下载保持其保存的版本，每次复用前会校验 FASTA 与索引指纹。需要获取新的来源快照时，使用新的参考目录。

| 情况 | 操作 |
| --- | --- |
| 下载中断 | 重复原命令。UniProt 从已保存分页续传；人类文件未完成时重新下载该文件。 |
| 下载期间 UniProt release 更新 | 换新目录开始下载，以获得一致版本。 |
| 处理或建索引中断 | 重复原命令，重新构建未完成阶段；参考库 embedding 从头计算。 |
| 需要重建索引 | `retrieve/generate` 使用 `--overwrite`；新训练或预测运行使用 `--reference-overwrite`。 |
| 来源文件或已完成产物校验失败 | 恢复保存的版本，或换新目录准备一份完整资源。 |
| 使用 `--resume` 续训 | 沿用记录的参考库和索引版本；更换参考库时创建新的训练运行。 |

为复现运行，应一起保留原始 FASTA、来源记录和索引。NCBI 下载使用官方 MD5，UniProt 下载逐页核对 release 与总记录数；后续处理和建索引通过 SHA-256 标识保存的文件。

来源引用和条款见[许可说明](licensing.md)、[NCBI 政策](https://www.ncbi.nlm.nih.gov/home/about/policies/)和 [UniProt 条款](https://www.uniprot.org/help/license)。
