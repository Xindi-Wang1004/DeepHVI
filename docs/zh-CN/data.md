# 数据说明

[English](../data.md) · [文档目录](README.md)

## 来源与署名

DeepHVI 使用 [Human–Virus Interaction Database（HVIDB）](https://zzdlab.com/hvidb/) 的相互作用记录，通过蛋白标识符匹配 [UniProt](https://www.uniprot.org/) 序列，构成人类—病毒蛋白配对。随附向量表是匹配、整理及特征处理后的派生数据。

HVIDB 引用：Yang X, Lian X, Fu C, Wuchty S, Yang S, Zhang Z. *HVIDB: a comprehensive database for human-virus protein-protein interactions*. Briefings in Bioinformatics, 2021, 22:832–844. [doi:10.1093/bib/bbaa425](https://doi.org/10.1093/bib/bbaa425)。UniProt 序列请按[官方引用指南](https://www.uniprot.org/help/publications)引用。

UniProt 中可受版权保护的数据库内容采用 [CC BY 4.0](https://www.uniprot.org/help/license)。HVIDB 汇集多个公开来源，已查阅页面及下载表未声明统一的数据许可证。使用时需保留来源及适用上游条款，项目的 MIT 许可不重新授权这些数据，详见[许可说明](licensing.md)。

原始数据库版本、下载日期、完整 ID 映射和未匹配记录清单尚不可用。校验值可以固定本仓库分发的快照，但不能保证从当前在线数据库重建出逐行相同的数据。

[百度网盘副本](https://pan.baidu.com/s/1Ygk06WfLfOzHvYW7N4y6qQ)，提取码 `g9cn`，是作者整理的归档，并非 HVIDB 或 UniProt 官方入口。恢复资源后，同一归档位于 `data/archives/data.7z`。

## 随附数据表

在仓库根目录运行 `python3 scripts/resource_packages.py extract` 恢复以下文件：

| 路径 | 行数 | 用途 |
| --- | ---: | --- |
| `data/training/SH_SV_feature_with_vectors.csv` | 33,468 | 预训练输入与历史 ESM 缓存生成 |
| `data/training/SH_SV_feature_with_vectors_0.5.csv` | 36,465 | 分类训练：24,312 正样本、12,153 负样本 |
| `data/auxiliary/SH_SV_feature_with_vectors_partial.csv` | 26,990 | 部分序列特征表 |
| `data/auxiliary/human_virus_test.csv` | 7,623 | 带蛋白标识符的有标签配对 |
| `data/auxiliary/S.csv` | 90,057 | 候选配对 |
| `data/auxiliary/inference_processed.csv` | 90,052 | 处理后的候选配对 |

两个训练文件分别包含 28,931 和 34,195 个唯一序列配对。第二个文件的正负样本行数对应去重前原表。详细统计位于 `data/provenance/training_statistics.json`，内容校验值位于 `data/manifest.json`。论文数据总量与实际可用训练表采用不同统计口径。

## CSV 格式

标准列为 `pair_id,human_seq,viral_seq,label`，序列列也可使用 `S_H,S_V` 或 `seq_a,seq_b`。`pair_id` 可省略，此时按序列生成稳定 ID。分类训练需要 0/1 标签，预测可不带标签。自定义列名用 `--human-col`、`--viral-col`、`--label-col` 和 `--pair-id-col`。

```csv
pair_id,human_seq,viral_seq,label
example_1,ACDEFGHIKLMNPQRSTVWY,MKTIIALSYIFCLVFAD,1
```

读取时移除空白和比对间隙，转为大写，并移除末尾终止标记。模糊残基的 AAindex 描述符为零。AAindex 使用前 100 个残基、每残基 12 项性质，形成 1,200 维向量，短序列补零。描述符使用安装包内的性质表重新计算，不直接信任 CSV 中的向量字符串。

```bash
uv run deephvi inspect-data --input data/training/SH_SV_feature_with_vectors_0.5.csv --count
uv run deephvi prepare-data --archive data/archives/data.7z --dest runs/extracted
```

原始归档包含顶层 `data/`，解压时会记录不完整 CSV 尾行的修复情况。

## 去重与划分

`prepare-features` 保存序列特征、配对索引、标签和 ID；`make-splits` 保存精确行索引与指纹。两者必须使用相同顺序的输入及配置。`data/splits/cpu_smoke/` 的样例划分附有对应有序 `pairs.csv`。

| 协议 | 含义 |
| --- | --- |
| `paper` | 默认保留 20% 测试集，在训练部分进行三折验证 |
| `group` | 按蛋白连接组 `union_find` 或病毒物种 `virus_species` 分组 |
| `server_historical` | 服务器随机十折约定，内部再划分 90/10 |
| `legacy` | 早期不打乱十折约定，验证集同时用作测试集 |

现代 `paper`、`group` 流程按完整序列配对去重，保留首行，标签冲突时报错；分类表去重后为 34,195 对。从旧版迁移时应同时重建特征和划分。配对层面的划分仍可共享单个蛋白；连接组方式将各侧相同蛋白保持在同一组。物种分组只分离物种组，不保证人类蛋白互斥。

历史协议为对照保留原始顺序与重复行，其结果不能视为与现代去重协议同等独立的评估。分阶段运行历史协议须使用一致的历史配置。各划分 manifest 记录蛋白和配对重叠；`--strict-leakage` 会拒绝跨分区蛋白重叠。预测保留输入重复行与 ID。
