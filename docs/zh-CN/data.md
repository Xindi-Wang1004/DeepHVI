# 数据说明

[English](../data.md) · [文档目录](README.md)

## 来源与署名

DeepHVI 使用 [Human–Virus Interaction Database（HVIDB）](https://zzdlab.com/hvidb/) 的相互作用记录，通过蛋白标识符匹配 [UniProt](https://www.uniprot.org/) 序列，构成人类—病毒蛋白配对。随附向量表是匹配、整理及特征处理后的派生数据。

HVIDB 引用：Yang X, Lian X, Fu C, Wuchty S, Yang S, Zhang Z. *HVIDB: a comprehensive database for human-virus protein-protein interactions*. Briefings in Bioinformatics, 2021, 22:832–844. [doi:10.1093/bib/bbaa425](https://doi.org/10.1093/bib/bbaa425)。UniProt 序列请按[官方引用指南](https://www.uniprot.org/help/publications)引用。

UniProt 数据库内容采用 [CC BY 4.0](https://www.uniprot.org/help/license)。HVIDB 汇集多个数据库和文献的记录，引用时保留对应原始来源。随附材料的条款和声明见[许可说明](licensing.md)。

随附表格通过文件校验值标识版本。其原始数据库版本、下载日期和完整 ID 映射尚不可用；依赖这些具体记录的实验可直接使用随附快照。

作者整理的数据归档可从[百度网盘副本](https://pan.baidu.com/s/1Ygk06WfLfOzHvYW7N4y6qQ)获取，提取码为 `g9cn`；恢复资源后，同一归档位于 `data/archives/data.7z`。

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

## 公共检索参考库

[参考序列流程](reference-sequences.md)下载完整人类 RefSeq `GCF_000001405.40` 和病毒 UniProtKB `taxonomy_id:10239` FASTA。两侧各按完整序列去重，将原始 accession 和标题保存在查询目录中。`--reference-source both` 可随训练一并准备参考库，记录供后续检索使用的索引。

## CSV 格式

标准列为 `pair_id,human_seq,viral_seq,label`，序列列也可使用 `S_H,S_V` 或 `seq_a,seq_b`。`pair_id` 可省略，此时按序列生成稳定 ID。分类训练需要 0/1 标签，预测可不带标签。自定义列名用 `--human-col`、`--viral-col`、`--label-col` 和 `--pair-id-col`。

```csv
pair_id,human_seq,viral_seq,label
example_1,ACDEFGHIKLMNPQRSTVWY,MKTIIALSYIFCLVFAD,1
```

读取时移除空白和比对间隙，转为大写，并移除末尾终止标记。模糊残基的 AAindex 描述符为零。AAindex 使用前 100 个残基、每残基 12 项性质，形成 1,200 维向量，短序列补零。描述符由安装包内的 AAindex 性质表计算。

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

`paper`、`group` 协议按完整序列配对去重，保留首行并检查标签冲突，分类表处理后为 34,195 对。从旧产物迁移时，一起重建特征和划分。

根据实验目标选择分组方式：`paper` 按配对划分，`group` 的 `union_find` 将相互连接的蛋白保持在同一组，`virus_species` 则按病毒物种分组。物种组之间可能共享人类蛋白。

历史协议保留原始行序和重复行，便于对照早期运行；分阶段命令沿用对应历史配置。划分 manifest 报告配对与蛋白重叠，需要按蛋白分离的实验可添加 `--strict-leakage`。预测保留输入行序、重复行与 ID。
