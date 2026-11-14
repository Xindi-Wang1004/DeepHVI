# 许可与署名

[English](../licensing.md) · [文档目录](README.md)

自有 DeepHVI 代码及作者持有权利的分类器、预训练和投影权重采用根目录 [MIT 许可](../../LICENSE)。MIT 允许使用、修改、再分发和商业使用，要求保留版权和许可声明。第三方模型与数据遵循下表中的条款。

| 内容 | 条款 | 保留的署名 |
| --- | --- | --- |
| 自有 DeepHVI 代码及权重 | MIT | 根目录 `LICENSE` |
| TMO-Net 兼容类 | MIT | `src/deephvi/compat/LICENSE.TMONet` |
| ESM-2 编码器和 tokenizer | 上游模型卡标注 MIT | 模型卡、来源链接及[完整上游许可](../licenses/LICENSE.ESM2) |
| UniProt 可受版权保护的数据库内容 | CC BY 4.0 | UniProt 名称、来源、许可链接和处理说明 |
| HVIDB 相互作用汇编 | 已查阅来源未声明统一数据许可证 | HVIDB 引用及原始数据库、文献来源 |

## 数据来源

作者从 [HVIDB](https://zzdlab.com/hvidb/) 整理相互作用记录，以蛋白标识符匹配 [UniProt](https://www.uniprot.org/) 序列，再完成配对处理和特征计算。HVIDB [下载表](https://zzdlab.com/hvidb/download/HVIDB_PPIs.txt)包含用于追溯的 `Source_Database` 和 `Pubmed_ID` 字段。

- HVIDB：保留数据库名、来源链接和[数据库论文](https://doi.org/10.1093/bib/bbaa425)，并遵循原始来源的署名与条款要求。
- UniProt：可受版权保护的数据库内容采用 [CC BY 4.0](https://www.uniprot.org/help/license)。需署名 UniProt，并说明 DeepHVI 的 ID 匹配、配对处理及特征派生。
- 派生表格：遵循相应来源数据的条款，保留原始引用。

文件用途、现有来源记录及引用见[数据说明](data.md)，组件详情见[第三方声明](../../THIRD_PARTY_NOTICES.zh-CN.md)。
