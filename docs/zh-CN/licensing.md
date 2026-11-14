# 许可与署名

[English](../licensing.md) · [文档目录](README.md)

自有 DeepHVI 代码及作者持有权利的分类器、预训练和投影权重采用根目录 [MIT 许可](../../LICENSE)，仅涵盖作者持有的权利。MIT 允许使用、修改、再分发和商业使用，要求保留版权和许可声明。第三方材料遵循各自条款。

| 内容 | 条款 | 保留的署名 |
| --- | --- | --- |
| 自有 DeepHVI 代码及权重 | MIT | 根目录 `LICENSE` |
| TMO-Net 兼容类 | MIT | `src/deephvi/compat/LICENSE.TMONet` |
| ESM-2 编码器和 tokenizer | 上游模型卡标注 MIT | 模型卡、来源链接及[完整上游许可](../licenses/LICENSE.ESM2) |
| UniProt 可受版权保护的数据库内容 | CC BY 4.0 | UniProt 名称、来源、许可链接和处理说明 |
| HVIDB 相互作用汇编 | 已查阅来源未声明统一数据许可证 | HVIDB 引用及原始数据库、文献来源 |

当前公开目录和资源压缩包不包含 LucaProt 代码或权重。其上游代码采用 Apache-2.0；将来纳入时需保留 LICENSE、NOTICE、适用组件声明和修改说明。微调或重新打包第三方模型不能消除底座条款。

## 数据来源

作者从 [HVIDB](https://zzdlab.com/hvidb/) 整理相互作用记录，以蛋白标识符匹配 [UniProt](https://www.uniprot.org/) 序列，再完成配对处理和特征计算。HVIDB [下载表](https://zzdlab.com/hvidb/download/HVIDB_PPIs.txt)包含用于追溯的 `Source_Database` 和 `Pubmed_ID` 字段。

- HVIDB：保留数据库名、来源链接和[数据库论文](https://doi.org/10.1093/bib/bbaa425)。本项目不为 HVIDB 汇编或上游内容另行指定 MIT 或 Creative Commons 许可。
- UniProt：可受版权保护的数据库内容采用 [CC BY 4.0](https://www.uniprot.org/help/license)。需署名 UniProt，并说明 DeepHVI 的 ID 匹配、配对处理及特征派生。
- 派生表格：代码和模型的 MIT 许可不替代外部数据条款，复用时需考虑原始来源及引用要求。

文件用途及引用见[数据说明](data.md)，组件详情见[第三方声明](../../THIRD_PARTY_NOTICES.zh-CN.md)。原始数据库版本和 ID 映射缺失限制了来源重建；校验值可以确认文件身份，不能证明再分发授权。
