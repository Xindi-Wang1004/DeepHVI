# 第三方声明

[English](THIRD_PARTY_NOTICES.md) · [项目说明](README.zh-CN.md)

## TMO-Net 兼容代码

`src/deephvi/compat/tmonet.py` 保留转换可信旧 checkpoint 所需的最小推理方法，适配 pickle 中的 `TMONet.model.TMO_Net_model` 类名。代码改编自 [FengAoWang/TMONet](https://github.com/FengAoWang/TMONet)，版权为 (c) 2024 FengAoWang，采用 MIT 许可。完整原始许可保留于 [LICENSE.TMONet](src/deephvi/compat/LICENSE.TMONet)。这些类支持转换，不包含原训练脚本。

## DeepHVI 参考样例

`tests/fixtures/aaindex_legacy_examples.txt` 仅包含从原始预测示例中提取的序列和特征字面量，用于独立核验恢复的 `aaindex12.csv`；测试不依赖旧版可执行代码。作者示例代码适用根目录 MIT，外部数据保留原始条款。

## 模型与数据

作者持有权利的 DeepHVI 分类器、预训练及投影权重适用根目录 MIT，不改变任何第三方底座许可。

ESM-2 编码器和 tokenizer 来自 [facebook/esm2_t6_8M_UR50D](https://huggingface.co/facebook/esm2_t6_8M_UR50D)。模型卡标注 MIT；Meta Platforms, Inc. 及其关联方的版权和完整许可保留于 [LICENSE.ESM2](docs/licenses/LICENSE.ESM2)，来源为 [facebookresearch/esm](https://github.com/facebookresearch/esm/blob/main/LICENSE)。解压目录 `data/models/esm2/` 保留上游模型卡，资源清单记录内容校验值。

当前公开目录和资源压缩包不包含 LucaProt 代码或权重。其原始 LICENSE 为 Apache-2.0，NOTICE 声明 Alibaba Group 版权及其他第三方事项；未来纳入时须保留许可、NOTICE、适用组件及修改声明。

数据将 [HVIDB](https://zzdlab.com/hvidb/) 的相互作用记录与 [UniProt](https://www.uniprot.org/) 的蛋白序列按标识符匹配。UniProt 中可受版权保护的数据库内容按[官方条款](https://www.uniprot.org/help/license)采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。HVIDB 汇集多个来源，其下载表含 `Source_Database`、`Pubmed_ID`；已查阅来源没有统一的数据许可声明。本项目保留引用，不为 HVIDB 汇编或上游内容另行指定 MIT 或 CC 许可。

百度网盘归档是整理数据的分发副本，并非原始数据库入口。数据与模型压缩包适用各自条款，详见[来源说明](docs/zh-CN/data.md)和[许可范围](docs/zh-CN/licensing.md)。原文许可声明保留原始文本，本页为说明性中文译文。
