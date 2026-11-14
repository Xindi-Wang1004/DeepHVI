# 编码与复现约定

[English](../reproducibility.md) · [文档目录](README.md)

本文说明特征编码与实验设置。操作命令见[使用说明](usage.md)，来源与数据处理见[数据说明](data.md)。

## 特征编码

每个配对包含四种模态：人类和病毒 ESM 嵌入，以及人类和病毒 AAindex 描述符。每条 AAindex 描述符由前 100 个残基的 12 项性质组成，共 1,200 维。

| 流程 | ESM pooling 与输入上限 | 投影 | 融合模型 |
| --- | --- | --- | --- |
| 已发布服务器预测 | 按 attention mask 求均值，包含特殊 token；1024 token | 保存的 `models/server_run/projection.pt`，320→256 | `server_tmonet_v1` |
| 重构模型训练 | 可配置；默认排除特殊 token 的 masked mean，1024 token | 每个 checkpoint 内训练并保存 | `deephvi` |

1024-token 上限包含 BOS/EOS，最多容纳 1022 个残基 token。历史服务器缓存使用 512 token：不超过 510 残基的序列复用已有值，较长序列按当前上限编码。长蛋白的输入表示会随之变化，因此对照历史预测时应记录 token 上限；原缓存仍保留供对照使用。

重构模型的高斯 Product of Experts 使用逆方差 `exp(-logvar)` 并结合先验精度，checkpoint 要求模型修订号 2，旧训练权重需重训。服务器兼容图保留历史计算，详见[模型权重](checkpoints.md)。

## 训练设置

`configs/default.yaml` 使用外层 80/20 划分、训练部分三折验证、预训练 10 轮、分类训练 30 轮、batch size 16，以及学习率 `1e-4` 的 Adam。分类损失为交叉熵加权重 `0.01` 的对比项。

实现采用温度 `0.07` 的监督对比损失。预训练结合重构、交叉重构、KL、对抗和一致性项，并对 AAindex 重构进行缩放。各项权重见 `configs/default.yaml` 与每次运行保存的配置。

模型选择依据验证集，测试集用于最终评估。现代协议划分前按完整配对去重。`paper` 按配对划分；`group` 的 `union_find` 按蛋白连接组划分，`virus_species` 按物种划分。

训练与推理共享由序列特征编码器准备的参考库。`--reference-source both` 将来源版本和索引位置写入 `references.json` 与 checkpoint 的 `reference_libraries`。配对表提供模型训练数据，参考库提供检索候选。模型初始化前会恢复训练种子，使首次建库与复用缓存得到相同的初始化。衔接方式见[参考序列流程](reference-sequences.md)。

## 历史对照

`configs/server_historical.yaml` 记录 seed=42、打乱的十折 KFold、零起始 fold=3，以及训练部分按行序 90/10 划分。早期 `legacy` 配置采用不打乱十折，并以验证集充当测试集。历史训练 loader 预测保存在 `data/examples/historical/`。

## 可追溯信息与指标

每个 checkpoint 保存模型与特征配置、数据和划分指纹、训练折、种子、epoch、验证指标及环境。资源 manifest 通过 SHA-256 标识服务器配套文件；特征包和参考索引也记录编码设置，供采用相同编码的运行复用。

```bash
uv run deephvi recompute-metrics \
  --predictions runs/experiment/predictions/test_fold0.csv --out runs/recomputed.json
```

该命令根据保存的标签和正类概率，重算准确率、宏平均 precision/recall/F1 和 ROC-AUC，默认分类阈值为 0.5。AUC 均值汇总同时包含两类样本的折。

运行结果包含逐样本预测和汇总指标。测试与模型检查命令见[验证方法](validation.md)。
