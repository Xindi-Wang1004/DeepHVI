# 编码与复现约定

[English](../reproducibility.md) · [文档目录](README.md)

本文说明解读 DeepHVI 输出所需的计算方式和实验假设。操作命令见[使用说明](usage.md)，数据构建限制见[数据说明](data.md)。

## 特征编码

每个配对包含四种模态：人类和病毒 ESM 嵌入，以及人类和病毒 AAindex 描述符。每条 AAindex 描述符由前 100 个残基的 12 项性质组成，共 1,200 维。

| 流程 | ESM pooling 与输入上限 | 投影 | 融合模型 |
| --- | --- | --- | --- |
| 已发布服务器预测 | 按 attention mask 求均值，包含特殊 token；1024 token | 保存的 `models/server_run/projection.pt`，320→256 | `server_tmonet_v1` |
| 重构模型训练 | 可配置；默认排除特殊 token 的 masked mean，1024 token | 每个 checkpoint 内训练并保存 | `deephvi` |

1024-token 上限包含 BOS/EOS，因此最多容纳 1022 个残基 token，与原 GitHub 编码器默认值一致。历史服务器缓存采用 512-token 上限：不超过 510 残基的完整序列可复用，较长序列按当前上限重算。服务器权重没有因此重训，长序列预测可能与历史 512-token 输出不同。磁盘中的原缓存会保留。

重构模型的高斯 Product of Experts 使用逆方差 `exp(-logvar)` 并结合先验精度，checkpoint 要求模型修订号 2，旧训练权重需重训。服务器兼容图保留历史计算，详见[模型权重](checkpoints.md)。

## 训练默认值与假设

`configs/default.yaml` 使用外层 80/20 划分、训练部分三折验证、预训练 10 轮、分类训练 30 轮、batch size 16，以及学习率 `1e-4` 的 Adam。分类损失为交叉熵加权重 `0.01` 的对比项。

论文未完整指定的细节按实现假设处理：温度 `0.07` 的监督对比损失、预训练中的重构/交叉重构/KL/对抗/一致性权重，以及 AAindex 重构缩放。精确数值见 `configs/default.yaml` 与运行时保存的解析配置。对照实验时应报告这些参考实现选择。

模型选择依据验证集，测试集用于最终评估。现代协议划分前移除相同完整配对，但默认 `paper` 配对划分仍可共享单个蛋白。评估未见蛋白的泛化时，应使用蛋白连接组并检查重叠统计；仅按物种分组不能保证人类蛋白互斥。

## 历史对照

`configs/server_historical.yaml` 记录 seed=42、打乱的十折 KFold、零起始 fold=3，以及训练部分按行序 90/10 划分。早期 `legacy` 配置采用不打乱十折，并以验证集充当测试集。分享包中的历史预测使用训练 loader，位于 `data/examples/historical/`，不能直接与独立测试指标比较。

## 可追溯信息与指标

新 checkpoint 保存模型与特征配置、精确划分和数据指纹、训练折、种子、epoch、验证指标及环境。资源 manifest 通过 SHA-256 绑定历史服务器配套文件。复用特征包和参考索引时要求编码器及编码设置一致。

```bash
uv run deephvi recompute-metrics \
  --predictions runs/experiment/predictions/test_fold0.csv --out runs/recomputed.json
```

准确率、宏平均 precision/recall/F1 和 ROC-AUC 均由保存的标签和正类概率重算，默认阈值为 0.5。单类别分区的 AUC 未定义，不计入有效折平均。应同时保留逐样本预测和汇总指标。

hash 后端仅用于检查软件行为；小型真实 ESM 实验验证集成。完整论文训练、CUDA 确定性及跨环境一致性需单独实验。可重复执行的检查见[验证方法](validation.md)。
