# DeepHVI 数据与模型资源

[English](README.md) · [项目说明](../README.zh-CN.md)

正式资源以 `resources-*.tar.gz` 分发，每包小于 100 MiB。Git 跟踪根目录压缩包、清单和说明，解压资源子目录被忽略。

使用 `uv sync --frozen --extra data` 安装依赖后，在仓库根目录运行：

```bash
python3 scripts/resource_packages.py verify
python3 scripts/resource_packages.py extract
uv run deephvi verify-resources --resources-dir data
uv run deephvi predict --resources-dir data \
  --pairs data/examples/inputs/new_pairs.csv --out runs/predictions.csv --device cpu
```

| 目录 | 内容 |
| --- | --- |
| `training/` | 预训练 CSV：33,468 行；分类 CSV：去重前 36,465 行 |
| `auxiliary/` | 部分特征、有标签测试配对及候选序列 |
| `models/esm2/` | ESM-2 编码器与 tokenizer |
| `models/server_run/` | 分类器、预训练模型及配套投影 |
| `models/original/` | 供显式转换的原始 checkpoint |
| `cache/` | 历史序列嵌入 |
| `examples/` | 示例输入及预期、历史输出 |
| `configs/`、`splits/` | 配置、固定索引和有序划分输入 |
| `provenance/` | 来源与数据统计 |
| `archives/` | 原始整理数据归档 |

随附预训练输入为 `training/SH_SV_feature_with_vectors.csv`，分类输入为 `training/SH_SV_feature_with_vectors_0.5.csv`。分类文件原始记录含 24,312 正样本、12,153 负样本，去重后为 34,195 对。

`manifest.json` 和 `SHA256SUMS` 校验资源文件，`PACKAGES.json` 和 `PACKAGES.sha256` 校验压缩包成员及内容。压缩包合计约 328 MiB，解压后约 1.76 GiB。服务器预测采用包含 BOS/EOS 的 1024-token 上限、attention-mask 均值及保存的 320→256 投影；历史 512-token 缓存中被截断的长序列会重算，不覆盖原缓存。

相互作用记录来自 HVIDB，序列来自 UniProt，各自条款独立于代码的 MIT 许可。来源、命令和许可范围见[数据说明](../docs/zh-CN/data.md)、[资源说明](../docs/zh-CN/resources.md)及[许可说明](../docs/zh-CN/licensing.md)。原始[百度网盘副本](https://pan.baidu.com/s/1Ygk06WfLfOzHvYW7N4y6qQ)，提取码 `g9cn`。
