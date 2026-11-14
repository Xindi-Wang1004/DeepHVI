# 使用说明

[English](../usage.md) · [文档目录](README.md)

需要 Python 3.10+。在仓库根目录用 `uv sync --frozen --extra dev --extra data` 安装依赖；预训练预测还需先运行 `python3 scripts/resource_packages.py extract` 恢复资源。使用 pip 安装时，将 `uv run deephvi` 替换为 `deephvi`。

```bash
uv run deephvi --help
uv run deephvi predict --help
```

公共参数包括 `--config FILE`、可重复的 `--set KEY=VALUE`、`--seed`、`--device cpu|cuda|auto` 和 `--resources-dir DIR`。相对路径以当前工作目录为基准。CLI 自动识别当前目录的 `data/manifest.json`；从其他目录运行时，请显式指定资源目录。

## CUDA

仓库的 uv 锁文件默认选择 CPU 版 PyTorch。CUDA 用户应另建环境，先使用 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)安装适合平台和驱动的版本，再安装项目：

```bash
python3 -m venv .venv-cuda
source .venv-cuda/bin/activate
python -m pip install --upgrade pip
# 在此运行 PyTorch 官方选择器生成的 CUDA 安装命令。
python -m pip install '.[data]'
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
deephvi predict --resources-dir data --pairs input.csv --out predictions.csv --device cuda
```

Windows 使用 `.venv-cuda\Scripts\activate` 激活。在此环境直接使用 `deephvi`；`uv sync` 和 `uv run` 会选择仓库的 CPU 环境。下文环境变量示例采用 POSIX 语法；PowerShell 请分别设置 `$env:OMP_NUM_THREADS='2'` 和 `$env:MKL_NUM_THREADS='2'`。

## 预测

```bash
uv run deephvi predict --resources-dir data \
  --pairs data/examples/inputs/new_pairs.csv --out runs/predictions.csv --device cpu
```

输入列为 `human_seq,viral_seq`，可附 `pair_id`，字段别名见[数据说明](data.md)。输出包含 `pair_id,probability,prediction` 和序列。可添加 `--set eval.threshold=0.7` 修改分类阈值。

## 小型 CPU 示例

无需资源的合成示例：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run deephvi demo \
  --backend hash --n-pairs 30 --epochs 1 --device cpu --out runs/demo
```

使用真实 ESM-2 资源：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run deephvi reproduce \
  --resources-dir data --config configs/cpu_smoke_esm.yaml \
  --pairs data/examples/inputs/cpu_smoke_pairs.csv --limit 18 --epochs 1 \
  --out runs/cpu-example --device cpu
```

两者用于检查流程，合成数据或小样本指标不代表论文基准。

## 数据与特征

```bash
uv run deephvi inspect-data --input data/training/SH_SV_feature_with_vectors_0.5.csv --count
uv run deephvi prepare-features --resources-dir data --config configs/default.yaml \
  --pairs data/training/SH_SV_feature_with_vectors_0.5.csv --out runs/features --device cpu
uv run deephvi make-splits --config configs/default.yaml \
  --pairs data/training/SH_SV_feature_with_vectors_0.5.csv --out runs/splits
```

特征和划分必须使用同一输入、配置与样本限制。现代协议先去除相同配对，历史协议保留原行。复用特征时检查样本、编码器权重、pooling、长度、投影和 AAindex 表；编码器改变后，请使用新目录或 `prepare-features --overwrite` 重建。

## 训练

统一流程：

```bash
uv run deephvi reproduce --resources-dir data --config configs/default.yaml \
  --pairs data/training/SH_SV_feature_with_vectors_0.5.csv --out runs/experiment
```

依次检查样本、生成特征与划分、逐折预训练和分类训练，最后保存评估输出与 `checkpoint_map.json`。`--epochs` 同时覆盖两个阶段。小规模检查时，`reproduce --limit N` 在去重后交替抽取两类；分阶段命令则读取前 N 行。比较两种流程时应省略限制，或提供同一份已保存的子集 CSV。

完成前述特征与划分命令后，可分阶段训练：

```bash
uv run deephvi train-pretrain --config configs/default.yaml \
  --bundle runs/features --splits runs/splits --fold 0 --epochs 1
uv run deephvi train-classifier --config configs/default.yaml \
  --bundle runs/features --splits runs/splits --fold 0 --epochs 1 \
  --pretrain-checkpoint PATH_TO_BEST_PRETRAIN_PT
```

将 `PATH_TO_BEST_PRETRAIN_PT` 替换为预训练输出的路径。结果写入 `runs/<name>/<timestamp>-<id>/`。使用 `--resume PATH_TO_LAST_PT` 续训，`--epochs` 表示目标总轮数；需保留配套优化器状态及最佳模型，详见[模型权重](checkpoints.md)。

## 评估与指标重算

```bash
uv run deephvi evaluate --checkpoint-map runs/experiment/checkpoint_map.json \
  --bundle runs/experiment/features --splits runs/experiment/splits \
  --fold all --out runs/evaluation
uv run deephvi recompute-metrics \
  --predictions runs/evaluation/predictions/test_fold0.csv --out runs/metrics.json
```

单折评估将 `--checkpoint-map ... --fold all` 替换为 `--checkpoint PATH_TO_BEST_PT --fold 0`。映射文件格式为 `{"0":"fold0/checkpoints/best.pt","1":"fold1/checkpoints/best.pt"}`，路径相对于映射文件。评估时检查各 checkpoint 的折号及划分指纹。

## 相似性检索

```bash
uv run deephvi retrieve --resources-dir data \
  --pairs data/examples/inputs/new_pairs.csv --reference reference.fasta \
  --target viral --top-k 5 --out runs/retrieval.json
```

需提供参考 FASTA，或用 `--index DIR` 复用相同编码器及设置生成的索引。`--target human` 检索人类侧序列。`generate` 采用同一检索接口；已发布资源没有论文完整条件生成任务所需的 decoder 权重。

## 诊断与配置

```bash
uv run deephvi verify-resources --resources-dir data
uv run deephvi doctor --resources-dir data --write dev-docs/reports/doctor.md
uv run deephvi config show --config configs/default.yaml
```

资源维护见[资源说明](resources.md)，旧权重转换见[模型权重](checkpoints.md)。
