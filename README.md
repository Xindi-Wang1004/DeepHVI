# DeepHVI

[English](README.md) · [简体中文](README.zh-CN.md)

DeepHVI predicts human–virus protein interactions by combining ESM-2 sequence embeddings with AAindex physicochemical descriptors. This repository provides pretrained inference, feature preparation, training, fold-specific evaluation, and protein similarity retrieval.

Paper: *DeepHVI: A multimodal deep learning framework for predicting human-virus protein-protein interactions using protein language models.* **Biosafety and Health**, 2025, 7:257–266.

## Install

Python 3.10 or later is required. Run the following commands from a terminal:

```bash
git clone https://github.com/Xindi-Wang1004/DeepHVI.git
cd DeepHVI
python3 -m pip install uv
uv sync --frozen --extra dev --extra data
```

The lockfile installs CPU PyTorch. For CUDA, follow the [installation guide](docs/usage.md#cuda). Alternatively, install `python -m pip install '.[data]'` inside a virtual environment and use `deephvi` directly in place of `uv run deephvi`.

## Try the pipeline without model downloads

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run deephvi demo \
  --backend hash --n-pairs 30 --epochs 1 --device cpu --out runs/demo
```

This runs feature preparation, training, evaluation, and retrieval on synthetic pairs. The hash encoder is a deterministic test backend; its metrics do not measure biological prediction performance.

## Restore data and pretrained models

Resource archives are distributed in `data/resources-*.tar.gz`. After cloning, run from the repository root:

```bash
python3 scripts/resource_packages.py verify
python3 scripts/resource_packages.py extract
uv run deephvi verify-resources --resources-dir data
```

The extraction tool uses the Python standard library and verifies archives and extracted content. Archives total about 328 MiB; extracted resources occupy about 1.76 GiB. Allow at least 8 GiB for resources, dependencies, Git history, and outputs. See the [resource guide](docs/resources.md) for file roles and integrity checks.

## Predict interactions

Create a CSV with `human_seq,viral_seq` columns and, optionally, `pair_id`. `S_H,S_V` and `seq_a,seq_b` are also accepted. A minimal example is:

```csv
pair_id,human_seq,viral_seq
example_1,ACDEFGHIKLMNPQRSTVWY,MKTIIALSYIFCLVFAD
```

Run a supplied example:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run deephvi predict \
  --resources-dir data --pairs data/examples/inputs/new_pairs.csv \
  --out runs/predictions.csv --device cpu
```

Output includes sample IDs, sequences, positive-class `probability`, and `prediction` (default threshold: 0.5). Input order and duplicate rows are preserved. The released server model uses its matching ESM-2 encoder, saved 320→256 projection, and compatible sequence cache. Uncached sequences are encoded automatically.

The encoding limit is 1024 tokens including BOS/EOS. Historical cache entries truncated at 512 tokens are recomputed when needed. See [encoding and reproducibility](docs/reproducibility.md) before comparing results with historical predictions.

## Train and evaluate

```bash
uv run deephvi reproduce --resources-dir data --config configs/default.yaml \
  --pairs data/training/SH_SV_feature_with_vectors_0.5.csv --out runs/experiment
```

This command audits data, encodes features, saves splits, trains each fold, and evaluates its selected checkpoint. Modern protocols deduplicate exact sequence pairs and reject conflicting labels. The default `paper` protocol uses a held-out 20% test set and three folds within the training portion. Shared proteins can still cross pair-level splits; use the `group` protocol for protein-disjoint experiments.

Training commands and smaller CPU examples are in the [usage guide](docs/usage.md). This implementation includes documented reconstruction assumptions. Passing a small pipeline check does not establish reproduction of the paper's full metrics.

## Documentation

| Topic | English | 简体中文 |
| --- | --- | --- |
| Installation and CLI workflows | [Usage](docs/usage.md) | [使用说明](docs/zh-CN/usage.md) |
| Data sources, schema, and splits | [Data](docs/data.md) | [数据说明](docs/zh-CN/data.md) |
| Resource archives and checksums | [Resources](docs/resources.md) | [资源说明](docs/zh-CN/resources.md) |
| Models and checkpoint compatibility | [Checkpoints](docs/checkpoints.md) | [模型权重](docs/zh-CN/checkpoints.md) |
| Encoding and experiment assumptions | [Reproducibility](docs/reproducibility.md) | [复现约定](docs/zh-CN/reproducibility.md) |
| Repository and contribution workflow | [Repository layout](docs/repository-layout.md) | [仓库布局](docs/zh-CN/repository-layout.md) |
| Validation commands and scope | [Validation](docs/validation.md) | [验证方法](docs/zh-CN/validation.md) |
| Licensing and attribution | [Licensing](docs/licensing.md) | [许可说明](docs/zh-CN/licensing.md) |

## License, citation, and support

Original code and author-owned DeepHVI weights are covered by [MIT](LICENSE). Third-party models and database content retain their upstream terms; see [third-party notices](THIRD_PARTY_NOTICES.md) and [license scope](docs/licensing.md). Interaction records originate from [HVIDB](https://zzdlab.com/hvidb/), and protein sequences from [UniProt](https://www.uniprot.org/). The data are not relicensed under the code's MIT license.

Please cite the DeepHVI paper above when using this implementation, and cite the [data sources](docs/data.md) when using the dataset. Report problems through [GitHub Issues](https://github.com/Xindi-Wang1004/DeepHVI/issues), including the command, error message, and relevant `deephvi doctor` output.
