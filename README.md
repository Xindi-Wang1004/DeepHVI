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

## Try the CPU demo

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run deephvi demo \
  --backend hash --n-pairs 30 --epochs 1 --device cpu --out runs/demo
```

This example uses synthetic pairs and a deterministic hash encoder to run feature preparation, training, evaluation, and retrieval on CPU.

## Restore data and pretrained models

Resource archives are distributed in `data/resources-*.tar.gz`. After cloning, run from the repository root:

```bash
python3 scripts/resource_packages.py verify
python3 scripts/resource_packages.py extract
uv run deephvi verify-resources --resources-dir data
```

The archives total about 328 MiB and expand to about 1.76 GiB. Allow at least 8 GiB for resources, dependencies, Git history, and outputs. The [resource guide](docs/resources.md) describes their contents and checksums; full reference libraries use additional storage.

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

The output CSV contains each sample's ID, sequences, interaction `probability` and `prediction`, with a default threshold of 0.5. Rows follow the input order, including duplicates. The model automatically encodes new sequences and reuses cached features where available.

The supplied encoder uses up to 1024 tokens including BOS/EOS. Pooling, projection and cache settings are described in [Encoding and reproducibility](docs/reproducibility.md).

## Train and evaluate

```bash
uv run deephvi reproduce --resources-dir data --config configs/default.yaml \
  --pairs data/training/SH_SV_feature_with_vectors_0.5.csv \
  --reference-source both --reference-dir data/reference --out runs/experiment
```

This command prepares pair features and splits, trains each fold, and evaluates the selected checkpoints. The default `paper` protocol reserves 20% for testing and uses three validation folds within the training portion. Use the `group` protocol for experiments that separate connected proteins across splits.

`--reference-source both` prepares human and viral reference libraries for later retrieval. Their versions and index locations are saved in `references.json` and each checkpoint. For training alone, omit the reference options. Staged `train-pretrain` and `train-classifier` runs support the same settings.

See the [usage guide](docs/usage.md) for staged training and CPU examples, and [experiment settings](docs/reproducibility.md) for encoding and training details.

## Generation and retrieval

`retrieve` and `generate` return natural protein candidates ranked by sequence-embedding cosine similarity. After the training command above, reuse its reference index directly:

```bash
uv run deephvi generate --checkpoint runs/experiment/fold0/checkpoints/best.pt \
  --pairs data/examples/inputs/new_pairs.csv --target viral --out runs/candidates.json
```

The checkpoint supplies the encoder settings and recorded index. Use `--target human` to search the human side. To start directly from public data, choose `--reference-source human` or `--reference-source viral`; the command handles downloading, preparation and indexing. You can also supply a FASTA with `--reference` or an existing index with `--index`.

Human references come from the complete NCBI RefSeq `GCF_000001405.40` protein FASTA. Viral references include all UniProtKB `taxonomy_id:10239` records. Each library merges identical complete sequences and retains all original accessions.

Commands are in the [retrieval guide](docs/usage.md#generation-and-retrieval).

## Documentation

| Topic | English | 简体中文 |
| --- | --- | --- |
| Installation and CLI workflows | [Usage](docs/usage.md) | [使用说明](docs/zh-CN/usage.md) |
| Reference downloads, training records and inference reuse | [Reference sequences](docs/reference-sequences.md) | [参考序列流程](docs/zh-CN/reference-sequences.md) |
| Data sources, schema, and splits | [Data](docs/data.md) | [数据说明](docs/zh-CN/data.md) |
| Resource archives and checksums | [Resources](docs/resources.md) | [资源说明](docs/zh-CN/resources.md) |
| Models and checkpoint compatibility | [Checkpoints](docs/checkpoints.md) | [模型权重](docs/zh-CN/checkpoints.md) |
| Encoding and experiment settings | [Reproducibility](docs/reproducibility.md) | [复现约定](docs/zh-CN/reproducibility.md) |
| Repository and contribution workflow | [Repository layout](docs/repository-layout.md) | [仓库布局](docs/zh-CN/repository-layout.md) |
| Validation commands and scope | [Validation](docs/validation.md) | [验证方法](docs/zh-CN/validation.md) |
| Licensing and attribution | [Licensing](docs/licensing.md) | [许可说明](docs/zh-CN/licensing.md) |

## License, citation, and support

Original code and author-owned DeepHVI weights use [MIT](LICENSE). Models and data from other sources follow the terms listed in [Third-party notices](THIRD_PARTY_NOTICES.md) and [Licensing](docs/licensing.md). Interaction records come from [HVIDB](https://zzdlab.com/hvidb/); sequence sources include [UniProt](https://www.uniprot.org/) and NCBI RefSeq.

Please cite the DeepHVI paper above when using this implementation, and cite the [data sources](docs/data.md) when using the dataset. Report problems through [GitHub Issues](https://github.com/Xindi-Wang1004/DeepHVI/issues), including the command, error message, and relevant `deephvi doctor` output.
