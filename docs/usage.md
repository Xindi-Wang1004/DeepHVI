# Usage

[简体中文](zh-CN/usage.md) · [Documentation index](README.md)

Run commands from the repository root after installing Python 3.10+ and dependencies with `uv sync --frozen --extra dev --extra data`. For pretrained inference, first restore resources using `python3 scripts/resource_packages.py extract`. With a pip installation, use `deephvi` in place of `uv run deephvi`.

```bash
uv run deephvi --help
uv run deephvi predict --help
```

Common options include `--config FILE`, repeated `--set KEY=VALUE`, `--seed`, `--device cpu|cuda|auto`, and `--resources-dir DIR`. Relative paths are resolved from the current working directory. The CLI detects `data/manifest.json` in that directory automatically; elsewhere, pass the resource directory explicitly.

## CUDA

The repository's uv lockfile selects CPU PyTorch. Create a separate environment for CUDA, install a build selected by the [official PyTorch installer](https://pytorch.org/get-started/locally/), then install this project:

```bash
python3 -m venv .venv-cuda
source .venv-cuda/bin/activate
python -m pip install --upgrade pip
# Run the CUDA installation command from the PyTorch installer here.
python -m pip install '.[data]'
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
deephvi predict --resources-dir data --pairs input.csv --out predictions.csv --device cuda
```

On Windows, activate with `.venv-cuda\Scripts\activate`. Use `deephvi` directly in this environment; `uv sync` and `uv run` select the repository's CPU environment. The shell environment-variable examples below use POSIX syntax; in PowerShell, set `$env:OMP_NUM_THREADS='2'` and `$env:MKL_NUM_THREADS='2'` separately.

## Prediction

```bash
uv run deephvi predict --resources-dir data \
  --pairs data/examples/inputs/new_pairs.csv --out runs/predictions.csv --device cpu
```

Input columns are `human_seq,viral_seq`, optionally with `pair_id`; aliases are documented in [Data](data.md). Output includes `pair_id,probability,prediction` and sequences. To change the decision threshold, add `--set eval.threshold=0.7`.

## Small CPU examples

Without resources:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run deephvi demo \
  --backend hash --n-pairs 30 --epochs 1 --device cpu --out runs/demo
```

With real ESM-2 resources:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run deephvi reproduce \
  --resources-dir data --config configs/cpu_smoke_esm.yaml \
  --pairs data/examples/inputs/cpu_smoke_pairs.csv --limit 18 --epochs 1 \
  --out runs/cpu-example --device cpu
```

Both commands check the pipeline. Synthetic or small-subset metrics are not paper benchmarks.

## Data and features

```bash
uv run deephvi inspect-data --input data/training/SH_SV_feature_with_vectors_0.5.csv --count
uv run deephvi prepare-features --resources-dir data --config configs/default.yaml \
  --pairs data/training/SH_SV_feature_with_vectors_0.5.csv --out runs/features --device cpu
uv run deephvi make-splits --config configs/default.yaml \
  --pairs data/training/SH_SV_feature_with_vectors_0.5.csv --out runs/splits
```

Use the same input, configuration, and sample limit for features and splits. Modern protocols deduplicate exact pairs; historical protocols preserve rows. Feature reuse checks records, encoder weights, pooling, length, projection, and AAindex table content. After changing an encoder, use a fresh directory or `prepare-features --overwrite`.

## Training

The unified workflow is:

```bash
uv run deephvi reproduce --resources-dir data --config configs/default.yaml \
  --pairs data/training/SH_SV_feature_with_vectors_0.5.csv --out runs/experiment
```

It audits records, builds features and splits, pretrains and fine-tunes each fold, then saves evaluation outputs and `checkpoint_map.json`. `--epochs` overrides both stages. `reproduce --limit N` alternates classes after deduplication for a small test; staged commands instead read the first N rows. To compare workflows, omit the limit or give both the same saved subset CSV.

To train in separate stages after the feature and split commands above:

```bash
uv run deephvi train-pretrain --config configs/default.yaml \
  --bundle runs/features --splits runs/splits --fold 0 --epochs 1
uv run deephvi train-classifier --config configs/default.yaml \
  --bundle runs/features --splits runs/splits --fold 0 --epochs 1 \
  --pretrain-checkpoint PATH_TO_BEST_PRETRAIN_PT
```

Replace `PATH_TO_BEST_PRETRAIN_PT` with the path printed by pretraining. Staged runs write to `runs/<name>/<timestamp>-<id>/`. Resume with `--resume PATH_TO_LAST_PT`; `--epochs` specifies the target total epoch count. Preserve the matching optimizer state and best checkpoint as described in [Checkpoints](checkpoints.md).

## Evaluation and metric recomputation

```bash
uv run deephvi evaluate --checkpoint-map runs/experiment/checkpoint_map.json \
  --bundle runs/experiment/features --splits runs/experiment/splits \
  --fold all --out runs/evaluation
uv run deephvi recompute-metrics \
  --predictions runs/evaluation/predictions/test_fold0.csv --out runs/metrics.json
```

For one fold, replace `--checkpoint-map ... --fold all` with `--checkpoint PATH_TO_BEST_PT --fold 0`. Mapping files use `{"0":"fold0/checkpoints/best.pt","1":"fold1/checkpoints/best.pt"}`; paths are relative to the mapping file. Evaluation validates each checkpoint's fold and split fingerprint.

## Similarity retrieval

```bash
uv run deephvi retrieve --resources-dir data \
  --pairs data/examples/inputs/new_pairs.csv --reference reference.fasta \
  --target viral --top-k 5 --out runs/retrieval.json
```

Provide your own reference FASTA, or use `--index DIR` to reuse an index built with the same encoder and encoding settings. `--target human` searches the human side. `generate` uses the same retrieval interface; released resources do not include the conditional decoder needed for the paper's full generative task.

## Diagnostics and configuration

```bash
uv run deephvi verify-resources --resources-dir data
uv run deephvi doctor --resources-dir data --write dev-docs/reports/doctor.md
uv run deephvi config show --config configs/default.yaml
```

Resource maintenance is covered in [Resources](resources.md); checkpoint conversion is covered in [Checkpoints](checkpoints.md).
