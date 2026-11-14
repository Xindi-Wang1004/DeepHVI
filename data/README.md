# DeepHVI data and model resources

[简体中文](README.zh-CN.md) · [Project guide](../README.md)

Release resources are distributed as `resources-*.tar.gz`, each below 100 MiB. Git tracks root archives, manifests, and instructions; extracted resource directories are ignored.

From the repository root, after installing dependencies with `uv sync --frozen --extra data`:

```bash
python3 scripts/resource_packages.py verify
python3 scripts/resource_packages.py extract
uv run deephvi verify-resources --resources-dir data
uv run deephvi predict --resources-dir data \
  --pairs data/examples/inputs/new_pairs.csv --out runs/predictions.csv --device cpu
```

| Directory | Contents |
| --- | --- |
| `training/` | Pretraining CSV: 33,468 rows; classification CSV: 36,465 rows before deduplication |
| `auxiliary/` | Partial features, labelled test pairs, and candidate sequences |
| `models/esm2/` | ESM-2 encoder and tokenizer |
| `models/server_run/` | Classifier, pretraining model, and matching projection |
| `models/original/` | Original checkpoints for explicit conversion |
| `cache/` | Historical sequence embeddings |
| `examples/` | Example inputs and expected/historical outputs |
| `configs/`, `splits/` | Configurations, fixed indices, and ordered split inputs |
| `provenance/` | Sources and dataset statistics |
| `archives/` | Original assembled data archive |

Use `training/SH_SV_feature_with_vectors.csv` for the distributed pretraining input and `training/SH_SV_feature_with_vectors_0.5.csv` for classification. The latter contains 24,312 positive and 12,153 negative original rows, or 34,195 unique pairs.

`manifest.json` and `SHA256SUMS` verify resource files; `PACKAGES.json` and `PACKAGES.sha256` verify archive membership and contents. Archives total about 328 MiB, expanding to about 1.76 GiB. Server inference uses a 1024-token cap including BOS/EOS, attention-mask mean, and the saved 320→256 projection. Long entries truncated by the historical 512-token cache are recomputed without overwriting that original cache.

Interaction records originate from HVIDB and sequences from UniProt. Their applicable terms are separate from the code's MIT license. Read [Data](../docs/data.md), [Resources](../docs/resources.md), and [Licensing](../docs/licensing.md) for provenance, commands, and license scope. Original [Baidu copy](https://pan.baidu.com/s/1Ygk06WfLfOzHvYW7N4y6qQ): extraction code `g9cn`.
