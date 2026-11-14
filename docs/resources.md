# Resource archives

[简体中文](zh-CN/resources.md) · [Documentation index](README.md)

The resource root is `data/`. Git distributes its root documentation, manifests, and `resources-*.tar.gz` archives. Extracted subdirectories are ignored by Git. Archives total about 328 MiB and expand to about 1.76 GiB; reserve at least 8 GiB including dependencies and working files.

## Restore and verify

From the repository root:

```bash
python3 scripts/resource_packages.py verify
python3 scripts/resource_packages.py extract
uv run deephvi verify-resources --resources-dir data
uv run python scripts/verify_resource_layout.py --resources-dir data --config-dir configs
```

The first two commands use only the Python standard library. Extraction verifies archive and member checksums before restoring targets; an existing file with different content causes a conflict instead of replacement. Use a new directory or move conflicting local files aside before retrying. The layout audit also checks packaged configurations and ordered inputs for saved splits.

`manifest.json` lists resource paths, roles, sizes, and SHA-256 values. `SHA256SUMS` covers that inventory and the manifest. `PACKAGES.json` maps archives to inventory members and binds the manifest checksum; `PACKAGES.sha256` covers archives and their index. The authoritative inventory is [data/manifest.json](../data/manifest.json).

## Contents

Paths below are relative to `data/`:

| Path | Purpose |
| --- | --- |
| `training/` | Pretraining CSV (33,468 rows) and classification CSV (36,465 rows before deduplication) |
| `auxiliary/` | Partial feature table, labelled pairs, and candidate sequence tables |
| `models/esm2/` | ESM-2 weights, configuration, tokenizer, and upstream model card |
| `models/server_run/classifier.pt` | Converted server classifier |
| `models/server_run/pretrain.pt` | Server TMO-Net pretraining weights |
| `models/server_run/projection.pt` | Matching 320→256 projection |
| `models/original/` | Original checkpoints for explicit conversion |
| `cache/seq_emb_cache.pt` | 8,259 historical entries; 4,797 can be reused directly at the current encoding limit |
| `examples/inputs/` | Cached, novel-sequence, and CPU training examples |
| `examples/expected_outputs/` | Reference prediction outputs |
| `examples/historical/` | Historical training-loader predictions for comparison |
| `configs/`, `splits/cpu_smoke/` | Matching configurations, fixed splits, and ordered input pairs |
| `provenance/` | Sources and training-data statistics |
| `archives/data.7z` | Original assembled data archive |

Data sources and their licensing are described in [Data](data.md) and [Licensing](licensing.md). Original distribution: [Baidu](https://pan.baidu.com/s/1Ygk06WfLfOzHvYW7N4y6qQ), code `g9cn`.

## Paths and inference

```bash
uv run deephvi predict --resources-dir data \
  --pairs data/examples/inputs/new_pairs.csv --out runs/predictions.csv --device cpu
```

Path precedence is explicit CLI option, explicit configuration, then resource-directory defaults. Overrides are `--checkpoint`, `--esm-dir`, `--projection`, and `--cache`; configuration keys are `predict.checkpoint` and `resources.paths.esm/projection/cache`. `data/manifest.json` is detected in the current directory automatically. When running elsewhere, pass an absolute `--resources-dir` or a path relative to that working directory.

The server encoder uses 1024 tokens including BOS/EOS, attention-mask mean pooling including those special tokens, and its saved projection. Historical 512-token cache entries longer than 510 residues are recomputed on first use; the original cache file stays intact. Weights were not retrained for this limit, so long-sequence predictions can differ from historical outputs.

## Public reference resources

The reference workflow downloads and prepares libraries under `data/reference/`. Training and inference share the raw FASTA, accession catalogs and encoder-specific indices there. Keep recorded indices alongside the corresponding checkpoints. Full libraries use additional disk, memory and encoding time beyond the model archives described above. Commands and version management are in [Reference sequences](reference-sequences.md).

## Building archives

For an existing verified resource directory:

```bash
python3 scripts/resource_packages.py build
```

The builder groups most files by top-level directory and model files by the second directory level. It splits groups into archives below 95 MiB; a single compressed file exceeding that limit must be handled separately. Archive outputs are deterministic for the same inventory. Move an old package set to ignored local storage before rebuilding changed resources, then regenerate and verify the manifest and checksum files together.

To collect a new resource directory from declared sources:

```bash
uv run python scripts/build_resource_bundle.py --manifest collection.json --out data
```

Example collection specification:

```json
{
  "version": "DeepHVI_resources_v1",
  "files": [
    {"source": "inputs/classifier.pt", "path": "models/server_run/classifier.pt", "role": "classifier"}
  ]
}
```

`source` is relative to the collection specification; `path` is relative to the destination. The collector checks inputs and refuses conflicting targets. Keep collection specifications, build logs, and working copies under ignored `dev-docs/`; only reviewed release files belong in the public tree.
