# Models and checkpoints

[简体中文](zh-CN/checkpoints.md) · [Documentation index](README.md)

Restore resources with `python3 scripts/resource_packages.py extract` before using the pretrained files below. All paths in the table are relative to `data/models/`.

| File | Purpose |
| --- | --- |
| `esm2/model.safetensors` | ESM-2 encoder; requires adjacent configuration and tokenizer files |
| `server_run/classifier.pt` | Server classifier, architecture `server_tmonet_v1` |
| `server_run/pretrain.pt` | Server TMO-Net pretraining weights |
| `server_run/projection.pt` | Matching 320→256 sequence projection |
| `original/server_best_cls_model_fold_3.pt` | Original server classification checkpoint |
| `original/server_best_pretrain_fold3_epoch9.pt` | Original server pretraining checkpoint |
| `original/initial_release_classifier.pt` | Earlier released classification checkpoint |

## Inference and compatibility

```bash
uv run deephvi predict --resources-dir data --pairs input.csv --out predictions.csv --device cpu
```

The CLI constructs the architecture identified by checkpoint `model_name`. Server inference uses its separately saved projection; newly trained refactored models contain their own projection. The server pretraining architecture differs from the refactored training architecture: `train-classifier --pretrain-checkpoint` expects a checkpoint produced by the matching `train-pretrain` implementation.

Current `deephvi` and `deephvi_pretrain` checkpoints use `model_revision: 2`, with inverse-variance Gaussian fusion (`exp(-logvar)`). Models from an earlier revision need retraining with the current implementation. Historical `server_tmonet_v1` checkpoints use their dedicated inference architecture.

## Convert original checkpoints

Original full-module pickle files require explicit opt-in because loading them can execute code. Convert only files from trusted sources:

```bash
uv run deephvi export-legacy \
  --checkpoint data/models/original/server_best_cls_model_fold_3.pt \
  --projection data/models/server_run/projection.pt \
  --cache data/cache/seq_emb_cache.pt --esm-dir data/models/esm2 \
  --out runs/server-export --allow-unsafe-pickle
```

The exporter writes a state-dict checkpoint, encoding settings, resource fingerprints and a numerical comparison report. Prediction loads the exported checkpoint with `weights_only=True`; compatibility classes are included in `deephvi.compat`. For an earlier checkpoint, export without `--projection` to inspect its tensor inventory first.

## New training and exact resume

Training writes `best_pretrain.pt` or `best.pt`, the latest model `last.pt`, and optimizer/RNG state `last_train_state.pt`. Metadata includes fold, exact split and data fingerprints, seed, feature settings, epoch, validation metrics, and environment.

To resume, retain `last.pt`, its matching `last_train_state.pt`, and the selected best checkpoint in the same directory:

```bash
uv run deephvi train-classifier --config configs/default.yaml \
  --bundle runs/features --splits runs/splits --fold 0 \
  --resume PATH_TO_LAST_PT --epochs 10
```

Replace `PATH_TO_LAST_PT` with the actual saved model path. `--epochs 10` means a total of ten epochs: a run stopped after epoch five continues with six through ten. State is bound to the model's epoch and SHA-256, and stores best score, early-stopping count, history, Python/NumPy/CPU/CUDA RNG states, and DataLoader generator states. Resuming into a new run directory copies the original best checkpoint so that later inferior epochs cannot replace it. Runs already stopped by early stopping return the saved best result.

Keeping these files together restores the optimizer, random generators and model selection state. CPU tests compare resumed and uninterrupted model parameters.

## Reference libraries recorded during training

Training with `--reference-source human|viral|both` saves a `references.json` file and adds `reference_libraries` to each checkpoint. These records identify the source versions, sequence and encoder fingerprints, index locations and completion records. Classification training inherits them from its pretraining checkpoint, and resume continues with the same library versions. Keep the recorded indices with the run’s resources.

Use `retrieve` or `generate` with `--checkpoint PATH --target human|viral` to load the saved encoder settings and search the recorded index. After moving an index directory, add `--index NEW_DIR`. Commands, file layouts and version management are in [Reference sequences](reference-sequences.md).

## AAindex reference recovery

The installed package includes the 20×12 AAindex table. To verify it against the saved reference vectors:

```bash
uv run python scripts/recover_aaindex_table.py \
  --source tests/fixtures/aaindex_legacy_examples.txt \
  --out dev-docs/reports/aaindex.csv --report dev-docs/reports/aaindex-recovery.json
```
