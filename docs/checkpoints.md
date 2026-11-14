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

Refactored `deephvi` and `deephvi_pretrain` checkpoints require `model_revision: 2`, which uses inverse-variance Gaussian fusion (`exp(-logvar)`). Older refactored checkpoints are rejected and require retraining. Editing metadata cannot repair learned weights. Historical `server_tmonet_v1` inference has its own compatibility graph and is unaffected by this revision.

## Convert original checkpoints

Original full-module pickle files require explicit opt-in because loading them can execute code. Convert only files from trusted sources:

```bash
uv run deephvi export-legacy \
  --checkpoint data/models/original/server_best_cls_model_fold_3.pt \
  --projection data/models/server_run/projection.pt \
  --cache data/cache/seq_emb_cache.pt --esm-dir data/models/esm2 \
  --out runs/server-export --allow-unsafe-pickle
```

The exporter writes a state-dict checkpoint, encoding settings, resource fingerprints, and a numeric comparison report. Compatibility classes are included in the installed `deephvi.compat` package. Exporting an earlier checkpoint without `--projection` produces a tensor inventory rather than assuming server compatibility. Normal prediction uses `weights_only=True` checkpoints.

## New training and exact resume

Training writes `best_pretrain.pt` or `best.pt`, the latest model `last.pt`, and optimizer/RNG state `last_train_state.pt`. Metadata includes fold, exact split and data fingerprints, seed, feature settings, epoch, validation metrics, and environment.

To resume, retain `last.pt`, its matching `last_train_state.pt`, and the selected best checkpoint in the same directory:

```bash
uv run deephvi train-classifier --config configs/default.yaml \
  --bundle runs/features --splits runs/splits --fold 0 \
  --resume PATH_TO_LAST_PT --epochs 10
```

Replace `PATH_TO_LAST_PT` with the actual saved model path. `--epochs 10` means a total of ten epochs: a run stopped after epoch five continues with six through ten. State is bound to the model's epoch and SHA-256, and stores best score, early-stopping count, history, Python/NumPy/CPU/CUDA RNG states, and DataLoader generator states. Resuming into a new run directory copies the original best checkpoint so that later inferior epochs cannot replace it. Runs already stopped by early stopping return the saved best result.

Older incomplete state files cannot provide exact resume. Parameter-identical CPU resume is covered by tests under unchanged inputs, configuration, and environment; cross-environment and CUDA equivalence require separate validation.

## AAindex reference recovery

The installed package includes the 20×12 AAindex table. To verify it against the saved reference vectors:

```bash
uv run python scripts/recover_aaindex_table.py \
  --source tests/fixtures/aaindex_legacy_examples.txt \
  --out dev-docs/reports/aaindex.csv --report dev-docs/reports/aaindex-recovery.json
```
