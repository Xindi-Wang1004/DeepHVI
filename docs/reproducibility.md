# Encoding and reproducibility

[简体中文](zh-CN/reproducibility.md) · [Documentation index](README.md)

This guide describes feature encoding and experiment settings. Commands are in [Usage](usage.md); sources and data preparation are in [Data](data.md).

## Feature encoding

Each protein pair has four modalities: human and viral ESM embeddings, plus human and viral AAindex descriptors. Each AAindex descriptor has 1,200 values from the first 100 residues and 12 properties per residue.

| Workflow | ESM pooling and input limit | Projection | Fusion model |
| --- | --- | --- | --- |
| Released server inference | Attention-mask mean including special tokens; 1024 tokens | Saved `models/server_run/projection.pt`, 320→256 | `server_tmonet_v1` |
| Refactored training | Configurable; default masked mean excluding special tokens, 1024 tokens | Learned and stored in each checkpoint | `deephvi` |

The 1024-token cap includes BOS/EOS and fits up to 1022 residue tokens. Historical server caches used 512 tokens; sequences up to 510 residues reuse those cached values, while longer sequences are encoded at the current limit. This changes the input representation of long proteins, so comparisons with historical predictions should record the token limit. The original cache stays available for those comparisons.

Refactored Gaussian Product of Experts uses inverse variance, `exp(-logvar)`, with prior precision. Refactored checkpoints require model revision 2; older trained weights need retraining. The server compatibility graph preserves its historical computation. See [Checkpoints](checkpoints.md).

## Training settings

`configs/default.yaml` uses an outer 80/20 split and three validation folds in the training portion, ten pretraining epochs, thirty classification epochs, batch size 16, and Adam with learning rate `1e-4`. Classification uses cross-entropy plus a contrastive term weighted by `0.01`.

The implementation uses supervised contrastive loss with temperature `0.07`. Pretraining combines reconstruction, cross-reconstruction, KL, adversarial, and consistency terms, with scaling for AAindex reconstruction. Exact weights are in `configs/default.yaml` and each run's saved configuration.

Validation selects the checkpoint; test data are used for final evaluation. Modern protocols deduplicate full pairs before splitting. Use `paper` for pair-level splits, or `group` with `union_find` for connected-protein splits and `virus_species` for species-level splits.

Training and inference share reference libraries prepared with the sequence feature encoder. `--reference-source both` records their versions and index locations in `references.json` and checkpoint `reference_libraries`. Pair tables supply model training data, and reference libraries supply retrieval candidates. Training restores its random seed before model initialization, giving the same initialization after either building or reusing a library. The [reference workflow](reference-sequences.md) describes this handoff.

## Historical comparisons

`configs/server_historical.yaml` records shuffled ten-fold KFold with seed 42, zero-based fold 3, and an inner 90/10 split by row order. The earlier `legacy` profile uses an unshuffled ten-fold convention and reports validation as test. Historical training-loader predictions are stored in `data/examples/historical/`.

## Traceability and metrics

Each checkpoint records the model and feature configuration, data and split fingerprints, fold, seed, epoch, validation metrics and environment. Resource manifests identify the server files by SHA-256. Feature bundles and reference indices also record encoder settings, allowing reuse across runs with the same encoding.

```bash
uv run deephvi recompute-metrics \
  --predictions runs/experiment/predictions/test_fold0.csv --out runs/recomputed.json
```

The command recomputes accuracy, macro precision/recall/F1 and ROC-AUC from saved labels and positive-class probabilities. The default classification threshold is 0.5. AUC averages include folds containing both classes.

Runs save per-sample predictions and aggregate metrics. See [Validation](validation.md) for test and model checks.
