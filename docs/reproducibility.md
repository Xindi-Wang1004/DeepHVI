# Encoding and reproducibility

[简体中文](zh-CN/reproducibility.md) · [Documentation index](README.md)

This document describes the computation and experiment assumptions needed to interpret DeepHVI outputs. Commands are in [Usage](usage.md); data construction limits are in [Data](data.md).

## Feature encoding

Each protein pair has four modalities: human and viral ESM embeddings, plus human and viral AAindex descriptors. Each AAindex descriptor has 1,200 values from the first 100 residues and 12 properties per residue.

| Workflow | ESM pooling and input limit | Projection | Fusion model |
| --- | --- | --- | --- |
| Released server inference | Attention-mask mean including special tokens; 1024 tokens | Saved `models/server_run/projection.pt`, 320→256 | `server_tmonet_v1` |
| Refactored training | Configurable; default masked mean excluding special tokens, 1024 tokens | Learned and stored in each checkpoint | `deephvi` |

The 1024-token cap includes BOS/EOS, so up to 1022 residue tokens fit. It follows the original GitHub encoder's default. Historical server caches used a 512-token cap; entries with at most 510 residues can be reused, while longer entries are recomputed at the current cap. Server weights were not retrained for this change, and long-sequence predictions may differ from historical 512-token outputs. Original cache files are preserved.

Refactored Gaussian Product of Experts uses inverse variance, `exp(-logvar)`, with prior precision. Refactored checkpoints require model revision 2; older trained weights need retraining. The server compatibility graph preserves its historical computation. See [Checkpoints](checkpoints.md).

## Training defaults and assumptions

`configs/default.yaml` uses an outer 80/20 split and three validation folds in the training portion, ten pretraining epochs, thirty classification epochs, batch size 16, and Adam with learning rate `1e-4`. Classification uses cross-entropy plus a contrastive term weighted by `0.01`.

Details not fully specified by the paper are implementation assumptions: supervised contrastive loss with temperature `0.07`, pretraining reconstruction/cross-reconstruction/KL/adversarial/consistency weights, and AAindex reconstruction scaling. Their exact values are in `configs/default.yaml` and the resolved configuration written to each run. These choices describe the reference implementation and should be reported when comparing experiments.

Validation selects the checkpoint; test data are reserved for final evaluation. Modern protocols remove duplicate full pairs before splitting, but the default pair-level `paper` split can share individual proteins. For generalization to unseen proteins, use connected-protein grouping and inspect overlap counts. Species grouping alone does not ensure human-protein separation.

## Historical comparisons

`configs/server_historical.yaml` records shuffled ten-fold KFold with seed 42, zero-based fold 3, and an inner 90/10 split by row order. The earlier `legacy` profile uses an unshuffled ten-fold convention and reports validation as test. Historical sharing-package predictions used the training loader and live in `data/examples/historical/`; do not compare them directly with held-out test metrics.

## Traceability and metrics

New checkpoints contain model and feature configuration, exact split and data fingerprints, fold, seed, epoch, validation metrics, and environment. Resource manifests bind historical server files by SHA-256. Feature bundles and reference indices require matching encoders and encoding settings when reused.

```bash
uv run deephvi recompute-metrics \
  --predictions runs/experiment/predictions/test_fold0.csv --out runs/recomputed.json
```

Accuracy, macro precision/recall/F1, and ROC-AUC are recomputed from saved labels and positive-class probabilities. The default threshold is 0.5. AUC is undefined for a single-class split and is excluded from valid-fold averages. Save per-sample predictions alongside aggregate metrics.

The hash backend checks software behavior only. Small real-ESM runs verify integration; full paper training, CUDA determinism, and cross-environment equivalence need their own experiments. See [Validation](validation.md) for reproducible checks.
