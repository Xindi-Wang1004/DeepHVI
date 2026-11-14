#!/usr/bin/env bash
# End-to-end CPU smoke test: synthetic data + deterministic hash encoder.
#
# This validates that pretraining, fine-tuning, evaluation, prediction and
# retrieval are wired together. It does NOT reproduce paper results: the hash
# backend is not ESM-2 and the data are synthetic.
set -euo pipefail
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

cd "$(dirname "$0")/.."

if command -v deephvi >/dev/null 2>&1; then
  DEEPHVI=(deephvi)
elif [[ -x .venv/bin/deephvi ]]; then
  DEEPHVI=(.venv/bin/deephvi)
elif command -v uv >/dev/null 2>&1; then
  DEEPHVI=(uv run deephvi)
else
  echo "deephvi not found. Run 'uv sync' or activate .venv first." >&2
  exit 1
fi

OUT="${1:-runs/smoke}"

"${DEEPHVI[@]}" demo \
  --config configs/demo_cpu.yaml \
  --backend hash \
  --n-pairs 64 \
  --epochs 1 \
  --device cpu \
  --out "${OUT}"

"${DEEPHVI[@]}" evaluate \
  --config configs/demo_cpu.yaml \
  --checkpoint "${OUT}/checkpoints/best.pt" \
  --bundle "${OUT}/features" \
  --splits "${OUT}/splits" \
  --fold 0 \
  --split test \
  --out "${OUT}/eval"

"${DEEPHVI[@]}" predict \
  --config configs/demo_cpu.yaml \
  --checkpoint "${OUT}/checkpoints/best.pt" \
  --pairs "${OUT}/data/pairs.csv" \
  --out "${OUT}/predictions/new_pairs.csv"

echo "Smoke test complete: ${OUT}"
