# Validation

[简体中文](zh-CN/validation.md) · [Documentation index](README.md)

Run these checks from the repository root to verify the code, resources, and installed package.

## Routine checks

After `uv sync --frozen --extra dev --extra data`, run from the repository root:

```bash
uv run python scripts/check_public_tree.py
uv run ruff check src tests scripts
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run pytest -q
```

The public-tree audit checks tracked files and unignored new files, rejecting internal documents, model files outside approved archives, and other generated artifacts. Tests cover record parsing and deduplication, splits and fold provenance, Gaussian fusion, metrics, feature compatibility, archive handling, resume state, retrieval, and CLI workflows.

Reference integration tests use local HTTP fixtures and the hash demo encoder to exercise downloads, index reuse, recovery, training records, staged inheritance and checkpoint-based retrieval. They compare model parameters, pair/split fingerprints and classification outputs across runs with and without reference preparation.

## Real resources

Restore the archives, then audit integrity and run the server comparison:

```bash
python3 scripts/resource_packages.py extract
uv run python scripts/verify_resource_layout.py --resources-dir data --config-dir configs
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run python scripts/verify_server_resources.py \
  --resources-dir data --out dev-docs/validation/server
```

The server check compares cached/recomputed embeddings, reference encoder outputs, boundary token lengths, novel-sequence encoding, and CLI predictions. It uses `atol=2e-5, rtol=2e-5` for numerical comparisons. Reports record the environment and observed errors.

## Installed-wheel acceptance

```bash
python3 scripts/verify_release.py --python 3.12 --out dev-docs/validation/release
```

This requires `uv` on PATH and enough space for another extracted resource copy and environment. The script audits public files, creates an independent snapshot, restores resources, installs locked dependencies, builds and installs the wheel, and verifies imports from `site-packages`. It then runs tests, resource audits, real ESM-2 server prediction, and an 18-pair three-fold training check with one epoch per stage.

Choose a new `--out` directory for each run. Temporary files, logs, builds and the report are saved together there. CI tests Python 3.10 and 3.12 and runs installed-wheel acceptance in a separate job.

The checks run on Linux CPU and cover the hash demo, real ESM-2 inference and training, and parameter consistency after resuming.
