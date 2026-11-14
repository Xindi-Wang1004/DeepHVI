# Validation

[简体中文](zh-CN/validation.md) · [Documentation index](README.md)

Validation checks software behavior, resource consistency, and compatibility with the released models. It does not establish reproduction of the paper's full experimental metrics. Run the commands below against the revision you intend to use; historical development reports are not a substitute for current checks.

## Routine checks

After `uv sync --frozen --extra dev --extra data`, run from the repository root:

```bash
uv run python scripts/check_public_tree.py
uv run ruff check src tests scripts
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run pytest -q
```

The public-tree audit checks tracked files and unignored new files, rejecting internal documents, model files outside approved archives, and other generated artifacts. Tests cover record parsing and deduplication, splits and fold provenance, Gaussian fusion, metrics, feature compatibility, archive handling, resume state, retrieval, and CLI workflows.

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

Choose a new `--out` directory for each run: nonempty output directories are rejected. Temporary files, logs, builds, and the report stay under that ignored directory. CI tests Python 3.10 and 3.12 and runs installed-wheel acceptance in a separate job.

## Interpretation

Hash-backed checks exercise the pipeline without a protein language model. Small real-ESM runs verify that training and inference execute with the bundled resources. CPU resume tests compare uninterrupted and resumed parameters in a fixed environment. Full training metrics, CUDA reproducibility, Windows/macOS behavior, and every minimum dependency combination need separate validation. Dataset reconstruction is also limited by unavailable original database snapshots; see [Data](data.md).
