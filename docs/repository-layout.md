# Repository layout and contributing

[简体中文](zh-CN/repository-layout.md) · [Documentation index](README.md)

## Public files

| Path | Contents |
| --- | --- |
| `src/deephvi/` | Maintained package, CLI, models, feature encoders, AAindex table, and checkpoint compatibility classes |
| `configs/` | Training, inference, historical-comparison, and CPU profiles |
| `tests/` | Unit/integration tests and small reference fixtures |
| `scripts/` | Resource packaging, content auditing, and validation tools |
| `docs/` | English public guides; Chinese equivalents in `docs/zh-CN/` |
| `data/` | Release archive files, inventories, and bilingual instructions at the root |
| `.github/workflows/` | CI checks for supported Python versions and installed-wheel acceptance |

Run `python3 scripts/resource_packages.py extract` to restore local `data/training/`, `data/models/`, and other resource subdirectories. Extracted files are ignored; root archives, manifests, and instructions are published. See [Resources](resources.md).

## Local development materials

Store development notes, intermediate documents, review reports, original source copies, packaging specifications, and build outputs under `dev-docs/`. This directory is ignored and excluded from Python distributions. `AGENTS.md`, agent configuration files, notes, plans, and draft documents are also ignored. Public documentation must explain the released behavior without depending on local materials.

`.gitignore` does not remove files that Git already tracks. Before a release, use the public-tree audit to detect tracked private files as well as unignored new files. If a private file was tracked, remove only its index entry with `git rm --cached PATH` after preserving the local copy. Existing remote history requires separate handling; adding an ignore rule cannot rewrite it.

## Contribution checks

Install development dependencies with `uv sync --frozen --extra dev --extra data`, then run:

```bash
uv run python scripts/check_public_tree.py
uv run ruff check src tests scripts
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run pytest -q
uv build --out-dir dev-docs/builds
```

Keep both language versions aligned when changing public commands or behavior. Add regression coverage for behavioral fixes. Keep generated reports and experiment output local. The wheel contains `src/deephvi`; the source distribution uses an explicit public-file allowlist and private-file exclusions.

The [validation guide](validation.md) explains installed-wheel acceptance. CI retains its standard job output; generated development reports are not uploaded as repository files or artifacts.
