# Repository layout and contributing

[简体中文](zh-CN/repository-layout.md) · [Documentation index](README.md)

## Public files

| Path | Contents |
| --- | --- |
| `src/deephvi/` | Maintained package, CLI, models, feature encoders, AAindex table, and checkpoint compatibility classes |
| `configs/` | Training, inference, historical-comparison, and CPU profiles |
| `tests/` | Unit/integration tests and small reference fixtures |
| `scripts/` | Reference downloads and preparation, resource packaging, audits and validation |
| `docs/` | English public guides; Chinese equivalents in `docs/zh-CN/` |
| `data/` | Release archive files, inventories, and bilingual instructions at the root |
| `.github/workflows/` | CI checks for supported Python versions and installed-wheel acceptance |

Run `python3 scripts/resource_packages.py extract` to restore local `data/training/`, `data/models/`, and other resource subdirectories. Extracted files are ignored; root archives, manifests, and instructions are published. See [Resources](resources.md).

## Local development materials

Use `dev-docs/` for development notes, review reports, source copies, packaging specifications and build outputs. It is ignored by Git and excluded from Python distributions. Public guides live in `docs/`, and experiment output belongs in `runs/`.

The public-tree audit checks tracked files and new files before release. To move an already tracked development file into local storage, preserve its local copy and remove the index entry with `git rm --cached PATH`.

## Contribution checks

Install development dependencies with `uv sync --frozen --extra dev --extra data`, then run:

```bash
uv run python scripts/check_public_tree.py
uv run ruff check src tests scripts
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run pytest -q
uv build --out-dir dev-docs/builds
```

Keep both language versions aligned when changing public commands or behavior. Add regression coverage for behavioral fixes. Keep generated reports and experiment output local. The wheel contains `src/deephvi`; the source distribution uses an explicit public-file allowlist and private-file exclusions.

The [validation guide](validation.md) describes installed-wheel acceptance and its reports. CI results are available in the job output.
