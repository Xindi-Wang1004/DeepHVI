"""Verify a public snapshot, installed wheel and real resources in an isolated directory."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from check_public_tree import audit


def verify_release(root: Path, out: Path, python: str):
    root, out = root.resolve(), out.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'Refusing to reuse nonempty validation directory: {out}')
    files, problems = audit(root)
    if problems:
        raise ValueError('\n'.join(problems))
    snapshot = out / 'public'
    logs = out / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    temporary = out / 'tmp'
    temporary.mkdir()
    for relative in files:
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, target)
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', UV_PYTHON=python,
               TMPDIR=str(temporary), TMP=str(temporary), TEMP=str(temporary))
    # A caller's activated environment/PYTHONPATH must not substitute local code.
    env.pop('VIRTUAL_ENV', None)
    env.pop('PYTHONPATH', None)
    env.pop('UV_PROJECT_ENVIRONMENT', None)
    completed = []

    def run(label, command):
        print(f'Verifying {label}', flush=True)
        with (logs / f'{label}.log').open('w') as handle:
            subprocess.run(command, cwd=snapshot, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
        completed.append(label)

    run('extract', [sys.executable, 'scripts/resource_packages.py', 'extract'])
    run('install', ['uv', 'sync', '--frozen', '--python', python, '--extra', 'dev', '--extra', 'data'])
    run('build', ['uv', 'build', '--out-dir', str(out / 'builds')])
    interpreter = snapshot / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    wheel = next((out / 'builds').glob('*.whl'))
    run('install-wheel', ['uv', 'pip', 'install', '--python', str(interpreter), '--no-deps', '--reinstall', str(wheel)])
    run('import-wheel', [str(interpreter), '-c',
        'import deephvi, sys; from pathlib import Path; '
        'p=Path(deephvi.__file__).resolve(); print(p); '
        'assert p.is_relative_to(Path(sys.prefix).resolve())'])
    run('wheel-tests', [str(interpreter), '-m', 'pytest', '-q'])
    run('resource-layout', [str(interpreter), 'scripts/verify_resource_layout.py', '--resources-dir', 'data', '--config-dir', 'configs'])
    run('server-prediction', [str(interpreter), 'scripts/verify_server_resources.py', '--resources-dir', 'data', '--out', 'runs/server'])
    run('esm-training', [str(interpreter), '-m', 'deephvi.cli', 'reproduce', '--resources-dir', 'data',
        '--config', 'configs/cpu_smoke_esm.yaml', '--pairs', 'data/examples/inputs/cpu_smoke_pairs.csv',
        '--limit', '18', '--epochs', '1', '--device', 'cpu', '--out', 'runs/esm-smoke'])
    report = {'public_files': len(files), 'python': python, 'checks': completed,
              'scope': 'Installed-wheel CPU acceptance; not full paper reproduction'}
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--out', type=Path, default=Path('dev-docs/release-validation'))
    parser.add_argument('--python', default=f'{sys.version_info.major}.{sys.version_info.minor}')
    args = parser.parse_args()
    verify_release(args.root, args.out, args.python)


if __name__ == '__main__':
    main()
