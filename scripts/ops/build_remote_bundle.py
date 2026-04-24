#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

ROOT = Path('/root/workspace/jepa')
DEFAULT_OUT = ROOT / 'deploy' / 'remote' / 'jepa_remote_bundle.tar.gz'
INCLUDE = [
    'requirements-phase1.txt',
    'requirements-remote.txt',
    'src',
    'scripts',
    'configs/phase1',
    'data/gsm8k/phase1',
    'data/gsm8k/phase1_views',
    'data/regexeval/phase1',
    'data/regexeval/phase1_views',
    'data/arc_challenge/phase1',
    'data/arc_challenge/phase1_views',
    'data/hellaswag/phase1',
    'data/hellaswag/phase1_views',
    'data/mmlu/phase1',
    'data/mmlu/phase1_views',
    'analysis/article_framing.md',
    'analysis/benchmark_suite.md',
    'analysis/mcq_support_benchmarks_protocol.md',
    'analysis/phase1_status.md',
    'analysis/phase1_runtime_note.md',
    'deploy/remote/Dockerfile',
    'deploy/provider_playbooks',
]


def should_skip(path: Path) -> bool:
    if '__pycache__' in path.parts:
        return True
    if path.suffix in {'.pyc', '.pyo'}:
        return True
    return False


def add_path(tf: tarfile.TarFile, path: Path):
    if should_skip(path):
        return
    if path.is_dir():
        for child in sorted(path.iterdir()):
            add_path(tf, child)
        return
    arcname = path.relative_to(ROOT)
    tf.add(path, arcname=str(arcname))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default=str(DEFAULT_OUT))
    args = ap.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    added = []
    with tarfile.open(out, 'w:gz') as tf:
        for rel in INCLUDE:
            path = ROOT / rel
            if not path.exists():
                raise FileNotFoundError(path)
            add_path(tf, path)
            added.append(rel)

    payload = {
        'bundle_path': str(out),
        'entries': added,
        'size_bytes': out.stat().st_size,
    }
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
