#!/usr/bin/env python3
import argparse
import importlib.util
import json
import sys
from pathlib import Path

REQUIRED_PKGS = ['torch', 'transformers', 'accelerate', 'peft', 'datasets', 'sentencepiece']


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_jsonl(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def check_runtime():
    status = {pkg: bool(importlib.util.find_spec(pkg)) for pkg in REQUIRED_PKGS}
    missing = [pkg for pkg, ok in status.items() if not ok]
    return status, missing


def validate_config(cfg):
    required_top = ['run_name', 'benchmark', 'backbone_checkpoint', 'data', 'training', 'evaluation']
    missing = [k for k in required_top if k not in cfg]
    if missing:
        raise ValueError(f'Missing top-level config keys: {missing}')
    for k in ['train_path', 'val_path', 'dev_analysis_path', 'test_frozen_path']:
        if k in cfg['data'] and not Path(cfg['data'][k]).exists():
            raise FileNotFoundError(f'Data path not found: {cfg["data"][k]}')


def summarize_config(cfg):
    return {
        'run_name': cfg['run_name'],
        'benchmark': cfg['benchmark'],
        'backbone_checkpoint': cfg['backbone_checkpoint'],
        'data_paths': cfg['data'],
        'training_keys': sorted(cfg['training'].keys()),
        'evaluation': cfg['evaluation'],
    }


def summarize_dataset(cfg):
    summary = {}
    for split_key, path in cfg['data'].items():
        rows = load_jsonl(path)
        summary[split_key] = {
            'path': str(Path(path).resolve()),
            'count': len(rows),
            'sample_keys': sorted(rows[0].keys()) if rows else [],
        }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config', help='Path to a phase-1 run config JSON')
    ap.add_argument('--action', choices=['validate', 'summarize', 'check-runtime'], default='summarize')
    args = ap.parse_args()

    cfg = load_json(args.config)

    if args.action == 'check-runtime':
        status, missing = check_runtime()
        print(json.dumps({'runtime': status, 'missing': missing}, indent=2))
        return 0 if not missing else 1

    validate_config(cfg)

    if args.action == 'validate':
        print(json.dumps({'status': 'ok', 'run_name': cfg['run_name']}, indent=2))
        return 0

    if args.action == 'summarize':
        payload = {
            'config': summarize_config(cfg),
            'dataset_summary': summarize_dataset(cfg),
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    return 0


if __name__ == '__main__':
    sys.exit(main())
