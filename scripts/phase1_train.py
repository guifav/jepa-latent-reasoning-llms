#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from jepa_phase1.configs import load_run_config
from jepa_phase1.train import run_training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config', help='Run config JSON path')
    ap.add_argument('--output-dir', default='')
    args = ap.parse_args()

    cfg = load_run_config(args.config)
    out = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / 'runs' / cfg.run_name
    summary = run_training(cfg, out)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
