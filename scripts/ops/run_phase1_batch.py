#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path('/root/workspace/jepa')
TRAIN = ROOT / 'scripts' / 'phase1_train.py'

PRESETS = {
    'vast-gsm8k-pilot': [
        'configs/phase1/runs/pilot_gsm8k/gsm8k_gemma4e2b_lm_pilot.json',
        'configs/phase1/runs/pilot_gsm8k/gsm8k_gemma4e2b_coupled_pilot.json',
        'configs/phase1/runs/pilot_gsm8k/gsm8k_gemma4e2b_decoupled_pilot.json',
    ],
    'lambda-phase1-full': [
        'configs/phase1/runs/gsm8k_gemma4e2b_lm.json',
        'configs/phase1/runs/gsm8k_gemma4e2b_coupled.json',
        'configs/phase1/runs/gsm8k_gemma4e2b_decoupled.json',
        'configs/phase1/runs/regexeval_gemma4e2b_lm.json',
        'configs/phase1/runs/regexeval_gemma4e2b_coupled.json',
        'configs/phase1/runs/regexeval_gemma4e2b_decoupled.json',
        'configs/phase1/runs/arc_challenge_gemma4e2b_lm.json',
        'configs/phase1/runs/arc_challenge_gemma4e2b_coupled.json',
        'configs/phase1/runs/arc_challenge_gemma4e2b_decoupled.json',
        'configs/phase1/runs/hellaswag_gemma4e2b_lm.json',
        'configs/phase1/runs/hellaswag_gemma4e2b_coupled.json',
        'configs/phase1/runs/hellaswag_gemma4e2b_decoupled.json',
    ],
    'lambda-phase1-broad': [
        'configs/phase1/runs/gsm8k_gemma4e2b_lm.json',
        'configs/phase1/runs/gsm8k_gemma4e2b_coupled.json',
        'configs/phase1/runs/gsm8k_gemma4e2b_decoupled.json',
        'configs/phase1/runs/regexeval_gemma4e2b_lm.json',
        'configs/phase1/runs/regexeval_gemma4e2b_coupled.json',
        'configs/phase1/runs/regexeval_gemma4e2b_decoupled.json',
        'configs/phase1/runs/arc_challenge_gemma4e2b_lm.json',
        'configs/phase1/runs/arc_challenge_gemma4e2b_coupled.json',
        'configs/phase1/runs/arc_challenge_gemma4e2b_decoupled.json',
        'configs/phase1/runs/hellaswag_gemma4e2b_lm.json',
        'configs/phase1/runs/hellaswag_gemma4e2b_coupled.json',
        'configs/phase1/runs/hellaswag_gemma4e2b_decoupled.json',
        'configs/phase1/runs/mmlu_gemma4e2b_lm.json',
        'configs/phase1/runs/mmlu_gemma4e2b_coupled.json',
        'configs/phase1/runs/mmlu_gemma4e2b_decoupled.json',
    ],
}


def run_name_from_config(path: Path) -> str:
    payload = json.loads(path.read_text())
    return payload['run_name']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preset', choices=sorted(PRESETS), required=True)
    ap.add_argument('--output-root', default=str(ROOT / 'runs'))
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    paths = [ROOT / p for p in PRESETS[args.preset]]
    if args.limit > 0:
        paths = paths[:args.limit]

    commands = []
    for cfg in paths:
        run_name = run_name_from_config(cfg)
        out = Path(args.output_root) / run_name
        cmd = ['python', str(TRAIN), str(cfg), '--output-dir', str(out)]
        commands.append({'config': str(cfg), 'run_name': run_name, 'command': cmd})

    if args.dry_run:
        print(json.dumps({'preset': args.preset, 'commands': commands}, indent=2))
        return

    for item in commands:
        subprocess.run(item['command'], check=True)


if __name__ == '__main__':
    main()
