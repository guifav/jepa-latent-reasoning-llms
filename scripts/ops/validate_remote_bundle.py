#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path('/root/workspace/jepa')

REQUIRED = [
    'deploy/remote/Dockerfile',
    'requirements-remote.txt',
    'scripts/phase1_train.py',
    'scripts/build_mcq_phase1_split.py',
    'scripts/prepare_mcq_phase1_views.py',
    'src/jepa_phase1/train.py',
    'configs/phase1/runs/gsm8k_gemma4e2b_lm.json',
    'configs/phase1/runs/gsm8k_gemma4e2b_coupled.json',
    'configs/phase1/runs/gsm8k_gemma4e2b_decoupled.json',
    'configs/phase1/runs/arc_challenge_gemma4e2b_lm.json',
    'configs/phase1/runs/hellaswag_gemma4e2b_lm.json',
    'configs/phase1/runs/mmlu_gemma4e2b_lm.json',
    'data/gsm8k/phase1_views/lm_answer_only/train_small.jsonl',
    'data/regexeval/phase1_views/lm_refined_to_expression/train_small.jsonl',
    'data/arc_challenge/phase1_views/lm_question_to_label/train_small.jsonl',
    'data/hellaswag/phase1_views/lm_context_to_label/train_small.jsonl',
    'data/mmlu/phase1_views/lm_question_to_label/train_small.jsonl',
    'analysis/benchmark_suite.md',
    'analysis/mcq_support_benchmarks_protocol.md',
    'deploy/provider_playbooks/vast.md',
    'deploy/provider_playbooks/lambda.md',
    'deploy/provider_playbooks/huggingface.md',
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', default=str(ROOT / 'deploy' / 'remote' / 'jepa_remote_bundle.tar.gz'))
    args = ap.parse_args()

    missing = [p for p in REQUIRED if not (ROOT / p).exists()]
    bundle = Path(args.bundle)
    payload = {
        'bundle_exists': bundle.exists(),
        'bundle_path': str(bundle),
        'missing': missing,
        'status': 'ok' if bundle.exists() and not missing else 'error',
    }
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
