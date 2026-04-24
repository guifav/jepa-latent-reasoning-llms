# Hugging Face final reproduction

Goal: rerun the final validated setup inside Hugging Face after the Lambda result is already accepted.

## Why this stage exists
- public-facing reproducibility
- artifact consolidation on the Hub
- final confirmation that the paper result is not tied to one vendor setup

## Prerequisites
- Hugging Face Pro account
- `HF_TOKEN` available
- final config family chosen from Lambda

## Local preparation on the VPS
Build the same bundle:

```bash
python /root/workspace/jepa/scripts/ops/build_remote_bundle.py
```

## Suggested execution model
Use Hugging Face Jobs.

### Minimal pattern
```bash
hf auth login
hf jobs run \
  --flavor a100-large \
  --timeout 86400 \
  --env HF_TOKEN=$HF_TOKEN \
  --volume /data \
  python:3.12 bash -lc '
    mkdir -p /workspace && cd /workspace && \
    tar -xzf /data/jepa_remote_bundle.tar.gz && \
    cd jepa && \
    python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install torch && \
    python -m pip install -r requirements-remote.txt && \
    python scripts/phase1_train.py configs/phase1/runs/gsm8k_gemma4e2b_lm.json --output-dir runs/hf_gsm8k_lm_seed1
  '
```

## What to run here
Only rerun the already chosen final paper setup.
Do not use HF as the place where we discover the final config.

At this stage the likely candidates should come from the paper-core suite:
- GSM8K
- RegexEval
- ARC-Challenge
- HellaSwag

MMLU can be repeated only if it became part of the accepted support package.

## Success condition
- same config family as Lambda
- materially similar result band
- artifacts uploaded and preserved in the Hub
