# Lambda paper-grade execution

Goal: run the valid paper-facing experiments after Vast has already shaken out bugs and obvious weak settings.

## What to provision
- preferred: A100 80GB
- acceptable: H100 80GB if price/availability is better
- disk: >= 200 GB
- persistent workspace enabled

## What to upload
Use the same bundle created on the VPS:

```bash
python /root/workspace/jepa/scripts/ops/build_remote_bundle.py
```

Upload `deploy/remote/jepa_remote_bundle.tar.gz`.

## Remote setup
```bash
mkdir -p ~/work && cd ~/work
tar -xzf jepa_remote_bundle.tar.gz
cd jepa
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch
python -m pip install -r requirements-remote.txt
```

## What to run
Run the frozen paper-core configs, not the pilot configs.

Paper-core preset:
```bash
python scripts/ops/run_phase1_batch.py --preset lambda-phase1-full
```

This preset covers:
- GSM8K
- RegexEval
- ARC-Challenge
- HellaSwag

If budget allows, run the broader support package afterward:
```bash
python scripts/ops/run_phase1_batch.py --preset lambda-phase1-broad
```

That adds:
- MMLU

Repeat with multiple seeds only after seed1 is clean.

## Required artifacts to preserve
- config used
- summary.json
- stdout/stderr logs
- checkpoints
- RegexEval semantic reports
- benchmark manifests for ARC-Challenge / HellaSwag / MMLU when those runs are included
