# Vast first-pass execution

Goal: use Vast for fast iteration and cheap improvement hunting.

## What to provision
- 1x GPU with at least 24 GB VRAM
- preferred: RTX 4090 / A5000 / A6000 / L40S
- disk: >= 150 GB
- CUDA-ready image with Docker or Python 3.12

## What to upload
Build the bundle locally:

```bash
python /root/workspace/jepa/scripts/ops/build_remote_bundle.py
```

Copy `deploy/remote/jepa_remote_bundle.tar.gz` to the instance.

## Remote setup
On the Vast instance:

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

If Docker is easier:

```bash
docker build -f deploy/remote/Dockerfile -t jepa-phase1 .
```

## First runs
Start with the bounded pilot configs only:

```bash
source .venv/bin/activate
python scripts/phase1_train.py configs/phase1/runs/pilot_gsm8k/gsm8k_gemma4e2b_lm_pilot.json --output-dir runs/vast_lm_pilot
python scripts/phase1_train.py configs/phase1/runs/pilot_gsm8k/gsm8k_gemma4e2b_coupled_pilot.json --output-dir runs/vast_coupled_pilot
python scripts/phase1_train.py configs/phase1/runs/pilot_gsm8k/gsm8k_gemma4e2b_decoupled_pilot.json --output-dir runs/vast_decoupled_pilot
```

## What to inspect
- runtime stability
- memory headroom
- step time
- early validation signal
- failure modes worth fixing before Lambda

## Exit condition
Only move to Lambda after:
- all three runs finish cleanly
- no obvious data/config bug remains
- one stable config family is chosen for the paper-core run set

Note:
- Vast still starts with GSM8K pilot only
- ARC-Challenge / HellaSwag / MMLU preparation happens on the VPS first and is promoted only after the core pilot is stable
