# Remote execution order

Execution order frozen on 2026-04-24:

1. **Vast**
   - purpose: cheap iteration and bug-finding
   - workload: bounded GSM8K pilot only
   - success criterion: all three families run cleanly and produce usable early signal

2. **Lambda**
   - purpose: valid paper-facing training runs
   - workload: full paper-core phase-1 grid (`GSM8K + RegexEval + ARC-Challenge + HellaSwag`)
   - success criterion: stable seed runs with preserved artifacts

3. **Hugging Face**
   - purpose: final reproduction and artifact consolidation
   - workload: rerun only the final chosen setup after Lambda validation
   - success criterion: reproducible final result in the Hub ecosystem

## Local preparation on the VPS
Prepared artifacts:
- remote bundle builder: `/root/workspace/jepa/scripts/ops/build_remote_bundle.py`
- remote bundle validator: `/root/workspace/jepa/scripts/ops/validate_remote_bundle.py`
- batch launcher: `/root/workspace/jepa/scripts/ops/run_phase1_batch.py`
- Docker image recipe: `/root/workspace/jepa/deploy/remote/Dockerfile`
- playbooks:
  - `/root/workspace/jepa/deploy/provider_playbooks/vast.md`
  - `/root/workspace/jepa/deploy/provider_playbooks/lambda.md`
  - `/root/workspace/jepa/deploy/provider_playbooks/huggingface.md`

## Presets
- Vast pilot preset: `vast-gsm8k-pilot`
- Lambda paper-core preset: `lambda-phase1-full`
- Lambda broad-support preset: `lambda-phase1-broad`

## Benchmark layering
- paper-core: `gsm8k`, `regexeval`, `arc_challenge`, `hellaswag`
- support: `mmlu`
- deferred: `humaneval`
