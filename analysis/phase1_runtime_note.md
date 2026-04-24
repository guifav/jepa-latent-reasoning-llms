# Phase-1 runtime note

The phase-1 artifacts now have a real training path in this workspace.

## Environment status
Installed locally in:
- `/root/workspace/jepa/.venv_phase1`

Current host runtime probe:
- Python `3.12.3`
- Torch `2.11.0+cu130`
- `torch.cuda.is_available() = False`
- visible CUDA devices: `0`

Declared requirements file:
- `/root/workspace/jepa/requirements-phase1.txt`

Installed runtime includes:
- `torch`
- `transformers`
- `accelerate`
- `peft`
- `datasets`
- `sentencepiece`
- `evaluate`
- supporting scientific stack

## Runtime entrypoints
- config/data validator: `/root/workspace/jepa/scripts/phase1_runner.py`
- gradient-training runner: `/root/workspace/jepa/scripts/phase1_train.py`
- training package root: `/root/workspace/jepa/src/jepa_phase1/`

## Implemented runner capabilities
The current training path can:
1. load a run config from `/root/workspace/jepa/configs/phase1/runs/`;
2. load JSONL view corpora from `/root/workspace/jepa/data/*/phase1_views/`;
3. instantiate one of three model families:
   - Gemma LM baseline wrapper
   - coupled LLM-JEPA wrapper
   - decoupled JEPA-Reasoner wrapper
4. run gradient-based training loops;
5. save stage summaries and model checkpoints;
6. execute a two-stage flow for the decoupled model.

## Smoke-test evidence
Validated end-to-end with a tiny public checkpoint (`sshleifer/tiny-gpt2`) and sliced GSM8K view corpora:
- LM smoke run: passed
- coupled JEPA smoke run: passed
- decoupled JEPA smoke run: passed

Smoke outputs:
- `/root/workspace/jepa/tmp_smoke/out_lm/summary.json`
- `/root/workspace/jepa/tmp_smoke/out_coupled/summary.json`
- `/root/workspace/jepa/tmp_smoke/out_decoupled/summary.json`

## Gemma target-host probe
Direct probe against `google/gemma-4-E2B` on this host:
- model load: succeeded
- observed parameter count: `5,104,297,504`
- load time: about `76s`
- load probe peak RSS: about `3.2 GB`

Real micro-train probes on frozen phase-1 data:
- LM baseline, 1 step on CPU: succeeded
  - wall time: about `68s`
  - peak RSS: about `11.8 GB`
  - output: `/root/workspace/jepa/tmp_smoke/out_gemma_lm_micro/summary.json`
- coupled LLM-JEPA, 1 step on CPU: succeeded
  - wall time: about `61s`
  - peak RSS: about `11.8 GB`
  - output: `/root/workspace/jepa/tmp_smoke/out_gemma_coupled_micro/summary.json`

Host memory context:
- `MemTotal`: about `16.4 GB`
- `MemAvailable` before probe: about `13.7 GB`
- swap: `0`

Implication after the first redesign pass:
- **LM and coupled Gemma runs are technically possible here in very small form on CPU**;
- the original decoupled design was not safe here because it duplicated the backbone;
- the decoupled path was then redesigned to use:
  - `target_encoder_mode = shared_backbone_stopgrad`
  - a small GRU talker for stage 2 instead of pushing giant `inputs_embeds` tensors back through full Gemma
- with that redesign, a 2-stage 1-step Gemma decoupled micro-run also succeeded:
  - wall time: about `82s`
  - peak RSS: about `12.0 GB`
  - output: `/root/workspace/jepa/tmp_smoke/out_gemma_decoupled_micro/summary.json`

## Honest current status
What is ready now:
- benchmark protocols
- frozen splits
- semantic evaluator
- view corpora
- concrete phase-1 run configs
- local ML runtime stack
- train-capable runner
- initial wrappers for all three model families
- first real Gemma micro-train validation for LM, coupled, and decoupled baselines

What is still missing for paper-grade execution:
- full benchmark-scale Gemma runs beyond 1-step micro validation
- real bounded pilot runs with enough steps to compare early learning signal
- deeper benchmark evaluation coverage beyond the current val-loop generation metrics
- hard-wiring `/root/workspace/jepa/scripts/eval_regex_semantics.py` as the canonical RegexEval report artifact path (the runner now reproduces the same search-based semantic rule internally, but not yet through the external report script)
- richer latent diagnostics and ablation reporting
- hardware/runtime confirmation for realistic multi-run training budgets

## Practical read
The project is no longer blocked on “there is no training code.”
The main runtime reality is now clearer:
- LM baseline: runnable here, but slow
- coupled JEPA: runnable here, but slow
- decoupled JEPA: runnable here after the host-feasible redesign, but still slow and methodologically provisional relative to the original EMA-copy variant
