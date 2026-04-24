# Phase-1 status snapshot

Last updated: 2026-04-24

## Scientific framing
- Primary use case: **GSM8K** for real language reasoning.
- Structured transfer use case: **RegexEval** for language-to-formal alignment.
- Additional paper-core benchmarks: **ARC-Challenge** and **HellaSwag**.
- Support benchmark: **MMLU**.
- Deferred benchmark: **HumanEval**.
- Shared backbone family: **Gemma**.
- Primary checkpoint: **`google/gemma-4-E2B`**.
- Benchmark-suite reference: `/root/workspace/jepa/analysis/benchmark_suite.md`

## What is done
- Literature landscape, catalog, and first full-text synthesis are in place.
- Phase-1 protocols, frozen splits, and view corpora are in place.
- Train-capable runner exists at `/root/workspace/jepa/scripts/phase1_train.py`.
- Model paths exist for:
  - LM baseline
  - coupled LLM-JEPA
  - decoupled JEPA-Reasoner
- Built-in validation/evaluation now covers:
  - GSM8K normalized final-answer accuracy + buckets
  - RegexEval exact/semantic match with `search` semantics
  - ARC-Challenge / HellaSwag / MMLU multiple-choice accuracy with benchmark-specific breakdowns

## Host/runtime reality
Current host:
- CPU only
- ~16 GB RAM
- no swap

Observed Gemma-4-E2B feasibility on this host:
- model load succeeds
- LM 1-step micro-train succeeds
- coupled 1-step micro-train succeeds
- decoupled 2-stage 1-step micro-train now also succeeds

Representative micro-run outputs:
- `/root/workspace/jepa/tmp_smoke/out_gemma_lm_micro/summary.json`
- `/root/workspace/jepa/tmp_smoke/out_gemma_coupled_micro/summary.json`
- `/root/workspace/jepa/tmp_smoke/out_gemma_decoupled_micro/summary.json`

## Important architectural adjustment
The original decoupled implementation duplicated the full backbone for a target encoder and was not memory-safe on this machine.

Current host-feasible redesign:
- `target_encoder_mode = shared_backbone_stopgrad`
- no full backbone copy during stage 1
- stage 2 uses a **small GRU talker** conditioned on the latent reasoner state
- talker projects back to token space through the backbone embedding matrix

This keeps the experiment decoupled while making it runnable on the current hardware.

## Current honest status
The project is now past the “no real training path exists” stage.

What is already true:
- the three phase-1 families are implemented
- Gemma target checkpoint has been exercised for all three families
- the current host can execute micro-runs for all three families
- the bounded local GSM8K pilot also completed successfully for LM, coupled, and decoupled runs

What is not true yet:
- no benchmark-scale training run has been completed
- no paper-grade multi-seed result table exists yet
- the bounded local GSM8K pilot was mainly an executability/cost check and did not yet produce useful task signal (`0/4` partial eval accuracy in the three pilot summaries)
- RegexEval canonical report handoff via `eval_regex_semantics.py` still needs to be wired as a saved artifact step
- latent diagnostics are still shallow relative to the final paper goal

## Execution order
- Vast first: cheap pilot and improvement hunting
- Lambda second: valid paper-facing runs
- Hugging Face last: final reproduction after Lambda is accepted

Remote execution plan:
- `/root/workspace/jepa/analysis/remote_execution_plan.md`

## Immediate next step
Use the completed local pilot as a pure readiness check, then run the bounded GSM8K pilot on Vast and promote only the stabilized setup to Lambda across the paper-core suite (`GSM8K + RegexEval + ARC-Challenge + HellaSwag`).

Current local preparation:
- pilot configs: `/root/workspace/jepa/configs/phase1/runs/pilot_gsm8k/`
- completed local pilot summary: `/root/workspace/jepa/analysis/local_gsm8k_pilot_summary.md`
- completed local pilot outputs:
  - `/root/workspace/jepa/runs/gsm8k_gemma4e2b_lm_pilot/summary.json`
  - `/root/workspace/jepa/runs/gsm8k_gemma4e2b_coupled_pilot/summary.json`
  - `/root/workspace/jepa/runs/gsm8k_gemma4e2b_decoupled_pilot/summary.json`
- logs: `/root/workspace/jepa/runs/pilot_logs/`
- batch launcher: `/root/workspace/jepa/scripts/ops/run_phase1_batch.py`
- bundle builder: `/root/workspace/jepa/scripts/ops/build_remote_bundle.py`
