# Phase-1 runnable config set

This directory contains the first concrete run configs for the paper-oriented phase-1 study.

## Benchmarks
- `gsm8k_*` → primary real reasoning benchmark
- `regexeval_*` → structured transfer benchmark (language → formal specification)
- `arc_challenge_*` → additional reasoning benchmark
- `hellaswag_*` → commonsense / continuation benchmark
- `mmlu_*` → support benchmark for generality coverage

## Model families per benchmark
- `*_lm.json` → Gemma LM baseline
- `*_coupled.json` → Gemma + LLM-JEPA coupled baseline
- `*_decoupled.json` → Gemma + JEPA-Reasoner decoupled baseline

## Methodological rules
- shared backbone: `google/gemma-4-E2B`
- shared adaptation regime: LoRA on the backbone
- newly introduced modules in decoupled runs are fully trainable
- matched-budget comparisons should be enforced within each benchmark

## Runtime entrypoints
- validation / inspection: `/root/workspace/jepa/scripts/phase1_runner.py`
- train-capable runner: `/root/workspace/jepa/scripts/phase1_train.py`
- local environment: `/root/workspace/jepa/.venv_phase1`

Example:

```bash
source /root/workspace/jepa/.venv_phase1/bin/activate
python /root/workspace/jepa/scripts/phase1_train.py \
  /root/workspace/jepa/configs/phase1/runs/gsm8k_gemma4e2b_lm.json
```

## Important note
These configs now map to a concrete training path that can:
- load the JSONL view corpora;
- instantiate the three model families;
- run train-capable smoke tests end-to-end.

What is still missing for full paper-grade execution:
- canonical handoff to `/root/workspace/jepa/scripts/eval_regex_semantics.py` as a saved report step for RegexEval runs;
- full Gemma-scale runtime validation on the target checkpoint;
- final choice of which of the expanded benchmark suite becomes part of the final paper headline table versus support appendix.
