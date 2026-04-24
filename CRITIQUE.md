# Critique guide

This repository is being opened for technical/scientific criticism before GPU spending on Vast/Lambda.

Please prioritize criticism over encouragement. The main value is to find flaws early.

## 1. Scientific claim

Check whether the proposed claim is too broad, too weak, or not testable:

> A decoupled JEPA-style latent-reasoning objective can be compared fairly against LM and coupled LLM-JEPA baselines under a matched Gemma backbone, using real language reasoning benchmarks.

Questions:

- Is the LM vs coupled vs decoupled comparison scientifically clean enough?
- Does the matched-backbone framing actually isolate the architectural/objective difference?
- Are GSM8K, RegexEval, ARC-Challenge, and HellaSwag the right paper-core suite?
- Is MMLU correctly treated as support rather than headline evidence?
- Are the current claims careful enough given there are no paper-scale results yet?

## 2. Methodology

Read:

- `analysis/methodology_protocol.md`
- `analysis/language_jepa_phase1_spec.md`
- `analysis/benchmark_suite.md`
- `analysis/article_framing.md`

Questions:

- Are train/eval splits and frozen-test rules adequate?
- Are the view definitions for JEPA-style training defensible?
- Is RegexEval evaluation using the right semantics? Current default is `search`, not `fullmatch`.
- Are the MCQ generation/evaluation rules too brittle?
- What ablations are mandatory before this can become paper-grade?

## 3. Architecture/code

Read:

- `src/jepa_phase1/models.py`
- `src/jepa_phase1/train.py`
- `src/jepa_phase1/evaluations.py`
- `scripts/phase1_train.py`

Questions:

- Is the decoupled `shared_backbone_stopgrad + SmallTalker` design still meaningfully decoupled?
- Does the coupled baseline receive an unfair advantage or disadvantage?
- Are LoRA target modules and freezing rules correct for Gemma-4-E2B?
- Are there hidden leakage risks in the views or evaluation prompts?
- Are losses/metrics being logged in a way that will support later analysis?

## 4. Execution plan

Read:

- `analysis/remote_execution_plan.md`
- `analysis/local_gsm8k_pilot_summary.md`
- `deploy/provider_playbooks/vast.md`
- `deploy/provider_playbooks/lambda.md`

Questions:

- Is the Vast -> Lambda -> Hugging Face order sensible?
- What should be the minimum success criterion before promoting from Vast to Lambda?
- What artifacts must be preserved for reproducibility?
- Are current GPU memory assumptions realistic?

## 5. Known weak spots to attack

- Tiny local pilot showed `0/4` partial GSM8K accuracy for all variants.
- Decoupled tiny pilot generated unstable/garbled outputs.
- No real GPU pilot has been run yet.
- No multi-seed result table exists.
- Latent diagnostics are still shallow.
- RegexEval report integration still needs a saved artifact path.
- MMLU auxiliary-train subject stratification is limited by dataset metadata.

## Desired review output

A useful review should include:

1. blocking issues before Vast
2. issues that can wait until after the Vast pilot
3. required fixes before Lambda paper-core runs
4. methodological concerns for paper framing
5. code/runtime risks likely to waste GPU budget
