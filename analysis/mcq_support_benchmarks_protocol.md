# MCQ support-benchmark protocol

Last updated: 2026-04-24

This document covers the additional multiple-choice benchmarks added after the original `GSM8K + RegexEval` phase-1 framing.

## Benchmarks covered
- **ARC-Challenge**
- **HellaSwag**
- **MMLU**

## Role in the project
- **ARC-Challenge** and **HellaSwag** are part of the paper-core suite.
- **MMLU** is a support benchmark for generality coverage.

## Shared prompt rule
All three benchmarks are materialized as option-selection tasks.
The generation target is normalized to:
- `Answer: <LABEL>`

Evaluation accepts either:
- the option label itself, or
- output that clearly resolves to the correct option text.

## View construction
### LM lane
- prompt = benchmark question/context + choices
- target = normalized answer label

### Coupled JEPA lane
- view A = question/context + choices
- view B = correct option text
- generation target = normalized answer label

### Decoupled JEPA lane
- condition = question/context + choices
- talker target = normalized answer label

## Dataset-specific notes
### ARC-Challenge
- train_small = full official train split
- val_small + dev_analysis = split from official validation
- test_frozen = official test

### HellaSwag
- public HF test is unlabeled, so validation is treated as frozen phase-1 test
- train_small / val_small / dev_analysis are sampled from official train

### MMLU
- train_small is sampled from `auxiliary_train`
- val_small + dev_analysis come from `dev`
- test_frozen = `validation`
- official `test` is mirrored separately for later use
- the aggregated `all` auxiliary-train view does not expose useful subject labels, so train sampling is primarily controlled by question length

## Metrics
### ARC-Challenge
- accuracy
- question-length bucket accuracy
- choice-length bucket accuracy

### HellaSwag
- accuracy
- context-length bucket accuracy
- ending-length bucket accuracy

### MMLU
- accuracy
- subject accuracy
- macro subject accuracy
- question-length bucket accuracy

## Artifact roots
- `/root/workspace/jepa/data/arc_challenge/phase1/`
- `/root/workspace/jepa/data/arc_challenge/phase1_views/`
- `/root/workspace/jepa/data/hellaswag/phase1/`
- `/root/workspace/jepa/data/hellaswag/phase1_views/`
- `/root/workspace/jepa/data/mmlu/phase1/`
- `/root/workspace/jepa/data/mmlu/phase1_views/`

## Config roots
- `/root/workspace/jepa/configs/phase1/runs/arc_challenge_*`
- `/root/workspace/jepa/configs/phase1/runs/hellaswag_*`
- `/root/workspace/jepa/configs/phase1/runs/mmlu_*`
