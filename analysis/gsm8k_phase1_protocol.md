# GSM8K phase-1 protocol

## Goal
Define a controlled and reproducible GSM8K setup for the first JEPA-language comparison.

## Role of GSM8K in the project
GSM8K is the primary real benchmark for phase 1 because it gives:
- a real language reasoning task;
- a natural decomposition into question, rationale, and final answer;
- continuity with JEPA-Reasoner;
- evaluation that is simple enough to keep the first paper disciplined.

## Data fields
Each example should be normalized into:
- `question`
- `solution_rationale`
- `final_answer`
- `rationale_length_tokens`
- `answer_value_normalized`

## View construction

### View A
`question`

### View B
`solution_rationale`

### View C
`final_answer`

## Model usage by view

### LM baseline
Primary mode:
- input: `question`
- target: `final_answer`

Optional secondary mode:
- input: `question`
- target: `solution_rationale + final_answer`

These two modes must not be conflated in the same headline result table.

### Coupled LLM-JEPA
Primary alignment target:
- align `question ↔ solution_rationale`
- generate `final_answer`

Secondary analysis:
- align `question ↔ final_answer`
- compare whether the stronger signal is in rationale alignment or answer alignment.

### Decoupled JEPA-Reasoner
Primary use:
- Reasoner conditioned on `question`
- latent rollout represents intermediate reasoning trajectory
- Talker verbalizes `solution_rationale` and/or `final_answer`

## Split protocol

### Official split discipline
- never mix official train and official test for tuning;
- official test remains frozen for final evaluation only.

### Phase-1 working split
From official train:
1. create a **train_small** subset;
2. create a **val_small** subset for model selection;
3. optionally create a **dev_analysis** slice for fast qualitative inspection.

### Recommended initial sizes
These are starting points, not sacred constants:
- `train_small`: 3k–5k examples
- `val_small`: 400–800 examples
- `dev_analysis`: 100–200 examples

The exact counts must be frozen once chosen.

## Stratification policy
The subset should be stratified by at least:
- rationale length;
- answer type when detectable (integer, decimal, money, count, ratio);
- approximate lexical question length.

Minimum bucket plan for rationale length:
- short
- medium
- long

These buckets are not just for sampling; they must also be used in reporting.

## Answer normalization
Because GSM8K evaluation can look better or worse depending on parsing sloppiness, the answer extraction rule must be frozen.

### Required normalization steps
- strip currency symbols when appropriate;
- normalize commas and spaces;
- normalize decimal formatting;
- extract the final scalar answer from rationale-form outputs deterministically;
- keep the extraction script versioned.

## Primary metrics
- final answer accuracy
- accuracy by rationale-length bucket
- accuracy by answer-type bucket

## Secondary metrics
- rationale exact or structured match when applicable
- robustness to light paraphrase of the question
- selective accuracy for the variational model

## Robustness protocol
Phase 1 must include at least one controlled robustness check.

### Preferred first robustness test
Light paraphrase of the question while preserving numeric content and task semantics.

### Constraint
The paraphrase transform must not inject new solution hints or alter quantities.

## Model selection rules
Tuning decisions may use only:
- `train_small`
- `val_small`

Do not use official test to decide:
- `lambda_jepa`
- rollout length
- Talker size
- stopping point
- regularization strength

## Reporting template
Every GSM8K result table should include:
- model name
- checkpoint (`google/gemma-4-E2B` in phase 1)
- training mode
- training subset size
- seed count
- final answer accuracy
- long-bucket accuracy
- robustness metric
- compute summary

## Minimal phase-1 result set
A phase-1 GSM8K section is only considered complete if it includes:
1. LM baseline
2. coupled LLM-JEPA
3. decoupled JEPA-Reasoner
4. bucketed performance by rationale length
5. at least one robustness result

## Paper-safety notes
- do not present dev-set wins as benchmark conclusions;
- do not merge rationale-generation and answer-only runs into one number;
- do not claim better reasoning if only verbosity changed;
- do not interpret longer rationales as better unless accuracy and robustness also improve.
