# Methodology protocol for article-grade rigor

## Purpose
This document defines the methodological rules for the JEPA-language project so the work can mature into a scientific paper rather than a sequence of ad hoc experiments.

## Core principle
Every central claim must be supported by:
1. a controlled comparison;
2. a frozen evaluation protocol;
3. reproducible configs;
4. at least one uncertainty or robustness check beyond raw accuracy.

## Claim structure
The project is allowed to make only four primary claim types:
1. **Architecture claim**: decoupled JEPA outperforms coupled JEPA or standard LM under matched budget.
2. **Geometry claim**: explicit latent regularization improves latent structure and/or stability.
3. **Uncertainty claim**: a variational JEPA formulation yields useful confidence or abstention behavior.
4. **Transfer/generality claim**: an effect repeats across more than one real task.

Any other observation should be labeled secondary or exploratory.

## Experimental hierarchy

### Tier 1 — primary experiments
These support the main paper claims.
- Backbone: `google/gemma-4-E2B`
- Tasks:
  - GSM8K
  - RegexEval
  - ARC-Challenge
  - HellaSwag
- Models:
  - LM baseline
  - LLM-JEPA coupled baseline
  - JEPA-Reasoner decoupled baseline

### Tier 2 — confirmatory experiments
These test whether tier-1 results are robust.
- repeat on another seed set;
- add support coverage on MMLU;
- optional scale-up to `google/gemma-4-E4B` if the tier-1 signal is real.

### Tier 3 — exploratory experiments
These are valuable but not allowed to carry the main conclusion alone.
- synthetic diagnostics;
- sparse latent variants;
- external teacher/guidance;
- wider hyperparameter sweeps after the main protocol is frozen.

## Comparison rules

### Rule 1 — matched backbone
Primary comparisons must use the same checkpoint family and same initial checkpoint when possible.

### Rule 2 — matched budget
For any central comparison, report and approximately match:
- number of training examples;
- optimizer family;
- batch size or effective batch size;
- total update steps;
- wall-clock cost when available;
- trainable parameter count if architectures diverge.

### Rule 3 — no mixed wins
Do not claim one model is “better” if it only wins after getting a larger model, more steps, or additional supervision not given to the comparator.

## Data protocol

### Primary rule
Use real benchmarks as the core evidence.

### Leakage control
- training/validation/test splits must be frozen and versioned;
- no test-driven hyperparameter tuning;
- if a benchmark has an official test set, keep it untouched until final evaluation;
- all preprocessing scripts must be saved.

### Multi-view construction rule
If views are derived rather than native, document:
- exact transform;
- whether the transform is deterministic or stochastic;
- whether it leaks the target answer;
- which model, if any, was used to create it.

## Evaluation protocol

### Metrics to report for all primary experiments
- main task accuracy / exact match;
- one robustness metric;
- one latent-space diagnostic;
- compute/cost summary.

### Robustness metrics
At least one of:
- input perturbation robustness;
- performance by reasoning-horizon bucket;
- paraphrase robustness;
- latent noise sensitivity.

### Latent diagnostics
At least two of:
- effective rank;
- covariance spectrum / isotropy deviation;
- latent norm stability;
- view alignment linearity;
- coupled positive-vs-negative contrastive margin;
- Reasoner→Talker alignment loss when stage 3 is enabled;
- Talker dependence ablation.

### Uncertainty metrics
Required once the variational branch starts:
- risk-coverage curve;
- selective accuracy;
- calibration proxy / ECE when applicable;
- AUROC for error detection if feasible.

## Statistical discipline

### Seeds
- minimum 3 seeds for central comparisons in phase 1;
- target 5 seeds for anything that becomes a paper figure or table headline.

### Reporting
Always report:
- mean;
- standard deviation or standard error;
- number of runs.

### Significance
When comparing central models, prefer:
- paired bootstrap or paired t-test when assumptions are acceptable;
- otherwise report effect size and confidence interval instead of only p-value.

Do not overclaim from single-run wins.

## Model selection protocol

### Allowed for tuning
- training split;
- validation split;
- internal development subset explicitly marked as dev-only.

### Not allowed for tuning
- frozen final test split;
- hand-picked examples shown after test inspection.

### Hyperparameter policy
Primary comparisons should use a small predeclared grid. If a model needs a much larger sweep to win, that must be disclosed.

## Ablation discipline
Every major component claim needs a nearest-neighbor ablation.

Examples:
- decoupling claim -> compare same backbone with and without Reasoner/Talker split;
- geometry claim -> compare same decoupled model with and without geometry regularization;
- uncertainty claim -> deterministic vs variational under same task and backbone.

## Failure analysis protocol
A negative result is publishable if documented correctly.

For every central experiment that fails, record:
- what exactly failed;
- whether the issue looks like optimization, data, evaluation, or architecture;
- whether the latent diagnostics moved in a useful direction despite flat task accuracy;
- one concrete next hypothesis.

## Figure and table policy
If an experiment is likely to appear in the paper, save:
- config file;
- commit hash;
- dataset version or hash;
- seed;
- checkpoint name;
- raw metrics;
- script used to produce the final table/figure.

## Writing discipline
When drafting the eventual paper:
- separate confirmed claims from exploratory observations;
- avoid saying “improves” without matched-budget evidence;
- avoid saying “robust” without explicit perturbation results;
- avoid saying “uncertainty” if the metric is only entropy of token logits;
- avoid saying “interpretable latent reasoning” unless the latent analysis truly supports it.

## Exit criteria for a credible first paper draft
A first paper draft is justified only if all of the following are true:
1. at least one real benchmark shows a repeatable effect;
2. the core comparison is matched and reproducible;
3. the result is supported by more than raw accuracy;
4. the paper can state a precise claim that survives ablation.
