# NL-to-regex phase-1 protocol

## Decision
The phase-1 real secondary benchmark is:
- **RegexEval** (`s2e-lab/RegexEval`) as the primary evaluation benchmark

Auxiliary non-headline training corpus, if needed later:
- **phongo/RegEx** as optional extra supervision only

## Why RegexEval
RegexEval is the right phase-1 secondary benchmark because it gives:
- real-user prompt distribution rather than toy templated prompts;
- gold regex targets;
- explicit positive (`matches`) and negative (`non_matches`) test cases;
- a more auditável multi-view structure than GSM8K.

For JEPA, this is valuable because the alignment problem is much cleaner:
- natural-language description
- formal regex expression
- semantic behavior through matches / non-matches

## What counts as the benchmark
Headline phase-1 benchmark claims should be based on **RegexEval**, not on auxiliary training corpora.

If `phongo/RegEx` is used, it must be described as:
- auxiliary fine-tuning data;
- warm-start data;
- or exploratory data.

It must not be confused with the benchmark itself.

## Dataset structure
RegexEval examples contain:
- `raw_prompt`
- `refined_prompt`
- `expression`
- `matches`
- `non_matches`
- `id`

## View definition

### View A
`raw_prompt`

### View B
`refined_prompt`

### View C
`expression`

### View D (behavioral semantic view)
`matches + non_matches`

## Why this view setup matters
This benchmark is especially useful for JEPA because it lets us test three different alignments:
1. language-to-formal alignment (`prompt ↔ expression`)
2. weak-language-to-strong-language alignment (`raw_prompt ↔ refined_prompt`)
3. representation-to-behavior alignment (`expression ↔ example behavior`)

## Model usage

### LM baseline
Primary mode:
- input: `refined_prompt`
- target: `expression`

Secondary mode:
- input: `raw_prompt`
- target: `expression`

The refined-prompt setting should be the default headline result because it is less noisy.

### Coupled LLM-JEPA
Primary alignment targets:
- `raw_prompt ↔ refined_prompt`
- `refined_prompt ↔ expression`

Primary generation target:
- `expression`

### Decoupled JEPA-Reasoner
Primary use:
- Reasoner conditioned on prompt text
- latent rollout should support construction of the formal regex
- Talker outputs `expression`

Optional later extension:
- train with semantic feedback from `matches/non_matches`

## Evaluation metrics

### Primary metrics
- exact match on `expression`
- semantic match via behavioral tests (`matches` must match; `non_matches` must fail)

### Matching semantics
Phase 1 should use **search semantics**, not `fullmatch`, because RegexEval contains many expressions intended to match substrings or patterns within larger strings.

### Secondary metrics
- robustness to prompt reformulation
- accuracy by prompt-length bucket
- accuracy by regex-length bucket

## Headline metric rule
For this benchmark, **semantic match is more important than exact string match**.

Reason:
Two regexes can be semantically equivalent while having different surface form.

So the reporting order should be:
1. semantic match
2. exact match
3. robustness / bucketed analyses

## Split protocol
RegexEval ships as one train split of 762 items, so phase 1 must create and freeze its own internal split.

### Required local split
- `train_small`
- `val_small`
- `test_frozen`
- `dev_analysis`

Phase-1 split already materialized at:
- `/root/workspace/jepa/data/regexeval/phase1/`
- manifest: `/root/workspace/jepa/data/regexeval/phase1/manifest.json`

### Stratification axes
At minimum stratify by:
- prompt length
- regex length

Optional third axis if stable enough:
- prompt type/domain

## Test discipline
The local `test_frozen` split must not be used for:
- hyperparameter tuning
- early stopping decisions
- architecture selection

## Behavioral semantic evaluator
The semantic evaluator must:
- compile predicted regex safely;
- test all positive examples in `matches` using search semantics;
- test all negative examples in `non_matches` using search semantics;
- mark invalid regex separately from valid-but-wrong regex.

Phase-1 evaluator script:
- `/root/workspace/jepa/scripts/eval_regex_semantics.py`

## Reporting template
Every RegexEval result table should include:
- model name
- backbone checkpoint
- prompt mode (`raw` or `refined`)
- exact match
- semantic match
- invalid-regex rate
- long-prompt bucket performance
- compute summary

## Minimal phase-1 result set
A phase-1 RegexEval section is only complete if it includes:
1. LM baseline on refined prompt
2. coupled LLM-JEPA on refined prompt
3. decoupled JEPA-Reasoner on refined prompt
4. semantic match metric
5. invalid-regex rate

## Paper-safety notes
- do not overclaim from exact match alone;
- do not treat pretty formatting differences as failure if behavior is identical;
- do not use the same split for tuning and final reporting;
- if auxiliary data like `phongo/RegEx` is used, report both with and without it.
