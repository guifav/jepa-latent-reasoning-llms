# Backbone selection for phase 1

## Decision
Primary phase-1 backbone:
- **`google/gemma-4-E2B`**

Confirmatory scale-up only after a positive phase-1 signal:
- **`google/gemma-4-E4B`**

## Why `google/gemma-4-E2B`

### 1. It keeps the phase-1 comparison affordable
Phase 1 is about architecture and methodology, not maximum benchmark score. The smaller Gemma 4 checkpoint makes it more realistic to run:
- multiple seeds;
- ablations;
- robustness checks;
- latent diagnostics.

That matters more for a paper than a single expensive headline run.

### 2. It reduces confounds
Using the **base** checkpoint instead of the instruction-tuned one keeps the comparison cleaner:
- less conversational alignment baggage;
- fewer hidden differences from RL/alignment style tuning;
- cleaner intervention when modifying the objective or architecture.

### 3. It matches the scientific question
The core question is not “what is the strongest Gemma variant on GSM8K?”
The real question is:
- does JEPA-style coupling help?
- does JEPA-style decoupling help more?
- does latent regularization help further?

For that, the cleanest setup is a shared non-instruction base checkpoint.

### 4. Gemma 4 is still a strong modern family
Using Gemma 4 preserves relevance. We are not choosing a weak legacy base just for convenience.

## Why not start from `-it`
Instruction-tuned checkpoints are useful later, but they introduce extra ambiguity:
- some gains may come from chat alignment rather than JEPA;
- rationale style may change independently of reasoning quality;
- answer formatting becomes harder to attribute to the architectural change.

For phase 1, that is noise.

## Why keep `google/gemma-4-E4B` as confirmatory
If phase 1 shows a real effect, the next scientific question is whether the effect survives a modest scale increase. `google/gemma-4-E4B` is the right confirmatory step because it tests scale sensitivity without changing family.

## What we are explicitly not claiming
- that `google/gemma-4-E2B` is the best model overall;
- that phase-1 results automatically transfer to all Gemma variants;
- that base checkpoints are always better than instruction-tuned ones.

This is a methodological choice for clean evidence, not a universal ranking claim.
