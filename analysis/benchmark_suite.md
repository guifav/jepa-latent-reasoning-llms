# Benchmark suite for language JEPA phase 1

Last updated: 2026-04-24

## Frozen decision
Phase 1 will no longer rely on **GSM8K alone**.
The benchmark suite is now split into three layers.

## Layer A — paper-core benchmarks
These are the main experiments that can support the article claim.

1. **GSM8K**
   - role: primary natural-language multi-step reasoning benchmark
   - why: strongest first test of latent reasoning in language

2. **RegexEval**
   - role: structured transfer benchmark
   - why: tests language-to-formal alignment with semantic evaluation

3. **ARC-Challenge**
   - role: additional reasoning benchmark
   - why: adds multiple-choice scientific/common-sense reasoning without collapsing into pure memorization

4. **HellaSwag**
   - role: commonsense inference / continuation benchmark
   - why: adds plausibility and narrative completion pressure beyond arithmetic reasoning

## Layer B — support benchmark
This benchmark is important, but it is not a headline claim carrier by itself.

5. **MMLU**
   - role: broad coverage / generality check
   - why: useful to see whether any JEPA effect survives across many subjects
   - caution: mixes reasoning with prior knowledge much more than GSM8K or ARC-Challenge

## Layer C — deferred benchmarks
These are intentionally postponed.

6. **HumanEval**
   - role: code-generation follow-up
   - status: phase 2, after the language benchmark stack is stable

## Explicit non-priority benchmarks for phase 1
These are not removed forever; they are just not part of the current frozen phase-1 suite.

- **TruthfulQA**
  - reason: more about factuality/alignment than the central latent-reasoning thesis
- **BIG-bench**
  - reason: too heterogeneous for a clean first paper claim

## Practical interpretation
The phase-1 paper should be able to say:
- the effect exists on a primary reasoning benchmark (**GSM8K**),
- transfers to a structured benchmark (**RegexEval**),
- survives on broader language reasoning/completion settings (**ARC-Challenge**, **HellaSwag**),
- and is checked for wider coverage (**MMLU**).

## Execution consequence
- **Vast**: cheap pilot / bug-finding, still starts with bounded GSM8K pilot
- **Lambda**: full paper-core suite (`GSM8K + RegexEval + ARC-Challenge + HellaSwag`)
- **Hugging Face**: rerun only the final chosen setup
- **MMLU**: support run after the paper-core grid is stable, or included in a broader Lambda package when budget allows
