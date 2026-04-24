# JEPA gap map

These gaps are derived from the corpus distribution, milestone ordering, and concentration patterns across the title/abstract metadata. They are meant to identify promising work fronts, not to claim the field is missing only these items.

## G1 — No common evaluation spine across JEPA modalities

- Priority: **high**
- Evidence from corpus: Only 15 title-level benchmark/evaluation/critique papers in a 176-paper corpus, while applications sprawl across vision, video, audio, medical, remote sensing, robotics and wireless.
- Why it matters: Claims are hard to compare across papers because datasets, metrics, and baselines vary wildly by vertical.
- Opportunity: Build a JEPA benchmark pack with standardized baselines, compute budgets, and modality-specific plus cross-modal evaluation slices.
- Representative papers: 2404.08471, 2512.24497, 2602.13507, 2603.22649, 2604.10514

## G2 — Language and latent reasoning are promising but still immature

- Priority: **high**
- Evidence from corpus: Language/reasoning papers appear mostly from late 2025 onward; only 12 papers land in this bucket with title-based taxonomy.
- Why it matters: This is the most interesting bridge from JEPA into agentic systems and modern LLM use-cases, but the area is still recipe-fragile.
- Opportunity: Prototype JEPA-based latent reasoning with stronger evaluation on reasoning, planning, and token-generation robustness.
- Representative papers: 2509.14252, 2512.19171, 2603.22281, 2601.00366

## G3 — Generative JEPA remains fragmented

- Priority: **high**
- Evidence from corpus: The corpus contains multiple bridge attempts—denoising, diffusion, text fusion, tokenizer-style, variational and reasoner-style papers—but no single dominant recipe.
- Why it matters: JEPA is strong at representation learning, but the lack of a stable generative interface limits adoption in mainstream generative AI workflows.
- Opportunity: Study a unified JEPA-to-generator stack that preserves predictive latent semantics while enabling controllable generation.
- Representative papers: 2410.03755, 2509.14252, 2510.00974, 2512.19171, 2603.20111

## G4 — Training recipe stability is still unsettled

- Priority: **high**
- Evidence from corpus: Objective/theory papers accelerate only in late 2025–2026, with repeated focus on Gaussian geometry, sparsity, auxiliary losses, SIGReg-like regularization and normalization.
- Why it matters: The field is still paying recipe tax; too much work goes into making JEPA stable instead of building on a settled core recipe.
- Opportunity: Reproduce and distill a small set of stable JEPA recipes across image, video and language settings.
- Representative papers: 2511.08544, 2602.01456, 2603.05924, 2603.20111, 2604.21046

## G5 — World-model papers are growing fast, but transfer/generalization is under-proven

- Priority: **high**
- Evidence from corpus: World-model/planning/control is one of the fastest-growing clusters, but most papers target narrow environments such as driving, navigation, robotics, wireless control or physics simulators.
- Why it matters: Without cross-domain transfer evidence, JEPA world models may remain a collection of local wins instead of a general latent planning paradigm.
- Opportunity: Test whether one JEPA world-model backbone can transfer across video prediction, manipulation, driving and navigation with minimal task-specific surgery.
- Representative papers: 2505.03176, 2506.09985, 2602.11389, 2602.10098, 2603.19312

## G6 — Audio and non-vision modalities are still thinly explored

- Priority: **medium**
- Evidence from corpus: Only 5 audio/speech papers are caught by title-based taxonomy, far below vision/video growth.
- Why it matters: If JEPA is genuinely a general predictive-learning principle, the audio/speech story should be deeper and better benchmarked by now.
- Opportunity: Run a focused scaling and benchmark study for audio/speech JEPA, ideally tied to the same objective-design questions explored in vision.
- Representative papers: 2311.15830, 2507.02915, 2509.23238, 2512.07168

## G7 — Scientific and industrial applications are numerous but siloed

- Priority: **medium**
- Evidence from corpus: The corpus shows scattered JEPA use in remote sensing, wireless, physics, sonar, genomics, and molecular graphs, but little evidence of shared abstraction layers or transfer recipes.
- Why it matters: A common JEPA recipe for structured scientific data could unlock much broader reuse.
- Opportunity: Abstract the common pattern behind domain papers into a reusable JEPA template for non-natural-signal data.
- Representative papers: 2412.05333, 2502.03933, 2602.17162, 2603.25216, 2604.01349

