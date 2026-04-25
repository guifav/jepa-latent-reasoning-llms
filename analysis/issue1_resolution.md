# Issue #1 resolution notes

Last updated: 2026-04-25

This note records the concrete changes made after the first architectural/methodological review.

## 1. Coupled JEPA baseline was too weak

Reviewer concern: pure cosine similarity over pooled LM embeddings can saturate and may not teach useful alignment.

Change:

- `CoupledLLMJepaWrapper` now supports `cosine_plus_infonce`.
- Each batch uses in-batch negatives: predicted view-B latents are scored against every target view-B latent in the batch.
- Configs now set:
  - `loss_type = cosine_plus_infonce`
  - `negative_sampling = in_batch`
  - `contrastive_temperature = 0.07`
  - `cosine_weight = 1.0`
  - `contrastive_weight = 1.0`
- Training logs now include `cosine_loss`, `contrastive_loss`, positive logit, and max negative logit.

Remaining caveat:

- With tiny per-device batches, in-batch negatives are weak. For paper-grade runs, either effective batch size or a queue/buffer should be considered if the contrastive signal is still too small.

## 2. Talker generation was underspecified

Reviewer concern: it was unclear whether `SmallTalker` is a decoder or an adapter into the causal-LM backbone.

Change:

- `SmallTalker` now has an explicit docstring.
- The implementation is documented as a tiny autoregressive decoder over backbone token embeddings.
- The causal-LM backbone is not called for decoding in the decoupled path; the backbone supplies the shared embedding matrix and vocabulary projection.
- The GRU talker maps previous token embeddings plus latent reasoner state to next-token logits.

Remaining caveat:

- This is still a small decoder. The reviewer's capacity concern remains valid for phase 2.

## 3. Reasoner/Talker misalignment risk

Reviewer concern: two-stage training can create a latent representation the Talker cannot verbalize.

Change:

- Added optional `stage3_joint_alignment` for the decoupled model.
- Stage 3 runs after reasoner pretraining and talker training.
- It jointly optimizes talker generation plus a small latent-prediction term at low LR.
- Configs now expose:
  - `stage_3_max_steps`
  - `learning_rate_joint`
  - `stage_3_unfreeze_backbone`
- Default config keeps full-backbone unfreeze disabled to avoid accidentally making the first GPU pilot too expensive; it still aligns reasoner/talker/new modules and LoRA trainables where enabled.

## 4. Latent geometry was not measured early enough

Reviewer concern: if task accuracy is low, geometry diagnostics are still needed to check whether JEPA signal is transmitted.

Change:

- Added batch-level latent diagnostics:
  - norm mean/std
  - effective rank
  - isotropy deviation
  - alignment cosine
  - coupled diag-vs-negative margin
- Training summaries now include `latent_diagnostics` for coupled and decoupled runs.
- Training history also records scalar model metrics at logging steps.

Remaining caveat:

- These are cheap first-pass diagnostics. A fuller latent analysis script should still be added before paper-grade Lambda runs.

## 5. Reasoner capacity concern

Reviewer concern: a GRUCell/GRU reasoner may be too weak for multi-step reasoning.

Change:

- No architectural capacity increase was made in this patch.
- This is intentionally deferred until after the Vast pilot, because adding transformer/cross-attention reasoner variants before the first GPU pilot would expand the search space too early.

Decision:

- Treat transformer/cross-attention reasoners as phase-2 ablations if the cheap Vast run shows that the current pipeline is stable and the latent diagnostics are non-degenerate.
