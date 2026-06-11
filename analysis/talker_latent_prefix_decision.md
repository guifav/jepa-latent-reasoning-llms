# Decision note: latent-prefix talker as the primary decoupled decoder

Last updated: 2026-06-11

This note records the architectural decision made in issue #6, before any paid
GPU run.

## Problem

The decoupled arm verbalized through `SmallTalker`, a single-layer GRU decoding
over the backbone embedding matrix, while the LM and coupled arms generate with
the full Gemma backbone. The matched-backbone framing held for encoding but not
for generation: a capacity confound sat exactly on the experiment's headline
comparison. If the decoupled arm lost, the result could not distinguish
"latent reasoning does not help" from "the talker is too weak to verbalize" —
a likely non-informative outcome, at GPU cost. Issue #1 review had already
flagged the talker capacity concern; the tiny local pilot's garbled decoupled
generations were consistent with it.

## Decision

`architecture.talker_mode` selects the decoupled decoder:

- **`latent_prefix` (new primary, set in all decoupled run configs):** the
  reasoner's predicted latent is projected into K soft-prefix embeddings
  (`architecture.latent_prefix_tokens`, default 8). The **frozen backbone**
  decodes conditioned on that prefix:
  - stage 2 trains only the prefix projector — teacher-forced target
    embeddings are prepended with the prefix and fed through the frozen
    backbone via `inputs_embeds`; cross-entropy is computed on target
    positions only;
  - generation runs a greedy loop with kv-cache, starting from the same
    start token used in training (`resolve_start_token_id()`);
  - optional stage 3 keeps the same joint structure (talker CE + small
    latent term).
- **`gru` (ablation):** the original `SmallTalker` path, unchanged. It now
  measures decoder-capacity dependence instead of carrying the headline
  result.

## Why this preserves decoupling

Reasoning still happens in latent space: the reasoner rolls out latently and
the only channel into generation is the K-token soft prefix. The backbone
receives no task gradient in stage 2 (it stays frozen; only the prefix
projector trains) and acts as a read-out/verbalizer. What changes is that all
three arms now share the same generation machinery, so differences on the
benchmarks reflect the conditioning, not the decoder size.

## Consequences for the experiment

- The three-way comparison (LM vs coupled vs decoupled) becomes interpretable
  on generation benchmarks; decoder capacity is no longer the obvious
  alternative explanation for a decoupled loss.
- `talker_size` grid values only apply to the `gru` ablation.
- Latent diagnostics are unchanged and still apply to both modes.

## Open questions / risks

- K (`latent_prefix_tokens`) is a new hyperparameter; 8 is a starting default,
  to be sanity-checked on the Vast pilot before any sweep.
- A single pooled latent expanded into K prefix tokens is still a tight
  information bottleneck; if the Vast pilot shows degenerate prefix
  conditioning (latent diagnostics + generations), richer reasoner-to-prefix
  mappings become the phase-2 ablation alongside stronger reasoners
  (issue #1, deferred).
- Stage-2 step cost differs between modes (frozen-backbone forward vs GRU);
  compute-matched comparisons should report per-step cost, as already required
  by the spec's mandatory logs.

## Code/test coverage

- Implementation: `src/jepa_phase1/models.py` (`DecoupledJepaReasonerWrapper`),
  mode-aware freeze rules in `src/jepa_phase1/train.py`.
- Unit tests with a tiny fake causal LM (no checkpoint download):
  `tests/test_latent_prefix_talker.py` — projector shapes, stage-2 gradient
  isolation (no backbone grads), stage-3 joint flow, generation contract for
  both modes, and `gru` regression.
