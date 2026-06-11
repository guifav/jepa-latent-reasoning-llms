# Local pilot summaries (pre-fix artifacts)

These `summary.json` files were generated on 2026-04-24 by the local CPU pilot,
**before** the eval-pipeline fixes from issue #2 / PR #3 landed. They are kept
unmodified as historical artifacts and as evidence of runtime readiness.

## What is affected

The eval/generation path at that time had bugs fixed later in PR #3. They hit
the variants differently:

- **LM and coupled** (`gsm8k_gemma4e2b_lm_pilot`, `gsm8k_gemma4e2b_coupled_pilot`):
  the decode slice echoed the prompt into predictions for every sequence
  shorter than the longest one in the left-padded batch — visible in the
  `prediction_raw` fields of these two summaries; since answer normalization
  takes the last number in the text, the echoed question could contaminate
  `prediction_normalized`;
- **decoupled** (`gsm8k_gemma4e2b_decoupled_pilot`): its decode path never
  included the input, so prompt echo does not apply here; instead, the
  condition latent was pooled from a padding position at eval time, and the
  talker started generation with a different token than it was trained with.

As a consequence, the `benchmark_eval` blocks (`accuracy: 0/4`, per-bucket
accuracies, recorded predictions) of **all three** summaries are not
comparable with any run executed after PR #3 — for the reasons above,
per variant.

## What remains valid

Training losses, stage execution, runtime/memory feasibility, and the overall
"the three variants execute end-to-end on this host" conclusion — the training
path was not affected by the fixes.

Next regeneration happens naturally at the bounded Vast GSM8K pilot. See
`analysis/local_gsm8k_pilot_summary.md` for the full pilot report and caveat.
