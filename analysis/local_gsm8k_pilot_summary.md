# Local GSM8K Gemma pilot summary

Last updated: 2026-04-24

## Purpose

This pilot was a local CPU readiness check for the Phase-1 JEPA/reasoning runtime, not a quality benchmark. The host is CPU-only with limited RAM, so the goal was to prove that the three Gemma paths execute end-to-end before moving the bounded pilot to Vast.

## Runs

| variant | run | status | device | validation loss | partial eval |
|---|---|---:|---|---:|---:|
| LM baseline | `gsm8k_gemma4e2b_lm_pilot` | code 0 | CPU | `6.8821` | `0/4` |
| Coupled LLM-JEPA | `gsm8k_gemma4e2b_coupled_pilot` | code 0 | CPU | `6.8746` | `0/4` |
| Decoupled JEPA-Reasoner | `gsm8k_gemma4e2b_decoupled_pilot` | code 0 | CPU | no val step in tiny 2-stage pilot | `0/4` |

Driver log:

```text
[start] gsm8k_gemma4e2b_coupled_pilot 2026-04-24T14:20:02Z
[done] gsm8k_gemma4e2b_coupled_pilot code=0 2026-04-24T14:58:48Z
[start] gsm8k_gemma4e2b_decoupled_pilot 2026-04-24T14:58:48Z
[done] gsm8k_gemma4e2b_decoupled_pilot code=0 2026-04-24T15:00:32Z
[start] gsm8k_gemma4e2b_lm_pilot 2026-04-24T15:00:32Z
[done] gsm8k_gemma4e2b_lm_pilot code=0 2026-04-24T15:13:27Z
```

## Interpretation

The useful result is execution readiness:

- Gemma-4-E2B loads and trains through the LM path.
- The coupled path trains and evaluates through the same runner.
- The redesigned decoupled path executes both stages without the old full-backbone duplication failure.
- Benchmark outputs are not meaningful yet: all three runs used a tiny partial GSM8K eval (`count=4`) and scored `0/4`; generations are unstable/garbled, especially in the decoupled talker.

## Current promotion status

The local pilot clears the “does this runtime execute?” gate. It does **not** clear a scientific quality gate.

Remote bundle status:

- bundle: `/root/workspace/jepa/deploy/remote/jepa_remote_bundle.tar.gz`
- validator: `status=ok`, `missing=[]`
- first remote preset: `vast-gsm8k-pilot`

## Next step

Launch the bounded GSM8K pilot on Vast using the prepared bundle and only inspect stability, memory, step time, and obvious config/data failures. Do not promote to Lambda until the Vast pilot finishes cleanly and any runtime issues are fixed.
