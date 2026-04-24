# JEPA Latent Reasoning for LLMs

This repository is a critique-ready research prototype for testing whether JEPA-style latent prediction can improve language-model reasoning under a matched-backbone experimental design.

The current repository is intentionally **not** a polished result release. It is prepared so reviewers can inspect the scientific framing, benchmark protocol, code paths, configuration choices, and known weaknesses before larger GPU runs are launched.

## Core question

Can a decoupled latent-reasoning objective, inspired by JEPA, provide measurable benefits over a standard LM baseline and a coupled LLM-JEPA baseline when all variants share the same Gemma backbone family?

## Current status

- Literature survey and first full-text synthesis are in `analysis/` and `fulltext_notes/`.
- Phase-1 benchmark protocol is specified for:
  - paper-core: `GSM8K + RegexEval + ARC-Challenge + HellaSwag`
  - support: `MMLU`
  - deferred: `HumanEval`
- Train/eval runtime exists in `src/jepa_phase1/` and `scripts/phase1_train.py`.
- Configs exist for LM, coupled LLM-JEPA, and decoupled JEPA-Reasoner variants.
- Local CPU smoke/pilot runs have completed for Gemma-4-E2B, but they are readiness checks only, not evidence of performance.
- Remote execution tooling is prepared for the planned order: `Vast -> Lambda -> Hugging Face`.

## What is included

- `analysis/` — research framing, methodology, benchmark suite, runtime notes, remote execution plan, local pilot summary.
- `configs/phase1/` — shared model configs and concrete run configs.
- `src/jepa_phase1/` — trainable wrappers, data loading, evaluation, model definitions, training loop.
- `scripts/` — dataset builders, view builders, validation runner, train entrypoint, remote batch helpers.
- `deploy/` — provider playbooks and Dockerfile; the generated remote tarball is intentionally excluded.
- `metadata/` — literature metadata/catalogs; PDFs are intentionally excluded.
- `fulltext_notes/` — human-readable notes from the first priority full-text pass; raw PDFs/full text are excluded.
- `data/**/manifest.json` — dataset/view manifests only; JSONL dataset payloads are intentionally excluded and should be rebuilt.
- `runs/pilot_summaries/` — small local pilot summaries only; checkpoints and large run folders are excluded.

## What is intentionally excluded

- raw PDFs (`pdfs/`)
- raw extracted full text (`fulltext/`)
- generated datasets (`data/**/*.jsonl`)
- checkpoints and full run outputs (`runs/`, `tmp_smoke/`)
- local virtual environments
- generated remote bundle tarballs

This keeps the repository reviewable and avoids turning GitHub into a data/checkpoint store.

## Quick orientation

Start here:

1. `analysis/article_framing.md`
2. `analysis/language_jepa_phase1_spec.md`
3. `analysis/methodology_protocol.md`
4. `analysis/benchmark_suite.md`
5. `analysis/local_gsm8k_pilot_summary.md`
6. `src/jepa_phase1/models.py`
7. `src/jepa_phase1/train.py`
8. `scripts/phase1_train.py`

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-phase1.txt
```

For GPU/remote execution, use `requirements-remote.txt` and the provider playbooks in `deploy/provider_playbooks/`.

## Rebuild data artifacts

The repository tracks manifests and builders, not the full dataset JSONL payloads. Rebuild locally with:

```bash
python scripts/build_gsm8k_phase1_subset.py
python scripts/prepare_gsm8k_phase1_views.py
python scripts/build_regexeval_phase1_split.py
python scripts/prepare_regexeval_phase1_views.py
python scripts/build_mcq_phase1_split.py --benchmark arc_challenge
python scripts/prepare_mcq_phase1_views.py --benchmark arc_challenge
python scripts/build_mcq_phase1_split.py --benchmark hellaswag
python scripts/prepare_mcq_phase1_views.py --benchmark hellaswag
python scripts/build_mcq_phase1_split.py --benchmark mmlu
python scripts/prepare_mcq_phase1_views.py --benchmark mmlu
```

Some builders may download from Hugging Face; authenticated `HF_TOKEN` is recommended for reliability.

## Validate configs

```bash
python scripts/phase1_runner.py configs/phase1/runs/gsm8k_gemma4e2b_lm.json --summary-only
python scripts/phase1_runner.py configs/phase1/runs/arc_challenge_gemma4e2b_lm.json --summary-only
python scripts/ops/run_phase1_batch.py --preset vast-gsm8k-pilot --dry-run
python scripts/ops/run_phase1_batch.py --preset lambda-phase1-full --dry-run
```

## Current known weaknesses

- No paper-grade multi-seed GPU result table exists yet.
- Local Gemma pilot accuracy is `0/4` on tiny partial GSM8K eval; it only proves runtime readiness.
- The decoupled talker produces unstable generations in tiny local pilot conditions.
- RegexEval canonical report handoff is not yet fully integrated as a saved train artifact.
- MMLU support-train sampling has a known caveat: aggregated `auxiliary_train` lacks useful subject labels.
- Local host is CPU-only and not suitable for paper-scale runs.

## Next planned execution

1. Run bounded GSM8K pilot on Vast.
2. Fix any runtime/data/config failures found there.
3. Promote stabilized configs to Lambda for the paper-core suite.
4. Use Hugging Face only for final reproduction/artifact consolidation.

See `analysis/remote_execution_plan.md` for details.

## License

License not selected yet. Treat this as a private/review-stage research prototype unless a license is added later.
