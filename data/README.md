# Data policy

This repository intentionally tracks dataset and view **manifests only**.

Full JSONL payloads are excluded from Git to keep the repository lightweight and reviewable. Rebuild them with the scripts in `scripts/`.

Tracked examples:

- `data/gsm8k/phase1/manifest.json`
- `data/gsm8k/phase1_views/manifest.json`
- `data/regexeval/phase1/manifest.json`
- `data/regexeval/phase1_views/manifest.json`
- `data/arc_challenge/phase1/manifest.json`
- `data/arc_challenge/phase1_views/manifest.json`
- `data/hellaswag/phase1/manifest.json`
- `data/hellaswag/phase1_views/manifest.json`
- `data/mmlu/phase1/manifest.json`
- `data/mmlu/phase1_views/manifest.json`

Rebuild commands are listed in the top-level `README.md`.
