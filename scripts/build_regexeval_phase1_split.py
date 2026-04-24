#!/usr/bin/env python3
import json
import math
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

DATASET = 's2e-lab/RegexEval'
CONFIG = 'default'
SPLIT = 'train'
SEED = 20260424
TRAIN_SMALL_N = 500
VAL_SMALL_N = 100
DEV_ANALYSIS_N = 40
PAGE_SIZE = 100
ROOT = Path('/root/workspace/jepa/data/regexeval')
PHASE1_DIR = ROOT / 'phase1'


def fetch_json(url: str, retries: int = 8):
    req = urllib.request.Request(url, headers={'User-Agent': 'roger-jepa-regexeval/1.0'})
    delay = 1.0
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries - 1:
                time.sleep(delay)
                delay = min(delay * 2, 20)
                continue
            raise


def fetch_rows():
    rows = []
    offset = 0
    while True:
        url = (
            'https://datasets-server.huggingface.co/rows?'
            + urllib.parse.urlencode({
                'dataset': DATASET,
                'config': CONFIG,
                'split': SPLIT,
                'offset': offset,
                'length': PAGE_SIZE,
            })
        )
        payload = fetch_json(url)
        batch = payload['rows']
        if not batch:
            break
        rows.extend(item['row'] for item in batch)
        offset += len(batch)
        if offset >= payload['num_rows_total']:
            break
        time.sleep(0.2)
    return rows


def quantile_thresholds(values):
    vals = sorted(values)
    q1_idx = max(0, math.floor((len(vals) - 1) / 3))
    q2_idx = max(0, math.floor(2 * (len(vals) - 1) / 3))
    return vals[q1_idx], vals[q2_idx]


def bucket(v, t1, t2):
    if v <= t1:
        return 'short'
    if v <= t2:
        return 'medium'
    return 'long'


def parse_record(row):
    raw_prompt = row['raw_prompt'].strip()
    refined_prompt = row['refined_prompt'].strip()
    expression = row['expression']
    matches = row['matches']
    non_matches = row['non_matches']
    return {
        'id': int(row['id']),
        'raw_prompt': raw_prompt,
        'refined_prompt': refined_prompt,
        'expression': expression,
        'matches': matches,
        'non_matches': non_matches,
        'raw_prompt_word_count': len(re.findall(r'\S+', raw_prompt)),
        'refined_prompt_word_count': len(re.findall(r'\S+', refined_prompt)),
        'regex_char_count': len(expression),
        'match_count': len(matches),
        'non_match_count': len(non_matches),
        'expression_empty': expression == '',
    }


def stratified_sample(records, n, rng):
    n = min(n, len(records))
    strata = defaultdict(list)
    for rec in records:
        strata[(rec['prompt_length_bucket'], rec['regex_length_bucket'])].append(rec)
    total = len(records)
    chosen = []
    leftovers = []
    for _, items in sorted(strata.items()):
        rng.shuffle(items)
        take = min(len(items), round(n * len(items) / total))
        chosen.extend(items[:take])
        leftovers.extend(items[take:])
    if len(chosen) > n:
        rng.shuffle(chosen)
        leftovers.extend(chosen[n:])
        chosen = chosen[:n]
    elif len(chosen) < n:
        rng.shuffle(leftovers)
        chosen.extend(leftovers[: n - len(chosen)])
    return sorted(chosen, key=lambda r: r['id'])


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def summarize(rows):
    return {
        'count': len(rows),
        'prompt_length_bucket': dict(Counter(r['prompt_length_bucket'] for r in rows)),
        'regex_length_bucket': dict(Counter(r['regex_length_bucket'] for r in rows)),
        'empty_expression_count': sum(r['expression_empty'] for r in rows),
        'avg_refined_prompt_word_count': round(sum(r['refined_prompt_word_count'] for r in rows) / max(1, len(rows)), 2),
        'avg_regex_char_count': round(sum(r['regex_char_count'] for r in rows) / max(1, len(rows)), 2),
    }


def main():
    rows = [parse_record(r) for r in fetch_rows()]

    p1, p2 = quantile_thresholds([r['refined_prompt_word_count'] for r in rows])
    r1, r2 = quantile_thresholds([r['regex_char_count'] for r in rows])
    for rec in rows:
        rec['prompt_length_bucket'] = bucket(rec['refined_prompt_word_count'], p1, p2)
        rec['regex_length_bucket'] = bucket(rec['regex_char_count'], r1, r2)

    rng = random.Random(SEED)
    pool = rows[:]
    rng.shuffle(pool)

    train_small = stratified_sample(pool, TRAIN_SMALL_N, random.Random(SEED + 1))
    train_ids = {r['id'] for r in train_small}
    remaining = [r for r in pool if r['id'] not in train_ids]

    val_small = stratified_sample(remaining, VAL_SMALL_N, random.Random(SEED + 2))
    val_ids = {r['id'] for r in val_small}
    remaining = [r for r in remaining if r['id'] not in val_ids]

    dev_analysis = stratified_sample(remaining, DEV_ANALYSIS_N, random.Random(SEED + 3))
    dev_ids = {r['id'] for r in dev_analysis}
    test_frozen = sorted([r for r in remaining if r['id'] not in dev_ids], key=lambda r: r['id'])

    PHASE1_DIR.mkdir(parents=True, exist_ok=True)
    subsets = {
        'train_small': train_small,
        'val_small': val_small,
        'dev_analysis': dev_analysis,
        'test_frozen': test_frozen,
    }
    for name, items in subsets.items():
        write_jsonl(PHASE1_DIR / f'{name}.jsonl', items)

    manifest = {
        'dataset': DATASET,
        'config': CONFIG,
        'source_split': SPLIT,
        'seed': SEED,
        'sampling': 'stratified_by_prompt_length_and_regex_length',
        'prompt_length_thresholds': {'short_max': p1, 'medium_max': p2},
        'regex_length_thresholds': {'short_max': r1, 'medium_max': r2},
        'splits': {
            name: {
                'path': str((PHASE1_DIR / f'{name}.jsonl').resolve()),
                'summary': summarize(items),
            }
            for name, items in subsets.items()
        },
        'notes': [
            'RegexEval is treated as the real secondary benchmark for phase 1.',
            'Semantic evaluation should use matches/non_matches and not exact match alone.',
            'Frozen test_frozen split must not be used for tuning.',
        ],
    }
    with (PHASE1_DIR / 'manifest.json').open('w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
