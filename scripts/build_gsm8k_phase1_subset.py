#!/usr/bin/env python3
import json
import math
import random
import re
import urllib.error
import urllib.parse
import urllib.request
import time
from collections import Counter, defaultdict
from pathlib import Path

DATASET = 'openai/gsm8k'
CONFIG = 'main'
SEED = 20260424
TRAIN_SMALL_N = 4000
VAL_SMALL_N = 600
DEV_ANALYSIS_N = 120
PAGE_SIZE = 100
ROOT = Path('/root/workspace/jepa/data/gsm8k')
RAW_DIR = ROOT / 'raw'
PHASE1_DIR = ROOT / 'phase1'


def fetch_json(url: str, retries: int = 8):
    req = urllib.request.Request(url, headers={'User-Agent': 'roger-jepa-phase1/1.0'})
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


def fetch_rows(split: str):
    rows = []
    offset = 0
    while True:
        url = (
            'https://datasets-server.huggingface.co/rows?'
            + urllib.parse.urlencode({
                'dataset': DATASET,
                'config': CONFIG,
                'split': split,
                'offset': offset,
                'length': PAGE_SIZE,
            })
        )
        payload = fetch_json(url)
        batch = payload['rows']
        if not batch:
            break
        for item in batch:
            rows.append(item['row'])
        offset += len(batch)
        if offset >= payload['num_rows_total']:
            break
        time.sleep(0.2)
    return rows


def normalize_answer_scalar(text: str) -> str:
    text = text.strip()
    text = text.replace(',', '')
    text = text.replace('$', '')
    text = text.replace('%', '')
    text = text.strip()
    # keep last scalar-looking token if there is extra text
    m = re.findall(r'-?\d+(?:\.\d+)?', text)
    if m:
        return m[-1]
    return text


def parse_record(split: str, idx: int, row: dict):
    question = row['question'].strip()
    answer = row['answer'].strip()
    if '\n#### ' in answer:
        rationale, final = answer.rsplit('\n#### ', 1)
    else:
        rationale, final = answer, answer
    rationale = rationale.strip()
    final = final.strip()
    final_norm = normalize_answer_scalar(final)
    rationale_words = len(re.findall(r"\S+", rationale))
    question_words = len(re.findall(r"\S+", question))
    answer_type = infer_answer_type(question, final, final_norm)
    return {
        'id': f'gsm8k-main-{split}-{idx:05d}',
        'split': split,
        'source_dataset': DATASET,
        'source_config': CONFIG,
        'question': question,
        'answer_raw': answer,
        'solution_rationale': rationale,
        'final_answer': final,
        'final_answer_normalized': final_norm,
        'rationale_word_count': rationale_words,
        'question_word_count': question_words,
        'answer_type': answer_type,
    }


def infer_answer_type(question: str, final: str, final_norm: str) -> str:
    q = question.lower()
    f = final.lower()
    if '$' in question or '$' in final or 'dollar' in q or 'cents' in q:
        return 'money'
    if ':' in final_norm:
        return 'other'
    if '.' in final_norm:
        return 'decimal'
    if re.fullmatch(r'-?\d+', final_norm):
        if any(k in q for k in ['how many', 'how much', 'total', 'altogether', 'left', 'remain']):
            return 'count_or_quantity'
        return 'integer'
    return 'other'


def quantile_thresholds(values):
    vals = sorted(values)
    q1_idx = max(0, math.floor((len(vals) - 1) / 3))
    q2_idx = max(0, math.floor(2 * (len(vals) - 1) / 3))
    return vals[q1_idx], vals[q2_idx]


def assign_length_bucket(count: int, t1: int, t2: int) -> str:
    if count <= t1:
        return 'short'
    if count <= t2:
        return 'medium'
    return 'long'


def stratified_sample(records, n, rng):
    n = min(n, len(records))
    strata = defaultdict(list)
    for rec in records:
        strata[(rec['length_bucket'], rec['answer_type'])].append(rec)
    total = len(records)
    chosen = []
    leftovers = []
    for key, items in sorted(strata.items()):
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
        'length_bucket': dict(Counter(r['length_bucket'] for r in rows)),
        'answer_type': dict(Counter(r['answer_type'] for r in rows)),
        'avg_rationale_word_count': round(sum(r['rationale_word_count'] for r in rows) / max(1, len(rows)), 2),
        'avg_question_word_count': round(sum(r['question_word_count'] for r in rows) / max(1, len(rows)), 2),
    }


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PHASE1_DIR.mkdir(parents=True, exist_ok=True)

    train_rows = [parse_record('train', i, row) for i, row in enumerate(fetch_rows('train'))]
    test_rows = [parse_record('test', i, row) for i, row in enumerate(fetch_rows('test'))]

    t1, t2 = quantile_thresholds([r['rationale_word_count'] for r in train_rows])
    for rows in (train_rows, test_rows):
        for rec in rows:
            rec['length_bucket'] = assign_length_bucket(rec['rationale_word_count'], t1, t2)

    write_jsonl(RAW_DIR / 'train.normalized.jsonl', train_rows)
    write_jsonl(RAW_DIR / 'test.normalized.jsonl', test_rows)

    rng = random.Random(SEED)
    train_pool = train_rows[:]
    rng.shuffle(train_pool)

    train_small = stratified_sample(train_pool, TRAIN_SMALL_N, random.Random(SEED + 1))
    train_small_ids = {r['id'] for r in train_small}
    remaining = [r for r in train_pool if r['id'] not in train_small_ids]

    val_small = stratified_sample(remaining, VAL_SMALL_N, random.Random(SEED + 2))
    val_small_ids = {r['id'] for r in val_small}
    remaining = [r for r in remaining if r['id'] not in val_small_ids]

    dev_analysis = stratified_sample(remaining, DEV_ANALYSIS_N, random.Random(SEED + 3))

    subsets = {
        'train_small': train_small,
        'val_small': val_small,
        'dev_analysis': dev_analysis,
        'test_official_full': test_rows,
    }
    for name, rows in subsets.items():
        write_jsonl(PHASE1_DIR / f'{name}.jsonl', rows)

    manifest = {
        'dataset': DATASET,
        'config': CONFIG,
        'seed': SEED,
        'sampling': 'stratified_by_length_bucket_and_answer_type',
        'length_bucket_thresholds': {
            'short_max': t1,
            'medium_max': t2,
        },
        'splits': {
            name: {
                'path': str((PHASE1_DIR / f'{name}.jsonl').resolve()),
                'summary': summarize(rows),
            }
            for name, rows in subsets.items()
        },
        'notes': [
            'Official test is mirrored locally for frozen final evaluation and must not be used for hyperparameter tuning.',
            'Length buckets are based on rationale word counts from the official train split.',
            'Answer normalization is deterministic and stored in this script.',
        ],
    }
    with (PHASE1_DIR / 'manifest.json').open('w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
