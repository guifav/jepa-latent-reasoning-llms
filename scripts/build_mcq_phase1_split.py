#!/usr/bin/env python3
import argparse
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

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

SEED = 20260424
PAGE_SIZE = 100
LABELS = ['A', 'B', 'C', 'D']


BENCHMARKS = {
    'arc_challenge': {
        'dataset': 'allenai/ai2_arc',
        'config': 'ARC-Challenge',
        'root': Path('/root/workspace/jepa/data/arc_challenge'),
        'train_split': 'train',
        'val_split': 'validation',
        'test_split': 'test',
        'train_small_n': None,
        'val_small_n': 220,
        'dev_analysis_n': 79,
        'sampling_note': 'train_small uses the full official train split; validation is split into val_small + dev_analysis; official test is frozen.',
    },
    'hellaswag': {
        'dataset': 'Rowan/hellaswag',
        'config': 'default',
        'root': Path('/root/workspace/jepa/data/hellaswag'),
        'train_split': 'train',
        'val_split': 'validation',
        'test_split': 'validation',
        'train_small_n': 8000,
        'val_small_n': 1000,
        'dev_analysis_n': 200,
        'sampling_note': 'Public HellaSwag test is unlabeled in the HF mirror; validation is treated as frozen phase-1 test.',
    },
    'mmlu': {
        'dataset': 'cais/mmlu',
        'config': 'all',
        'root': Path('/root/workspace/jepa/data/mmlu'),
        'train_split': 'auxiliary_train',
        'val_split': 'dev',
        'test_split': 'validation',
        'extra_raw_split': 'test',
        'train_small_n': 5000,
        'val_small_n': 200,
        'dev_analysis_n': 85,
        'sampling_note': 'Phase-1 train comes from auxiliary_train; dev is split into val_small + dev_analysis; validation is the frozen phase-1 test and official test is mirrored separately for later use.',
    },
}


def fetch_json(url: str, retries: int = 8):
    req = urllib.request.Request(url, headers={'User-Agent': 'roger-jepa-mcq-phase1/1.0'})
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


def fetch_rows(dataset: str, config: str, split: str):
    if load_dataset is not None:
        ds = load_dataset(dataset, config, split=split)
        return [dict(row) for row in ds]

    rows = []
    offset = 0
    while True:
        url = (
            'https://datasets-server.huggingface.co/rows?'
            + urllib.parse.urlencode({
                'dataset': dataset,
                'config': config,
                'split': split,
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
        time.sleep(0.15)
    return rows


def quantile_thresholds(values):
    vals = sorted(values)
    q1_idx = max(0, math.floor((len(vals) - 1) / 3))
    q2_idx = max(0, math.floor(2 * (len(vals) - 1) / 3))
    return vals[q1_idx], vals[q2_idx]


def bucket(value: int, t1: int, t2: int) -> str:
    if value <= t1:
        return 'short'
    if value <= t2:
        return 'medium'
    return 'long'


def normalize_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def word_count(text: str) -> int:
    return len(re.findall(r'\S+', text))


def choice_avg_word_count(choice_texts: list[str]) -> float:
    if not choice_texts:
        return 0.0
    return sum(word_count(x) for x in choice_texts) / len(choice_texts)


def stratified_sample(records, n, rng, key_fn):
    n = min(n, len(records))
    strata = defaultdict(list)
    for rec in records:
        strata[key_fn(rec)].append(rec)
    total = len(records)
    chosen = []
    leftovers = []
    for _, items in sorted(strata.items(), key=lambda kv: repr(kv[0])):
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


def summarize(rows, bucket_key: str, aux_bucket_key: str, extra_key: str | None = None):
    payload = {
        'count': len(rows),
        bucket_key: dict(Counter(r[bucket_key] for r in rows)),
        aux_bucket_key: dict(Counter(r[aux_bucket_key] for r in rows)),
        'avg_question_word_count': round(sum(r['question_word_count'] for r in rows) / max(1, len(rows)), 2),
        'avg_choice_word_count': round(sum(r['choice_avg_word_count'] for r in rows) / max(1, len(rows)), 2),
    }
    if extra_key:
        payload[extra_key] = dict(Counter(r[extra_key] for r in rows))
    return payload


def parse_arc(split: str, idx: int, row: dict):
    labels = row['choices']['label']
    texts = [normalize_whitespace(x) for x in row['choices']['text']]
    answer_label = row['answerKey'].strip()
    answer_index = labels.index(answer_label)
    question = normalize_whitespace(row['question'])
    return {
        'id': f"arc-challenge-{split}-{idx:05d}",
        'split': split,
        'source_dataset': 'allenai/ai2_arc',
        'source_config': 'ARC-Challenge',
        'question': question,
        'choice_labels': labels,
        'choice_texts': texts,
        'answer_label': answer_label,
        'answer_index': answer_index,
        'answer_text': texts[answer_index],
        'question_word_count': word_count(question),
        'choice_avg_word_count': round(choice_avg_word_count(texts), 2),
    }


def parse_hellaswag(split: str, idx: int, row: dict):
    context = normalize_whitespace(row['ctx'])
    endings = [normalize_whitespace(x) for x in row['endings']]
    label_raw = row.get('label', '')
    answer_index = int(label_raw) if str(label_raw).strip() != '' else None
    answer_label = LABELS[answer_index] if answer_index is not None else ''
    answer_text = endings[answer_index] if answer_index is not None else ''
    return {
        'id': f"hellaswag-{split}-{idx:05d}",
        'split': split,
        'source_dataset': 'Rowan/hellaswag',
        'source_config': 'default',
        'context': context,
        'choice_labels': LABELS[: len(endings)],
        'choice_texts': endings,
        'answer_label': answer_label,
        'answer_index': answer_index,
        'answer_text': answer_text,
        'question_word_count': word_count(context),
        'choice_avg_word_count': round(choice_avg_word_count(endings), 2),
        'activity_label': row.get('activity_label', '').strip(),
        'split_type': row.get('split_type', '').strip(),
        'source_id': row.get('source_id', '').strip(),
    }


def parse_mmlu(split: str, idx: int, row: dict):
    question = normalize_whitespace(row['question'])
    texts = [normalize_whitespace(x) for x in row['choices']]
    answer_index = int(row['answer'])
    answer_label = LABELS[answer_index]
    return {
        'id': f"mmlu-{split}-{idx:05d}",
        'split': split,
        'source_dataset': 'cais/mmlu',
        'source_config': 'all',
        'question': question,
        'choice_labels': LABELS[: len(texts)],
        'choice_texts': texts,
        'answer_label': answer_label,
        'answer_index': answer_index,
        'answer_text': texts[answer_index],
        'question_word_count': word_count(question),
        'choice_avg_word_count': round(choice_avg_word_count(texts), 2),
        'subject': row.get('subject') or '_aggregate',
    }


def assign_buckets(rows, q_key='question_length_bucket', c_key='choice_length_bucket'):
    q1, q2 = quantile_thresholds([r['question_word_count'] for r in rows])
    c1, c2 = quantile_thresholds([r['choice_avg_word_count'] for r in rows])
    for rec in rows:
        rec[q_key] = bucket(rec['question_word_count'], q1, q2)
        rec[c_key] = bucket(int(round(rec['choice_avg_word_count'])), c1, c2)
    return (q1, q2), (c1, c2)


def build_arc(spec):
    root = spec['root']
    raw_dir = root / 'raw'
    phase1_dir = root / 'phase1'
    raw_dir.mkdir(parents=True, exist_ok=True)
    phase1_dir.mkdir(parents=True, exist_ok=True)

    train_rows = [parse_arc(spec['train_split'], i, r) for i, r in enumerate(fetch_rows(spec['dataset'], spec['config'], spec['train_split']))]
    val_rows = [parse_arc(spec['val_split'], i, r) for i, r in enumerate(fetch_rows(spec['dataset'], spec['config'], spec['val_split']))]
    test_rows = [parse_arc(spec['test_split'], i, r) for i, r in enumerate(fetch_rows(spec['dataset'], spec['config'], spec['test_split']))]

    q_thresh, c_thresh = assign_buckets(train_rows)
    for rows in (val_rows, test_rows):
        for rec in rows:
            rec['question_length_bucket'] = bucket(rec['question_word_count'], *q_thresh)
            rec['choice_length_bucket'] = bucket(int(round(rec['choice_avg_word_count'])), *c_thresh)

    write_jsonl(raw_dir / 'train.normalized.jsonl', train_rows)
    write_jsonl(raw_dir / 'validation.normalized.jsonl', val_rows)
    write_jsonl(raw_dir / 'test.normalized.jsonl', test_rows)

    train_small = sorted(train_rows, key=lambda r: r['id'])
    val_small = stratified_sample(val_rows, spec['val_small_n'], random.Random(SEED + 11), lambda r: (r['question_length_bucket'], r['choice_length_bucket']))
    val_ids = {r['id'] for r in val_small}
    remaining = [r for r in val_rows if r['id'] not in val_ids]
    dev_analysis = stratified_sample(remaining, spec['dev_analysis_n'], random.Random(SEED + 12), lambda r: (r['question_length_bucket'], r['choice_length_bucket']))

    subsets = {
        'train_small': train_small,
        'val_small': val_small,
        'dev_analysis': dev_analysis,
        'test_frozen': test_rows,
    }
    for name, rows in subsets.items():
        write_jsonl(phase1_dir / f'{name}.jsonl', rows)

    manifest = {
        'benchmark': 'arc_challenge',
        'dataset': spec['dataset'],
        'config': spec['config'],
        'seed': SEED,
        'sampling': 'train_full_plus_validation_split',
        'question_length_thresholds': {'short_max': q_thresh[0], 'medium_max': q_thresh[1]},
        'choice_length_thresholds': {'short_max': c_thresh[0], 'medium_max': c_thresh[1]},
        'splits': {
            name: {
                'path': str((phase1_dir / f'{name}.jsonl').resolve()),
                'summary': summarize(rows, 'question_length_bucket', 'choice_length_bucket'),
            }
            for name, rows in subsets.items()
        },
        'notes': [
            spec['sampling_note'],
            'ARC-Challenge is added as an additional reasoning benchmark beyond GSM8K.',
            'Official ARC test is treated as frozen and must not be used for tuning.',
        ],
    }
    with (phase1_dir / 'manifest.json').open('w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


def build_hellaswag(spec):
    root = spec['root']
    raw_dir = root / 'raw'
    phase1_dir = root / 'phase1'
    raw_dir.mkdir(parents=True, exist_ok=True)
    phase1_dir.mkdir(parents=True, exist_ok=True)

    train_rows = [parse_hellaswag(spec['train_split'], i, r) for i, r in enumerate(fetch_rows(spec['dataset'], spec['config'], spec['train_split']))]
    val_rows = [parse_hellaswag(spec['val_split'], i, r) for i, r in enumerate(fetch_rows(spec['dataset'], spec['config'], spec['val_split']))]

    q_thresh, c_thresh = assign_buckets(train_rows, q_key='context_length_bucket', c_key='ending_length_bucket')
    for rec in val_rows:
        rec['context_length_bucket'] = bucket(rec['question_word_count'], *q_thresh)
        rec['ending_length_bucket'] = bucket(int(round(rec['choice_avg_word_count'])), *c_thresh)

    write_jsonl(raw_dir / 'train.normalized.jsonl', train_rows)
    write_jsonl(raw_dir / 'validation.normalized.jsonl', val_rows)

    train_small = stratified_sample(train_rows, spec['train_small_n'], random.Random(SEED + 21), lambda r: (r['context_length_bucket'], r['ending_length_bucket'], r['split_type']))
    train_ids = {r['id'] for r in train_small}
    remaining = [r for r in train_rows if r['id'] not in train_ids]
    val_small = stratified_sample(remaining, spec['val_small_n'], random.Random(SEED + 22), lambda r: (r['context_length_bucket'], r['ending_length_bucket'], r['split_type']))
    val_ids = {r['id'] for r in val_small}
    remaining = [r for r in remaining if r['id'] not in val_ids]
    dev_analysis = stratified_sample(remaining, spec['dev_analysis_n'], random.Random(SEED + 23), lambda r: (r['context_length_bucket'], r['ending_length_bucket'], r['split_type']))

    subsets = {
        'train_small': train_small,
        'val_small': val_small,
        'dev_analysis': dev_analysis,
        'test_frozen': val_rows,
    }
    for name, rows in subsets.items():
        write_jsonl(phase1_dir / f'{name}.jsonl', rows)

    manifest = {
        'benchmark': 'hellaswag',
        'dataset': spec['dataset'],
        'config': spec['config'],
        'seed': SEED,
        'sampling': 'train_subset_plus_validation_frozen_test',
        'context_length_thresholds': {'short_max': q_thresh[0], 'medium_max': q_thresh[1]},
        'ending_length_thresholds': {'short_max': c_thresh[0], 'medium_max': c_thresh[1]},
        'splits': {
            name: {
                'path': str((phase1_dir / f'{name}.jsonl').resolve()),
                'summary': summarize(rows, 'context_length_bucket', 'ending_length_bucket', extra_key='split_type'),
            }
            for name, rows in subsets.items()
        },
        'notes': [
            spec['sampling_note'],
            'HellaSwag is used as an additional commonsense / completion benchmark.',
            'Only labeled train + validation rows are used in phase 1.',
        ],
    }
    with (phase1_dir / 'manifest.json').open('w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


def build_mmlu(spec):
    root = spec['root']
    raw_dir = root / 'raw'
    phase1_dir = root / 'phase1'
    raw_dir.mkdir(parents=True, exist_ok=True)
    phase1_dir.mkdir(parents=True, exist_ok=True)

    train_rows = [parse_mmlu(spec['train_split'], i, r) for i, r in enumerate(fetch_rows(spec['dataset'], spec['config'], spec['train_split']))]
    dev_rows = [parse_mmlu(spec['val_split'], i, r) for i, r in enumerate(fetch_rows(spec['dataset'], spec['config'], spec['val_split']))]
    val_rows = [parse_mmlu(spec['test_split'], i, r) for i, r in enumerate(fetch_rows(spec['dataset'], spec['config'], spec['test_split']))]
    official_test_rows = [parse_mmlu(spec['extra_raw_split'], i, r) for i, r in enumerate(fetch_rows(spec['dataset'], spec['config'], spec['extra_raw_split']))]

    q_thresh, c_thresh = assign_buckets(train_rows)
    for rows in (dev_rows, val_rows, official_test_rows):
        for rec in rows:
            rec['question_length_bucket'] = bucket(rec['question_word_count'], *q_thresh)
            rec['choice_length_bucket'] = bucket(int(round(rec['choice_avg_word_count'])), *c_thresh)

    write_jsonl(raw_dir / 'auxiliary_train.normalized.jsonl', train_rows)
    write_jsonl(raw_dir / 'dev.normalized.jsonl', dev_rows)
    write_jsonl(raw_dir / 'validation.normalized.jsonl', val_rows)
    write_jsonl(raw_dir / 'test_official_full.jsonl', official_test_rows)

    train_small = stratified_sample(train_rows, spec['train_small_n'], random.Random(SEED + 31), lambda r: (r['subject'], r['question_length_bucket']))
    val_small = stratified_sample(dev_rows, spec['val_small_n'], random.Random(SEED + 32), lambda r: (r['subject'], r['question_length_bucket']))
    val_ids = {r['id'] for r in val_small}
    remaining = [r for r in dev_rows if r['id'] not in val_ids]
    dev_analysis = stratified_sample(remaining, spec['dev_analysis_n'], random.Random(SEED + 33), lambda r: (r['subject'], r['question_length_bucket']))

    subsets = {
        'train_small': train_small,
        'val_small': val_small,
        'dev_analysis': dev_analysis,
        'test_frozen': val_rows,
    }
    for name, rows in subsets.items():
        write_jsonl(phase1_dir / f'{name}.jsonl', rows)

    manifest = {
        'benchmark': 'mmlu',
        'dataset': spec['dataset'],
        'config': spec['config'],
        'seed': SEED,
        'sampling': 'auxiliary_train_subset_plus_dev_split',
        'question_length_thresholds': {'short_max': q_thresh[0], 'medium_max': q_thresh[1]},
        'choice_length_thresholds': {'short_max': c_thresh[0], 'medium_max': c_thresh[1]},
        'splits': {
            name: {
                'path': str((phase1_dir / f'{name}.jsonl').resolve()),
                'summary': summarize(rows, 'question_length_bucket', 'choice_length_bucket', extra_key='subject'),
            }
            for name, rows in subsets.items()
        },
        'extra_raw_paths': {
            'official_test_full': str((raw_dir / 'test_official_full.jsonl').resolve()),
        },
        'notes': [
            spec['sampling_note'],
            'MMLU is a support benchmark for generality, not the primary paper headline.',
            'The aggregated auxiliary_train split does not provide useful subject labels in this HF view and is therefore sampled primarily by question length.',
            'Phase-1 frozen test uses validation; official test is mirrored separately and left untouched.',
        ],
    }
    with (phase1_dir / 'manifest.json').open('w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--benchmark', choices=sorted(BENCHMARKS), required=True)
    args = ap.parse_args()

    spec = BENCHMARKS[args.benchmark]
    if args.benchmark == 'arc_challenge':
        build_arc(spec)
    elif args.benchmark == 'hellaswag':
        build_hellaswag(spec)
    else:
        build_mmlu(spec)


if __name__ == '__main__':
    main()
