#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


def load_jsonl(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_prediction_text(text: str) -> str:
    text = text.strip()
    if text.startswith('```'):
        text = re.sub(r'^```[a-zA-Z0-9_+-]*\n?', '', text)
        text = re.sub(r'\n?```$', '', text)
        text = text.strip()
    return text


def compile_regex(expr: str):
    try:
        return re.compile(expr), None
    except re.error as e:
        return None, str(e)


def semantic_match(compiled, matches, non_matches, mode='search'):
    matcher = compiled.search if mode == 'search' else compiled.fullmatch
    for s in matches:
        if matcher(s) is None:
            return False
    for s in non_matches:
        if matcher(s) is not None:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ground-truth', required=True)
    ap.add_argument('--predictions', required=True)
    ap.add_argument('--prediction-key', default='prediction')
    ap.add_argument('--output', default='')
    ap.add_argument('--match-mode', choices=['search', 'fullmatch'], default='search')
    args = ap.parse_args()

    gt_rows = load_jsonl(args.ground_truth)
    pred_rows = load_jsonl(args.predictions)
    pred_by_id = {row['id']: row for row in pred_rows}

    results = []
    exact = 0
    semantic = 0
    invalid = 0
    missing = 0

    for gt in gt_rows:
        rid = gt['id']
        pred_row = pred_by_id.get(rid)
        if pred_row is None:
            missing += 1
            results.append({
                'id': rid,
                'status': 'missing_prediction'
            })
            continue

        pred = normalize_prediction_text(str(pred_row.get(args.prediction_key, '')))
        gold = gt['expression']
        ex = pred == gold
        comp, err = compile_regex(pred)
        inv = comp is None
        sem = False if inv else semantic_match(comp, gt['matches'], gt['non_matches'], mode=args.match_mode)

        exact += int(ex)
        semantic += int(sem)
        invalid += int(inv)
        results.append({
            'id': rid,
            'gold_expression': gold,
            'predicted_expression': pred,
            'exact_match': ex,
            'semantic_match': sem,
            'invalid_regex': inv,
            'compile_error': err,
        })

    denom = len(gt_rows)
    summary = {
        'ground_truth_path': str(Path(args.ground_truth).resolve()),
        'predictions_path': str(Path(args.predictions).resolve()),
        'count_ground_truth': denom,
        'count_predictions': len(pred_rows),
        'missing_prediction_count': missing,
        'exact_match': exact / denom if denom else 0.0,
        'semantic_match': semantic / denom if denom else 0.0,
        'invalid_regex_rate': invalid / denom if denom else 0.0,
        'match_mode': args.match_mode,
    }

    payload = {
        'summary': summary,
        'results': results,
    }

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
