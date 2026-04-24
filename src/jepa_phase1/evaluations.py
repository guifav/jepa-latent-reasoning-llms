from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from .configs import RunConfig
from .data import pad_to_length

OPTION_LABELS = ['A', 'B', 'C', 'D', 'E', 'F']


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith('```'):
        text = re.sub(r'^```[a-zA-Z0-9_+-]*\n?', '', text)
        text = re.sub(r'\n?```$', '', text)
        text = text.strip()
    return text


def normalize_scalar_answer(text: str) -> str:
    text = strip_code_fences(text)
    if 'Final answer:' in text:
        text = text.split('Final answer:')[-1].strip()
    matches = re.findall(r'-?\d[\d,]*(?:\.\d+)?', text)
    if not matches:
        return ''
    answer = matches[-1].replace(',', '').strip()
    if answer.endswith('.'):
        answer = answer[:-1]
    if answer.startswith('+'):
        answer = answer[1:]
    return answer


def normalize_regex_text(text: str) -> str:
    text = strip_code_fences(text)
    if 'Regex:' in text:
        text = text.split('Regex:')[-1].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ''
    first = lines[0]
    if first.lower().startswith('regex:'):
        first = first.split(':', 1)[1].strip()
    return first


def normalize_option_prediction(text: str, raw_row: dict[str, Any]) -> str:
    text = strip_code_fences(text)
    text = text.replace('**', ' ').strip()
    valid_labels = raw_row.get('choice_labels') or OPTION_LABELS

    if 'Answer:' in text:
        text = text.split('Answer:')[-1].strip()
    candidate = text.strip()

    patterns = [
        r'^[\(\[]?([A-F])[\)\].:-]?\b',
        r'\boption\s+([A-F])\b',
        r'\bchoice\s+([A-F])\b',
        r'\b([A-F])\b',
    ]
    for pattern in patterns:
        m = re.search(pattern, candidate, flags=re.IGNORECASE)
        if m:
            label = m.group(1).upper()
            if label in valid_labels:
                return label

    normalized_text = candidate.lower().strip(' .:-')
    for label, choice in zip(raw_row.get('choice_labels', []), raw_row.get('choice_texts', [])):
        choice_norm = choice.lower().strip(' .:-')
        if normalized_text == choice_norm or normalized_text.startswith(choice_norm):
            return label

    return ''


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def derive_raw_phase1_path(view_path: str | Path, benchmark: str) -> Path:
    view_path = Path(view_path)
    parts = list(view_path.parts)
    if 'phase1_views' in parts:
        idx = parts.index('phase1_views')
        parts[idx] = 'phase1'
        del parts[idx + 1]
        return Path(*parts)
    candidate = Path('/root/workspace/jepa/data') / benchmark / 'phase1' / view_path.name
    if candidate.exists():
        return candidate
    name = view_path.name
    if 'train' in name:
        fallback = 'train_small.jsonl'
    elif 'val' in name:
        fallback = 'val_small.jsonl'
    elif 'dev' in name:
        fallback = 'dev_analysis.jsonl'
    else:
        fallback = 'test_frozen.jsonl' if benchmark in {'regexeval', 'arc_challenge', 'hellaswag', 'mmlu'} else 'test_official_full.jsonl'
    return Path('/root/workspace/jepa/data') / benchmark / 'phase1' / fallback


def build_prompt_text(kind: str, row: dict[str, Any]) -> str:
    if kind == 'lm':
        return row['input_text']
    if kind == 'coupled':
        return row['generation_prompt_text']
    return row['condition_text']


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


def batch_tokenize(tokenizer, prompts: list[str], max_length: int):
    encoded = [tokenizer.encode(p, add_special_tokens=False, truncation=True, max_length=max_length) for p in prompts]
    input_ids = pad_to_length(encoded, tokenizer.pad_token_id, side='left')
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    return input_ids, attention_mask


@torch.no_grad()
def generate_predictions(model, tokenizer, prompts: list[str], max_input_tokens: int, max_new_tokens: int, device: torch.device, kind: str):
    input_ids, attention_mask = batch_tokenize(tokenizer, prompts, max_input_tokens)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    generated = model.generate(
        input_ids if kind != 'decoupled' else input_ids,
        attention_mask,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    texts = []
    if kind == 'decoupled':
        for seq in generated:
            texts.append(tokenizer.decode(seq, skip_special_tokens=True).strip())
    else:
        prompt_lengths = attention_mask.sum(dim=1).tolist()
        for seq, prompt_len in zip(generated, prompt_lengths):
            texts.append(tokenizer.decode(seq[prompt_len:], skip_special_tokens=True).strip())
    return texts


def evaluate_gsm8k(model, tokenizer, cfg: RunConfig, kind: str, device: torch.device, limit: int | None = None):
    view_rows = load_jsonl(cfg.data['val_path'])
    raw_rows = load_jsonl(derive_raw_phase1_path(cfg.data['val_path'], cfg.benchmark))
    raw_by_id = {row['id']: row for row in raw_rows}
    max_input = cfg.training.get('max_input_tokens') or cfg.training.get('max_packed_input_tokens') or cfg.training.get('max_condition_tokens') or 384
    max_new = cfg.training.get('max_target_tokens') or cfg.training.get('max_generation_target_tokens') or cfg.training.get('max_talker_target_tokens') or 64
    rows = view_rows[:limit] if limit else view_rows
    prompts = [build_prompt_text(kind, row) for row in rows]
    predictions = generate_predictions(model, tokenizer, prompts, max_input, max_new, device, kind)

    correct = 0
    by_length = defaultdict(lambda: {'correct': 0, 'total': 0})
    by_answer_type = defaultdict(lambda: {'correct': 0, 'total': 0})
    records = []
    for row, pred_text in zip(rows, predictions):
        raw = raw_by_id[row['id']]
        pred = normalize_scalar_answer(pred_text)
        gold = raw['final_answer_normalized']
        is_correct = pred == gold
        correct += int(is_correct)
        by_length[raw['length_bucket']]['correct'] += int(is_correct)
        by_length[raw['length_bucket']]['total'] += 1
        by_answer_type[raw['answer_type']]['correct'] += int(is_correct)
        by_answer_type[raw['answer_type']]['total'] += 1
        records.append({
            'id': row['id'],
            'prediction_raw': pred_text,
            'prediction_normalized': pred,
            'gold': gold,
            'correct': is_correct,
        })

    total = len(rows)
    return {
        'metric': 'final_answer_accuracy',
        'count': total,
        'accuracy': correct / total if total else 0.0,
        'length_bucket_accuracy': {k: v['correct'] / v['total'] for k, v in by_length.items()},
        'answer_type_accuracy': {k: v['correct'] / v['total'] for k, v in by_answer_type.items()},
        'predictions': records[: min(10, len(records))],
        'partial_eval': total != len(view_rows),
    }


def evaluate_regexeval(model, tokenizer, cfg: RunConfig, kind: str, device: torch.device, limit: int | None = None):
    view_rows = load_jsonl(cfg.data['val_path'])
    raw_rows = load_jsonl(derive_raw_phase1_path(cfg.data['val_path'], cfg.benchmark))
    raw_by_id = {row['id']: row for row in raw_rows}
    max_input = cfg.training.get('max_input_tokens') or cfg.training.get('max_packed_input_tokens') or cfg.training.get('max_condition_tokens') or 384
    max_new = cfg.training.get('max_target_tokens') or cfg.training.get('max_generation_target_tokens') or cfg.training.get('max_talker_target_tokens') or 64
    rows = view_rows[:limit] if limit else view_rows
    prompts = [build_prompt_text(kind, row) for row in rows]
    predictions = generate_predictions(model, tokenizer, prompts, max_input, max_new, device, kind)

    exact = 0
    semantic = 0
    invalid = 0
    by_prompt = defaultdict(lambda: {'semantic': 0, 'total': 0})
    by_regex = defaultdict(lambda: {'semantic': 0, 'total': 0})
    records = []
    for row, pred_text in zip(rows, predictions):
        raw = raw_by_id[row['id']]
        pred = normalize_regex_text(pred_text)
        gold = raw['expression']
        is_exact = pred == gold
        compiled, err = compile_regex(pred)
        is_invalid = compiled is None
        is_semantic = False if is_invalid else semantic_match(compiled, raw['matches'], raw['non_matches'], mode='search')
        exact += int(is_exact)
        semantic += int(is_semantic)
        invalid += int(is_invalid)
        by_prompt[raw['prompt_length_bucket']]['semantic'] += int(is_semantic)
        by_prompt[raw['prompt_length_bucket']]['total'] += 1
        by_regex[raw['regex_length_bucket']]['semantic'] += int(is_semantic)
        by_regex[raw['regex_length_bucket']]['total'] += 1
        records.append({
            'id': row['id'],
            'prediction_raw': pred_text,
            'prediction_normalized': pred,
            'gold': gold,
            'exact_match': is_exact,
            'semantic_match': is_semantic,
            'invalid_regex': is_invalid,
            'compile_error': err,
        })

    total = len(rows)
    return {
        'metric': 'semantic_match',
        'count': total,
        'semantic_match': semantic / total if total else 0.0,
        'exact_match': exact / total if total else 0.0,
        'invalid_regex_rate': invalid / total if total else 0.0,
        'prompt_length_bucket_semantic': {k: v['semantic'] / v['total'] for k, v in by_prompt.items()},
        'regex_length_bucket_semantic': {k: v['semantic'] / v['total'] for k, v in by_regex.items()},
        'predictions': records[: min(10, len(records))],
        'partial_eval': total != len(view_rows),
        'match_mode': 'search',
    }


def evaluate_mcq(model, tokenizer, cfg: RunConfig, kind: str, device: torch.device, *, limit: int | None = None, benchmark: str):
    view_rows = load_jsonl(cfg.data['val_path'])
    raw_rows = load_jsonl(derive_raw_phase1_path(cfg.data['val_path'], cfg.benchmark))
    raw_by_id = {row['id']: row for row in raw_rows}
    max_input = cfg.training.get('max_input_tokens') or cfg.training.get('max_packed_input_tokens') or cfg.training.get('max_condition_tokens') or 384
    max_new = cfg.training.get('max_target_tokens') or cfg.training.get('max_generation_target_tokens') or cfg.training.get('max_talker_target_tokens') or 32
    rows = view_rows[:limit] if limit else view_rows
    prompts = [build_prompt_text(kind, row) for row in rows]
    predictions = generate_predictions(model, tokenizer, prompts, max_input, max_new, device, kind)

    correct = 0
    by_primary = defaultdict(lambda: {'correct': 0, 'total': 0})
    by_secondary = defaultdict(lambda: {'correct': 0, 'total': 0})
    by_subject = defaultdict(lambda: {'correct': 0, 'total': 0})
    records = []

    if benchmark == 'arc_challenge':
        primary_key = 'question_length_bucket'
        secondary_key = 'choice_length_bucket'
    elif benchmark == 'hellaswag':
        primary_key = 'context_length_bucket'
        secondary_key = 'ending_length_bucket'
    else:
        primary_key = 'question_length_bucket'
        secondary_key = 'choice_length_bucket'

    for row, pred_text in zip(rows, predictions):
        raw = raw_by_id[row['id']]
        pred = normalize_option_prediction(pred_text, raw)
        gold = raw['answer_label']
        is_correct = pred == gold
        correct += int(is_correct)
        by_primary[raw[primary_key]]['correct'] += int(is_correct)
        by_primary[raw[primary_key]]['total'] += 1
        by_secondary[raw[secondary_key]]['correct'] += int(is_correct)
        by_secondary[raw[secondary_key]]['total'] += 1
        if benchmark == 'mmlu':
            by_subject[raw['subject']]['correct'] += int(is_correct)
            by_subject[raw['subject']]['total'] += 1
        records.append({
            'id': row['id'],
            'prediction_raw': pred_text,
            'prediction_normalized': pred,
            'gold': gold,
            'correct': is_correct,
        })

    total = len(rows)
    payload = {
        'metric': 'mcq_accuracy',
        'count': total,
        'accuracy': correct / total if total else 0.0,
        'predictions': records[: min(10, len(records))],
        'partial_eval': total != len(view_rows),
    }
    if benchmark == 'arc_challenge':
        payload['question_length_bucket_accuracy'] = {k: v['correct'] / v['total'] for k, v in by_primary.items()}
        payload['choice_length_bucket_accuracy'] = {k: v['correct'] / v['total'] for k, v in by_secondary.items()}
    elif benchmark == 'hellaswag':
        payload['context_length_bucket_accuracy'] = {k: v['correct'] / v['total'] for k, v in by_primary.items()}
        payload['ending_length_bucket_accuracy'] = {k: v['correct'] / v['total'] for k, v in by_secondary.items()}
    else:
        subject_acc = {k: v['correct'] / v['total'] for k, v in by_subject.items()}
        payload['question_length_bucket_accuracy'] = {k: v['correct'] / v['total'] for k, v in by_primary.items()}
        payload['subject_accuracy'] = subject_acc
        payload['macro_subject_accuracy'] = sum(subject_acc.values()) / max(1, len(subject_acc))
    return payload


def evaluate_benchmark(model, tokenizer, cfg: RunConfig, kind: str, device: torch.device):
    val_rows = load_jsonl(cfg.data['val_path'])
    limit = cfg.evaluation.get('generation_eval_max_examples')
    if limit is None and len(val_rows) > 64 and device.type == 'cpu':
        limit = 32
    if cfg.benchmark == 'gsm8k':
        return evaluate_gsm8k(model, tokenizer, cfg, kind, device, limit=limit)
    if cfg.benchmark == 'regexeval':
        return evaluate_regexeval(model, tokenizer, cfg, kind, device, limit=limit)
    if cfg.benchmark in {'arc_challenge', 'hellaswag', 'mmlu'}:
        return evaluate_mcq(model, tokenizer, cfg, kind, device, limit=limit, benchmark=cfg.benchmark)
    return {'status': 'unsupported_benchmark', 'benchmark': cfg.benchmark}
