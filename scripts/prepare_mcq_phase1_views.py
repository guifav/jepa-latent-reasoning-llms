#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

ROOTS = {
    'arc_challenge': Path('/root/workspace/jepa/data/arc_challenge/phase1'),
    'hellaswag': Path('/root/workspace/jepa/data/hellaswag/phase1'),
    'mmlu': Path('/root/workspace/jepa/data/mmlu/phase1'),
}
OUT_ROOTS = {
    'arc_challenge': Path('/root/workspace/jepa/data/arc_challenge/phase1_views'),
    'hellaswag': Path('/root/workspace/jepa/data/hellaswag/phase1_views'),
    'mmlu': Path('/root/workspace/jepa/data/mmlu/phase1_views'),
}
SPLITS = ['train_small', 'val_small', 'dev_analysis', 'test_frozen']


def load_jsonl(path: Path):
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def render_choices(row):
    return '\n'.join(f"{label}. {text}" for label, text in zip(row['choice_labels'], row['choice_texts']))


def base_metadata(row, benchmark):
    meta = {
        'choice_length_bucket': row.get('choice_length_bucket'),
        'question_word_count': row.get('question_word_count'),
        'choice_avg_word_count': row.get('choice_avg_word_count'),
    }
    if benchmark == 'arc_challenge':
        meta['question_length_bucket'] = row.get('question_length_bucket')
    elif benchmark == 'hellaswag':
        meta['context_length_bucket'] = row.get('context_length_bucket')
        meta['ending_length_bucket'] = row.get('ending_length_bucket')
        meta['activity_label'] = row.get('activity_label')
        meta['split_type'] = row.get('split_type')
    elif benchmark == 'mmlu':
        meta['question_length_bucket'] = row.get('question_length_bucket')
        meta['subject'] = row.get('subject')
    return meta


def build_arc_lm(row):
    prompt = (
        'Answer the following multiple-choice science question. '
        'Return only the option letter.\n\n'
        f"Question:\n{row['question']}\n\nChoices:\n{render_choices(row)}\n\nAnswer:"
    )
    return {
        'id': row['id'],
        'benchmark': 'arc_challenge',
        'lane': 'lm_question_to_label',
        'split': row['split'],
        'input_text': prompt,
        'target_text': f"Answer: {row['answer_label']}",
        'metadata': base_metadata(row, 'arc_challenge'),
    }


def build_arc_coupled(row):
    return {
        'id': row['id'],
        'benchmark': 'arc_challenge',
        'lane': 'coupled_question_correct_option',
        'split': row['split'],
        'view_a_name': 'question',
        'view_a_text': f"Question:\n{row['question']}\n\nChoices:\n{render_choices(row)}",
        'view_b_name': 'correct_option',
        'view_b_text': f"Correct option ({row['answer_label']}): {row['answer_text']}",
        'generation_prompt_text': (
            'Answer the following multiple-choice science question. '
            'Return only the option letter.\n\n'
            f"Question:\n{row['question']}\n\nChoices:\n{render_choices(row)}\n\nAnswer:"
        ),
        'generation_target_text': f"Answer: {row['answer_label']}",
        'metadata': base_metadata(row, 'arc_challenge'),
    }


def build_arc_decoupled(row):
    return {
        'id': row['id'],
        'benchmark': 'arc_challenge',
        'lane': 'decoupled_reasoner',
        'split': row['split'],
        'condition_text': (
            'Answer the following multiple-choice science question. '
            'Return only the option letter.\n\n'
            f"Question:\n{row['question']}\n\nChoices:\n{render_choices(row)}\n\nAnswer:"
        ),
        'talker_target': f"Answer: {row['answer_label']}",
        'metadata': base_metadata(row, 'arc_challenge'),
    }


def build_hellaswag_lm(row):
    prompt = (
        'Choose the most plausible continuation. '
        'Return only the option letter.\n\n'
        f"Context:\n{row['context']}\n\nOptions:\n{render_choices(row)}\n\nAnswer:"
    )
    return {
        'id': row['id'],
        'benchmark': 'hellaswag',
        'lane': 'lm_context_to_label',
        'split': row['split'],
        'input_text': prompt,
        'target_text': f"Answer: {row['answer_label']}",
        'metadata': base_metadata(row, 'hellaswag'),
    }


def build_hellaswag_coupled(row):
    return {
        'id': row['id'],
        'benchmark': 'hellaswag',
        'lane': 'coupled_context_correct_ending',
        'split': row['split'],
        'view_a_name': 'context',
        'view_a_text': f"Context:\n{row['context']}\n\nOptions:\n{render_choices(row)}",
        'view_b_name': 'correct_ending',
        'view_b_text': f"Correct ending ({row['answer_label']}): {row['answer_text']}",
        'generation_prompt_text': (
            'Choose the most plausible continuation. '
            'Return only the option letter.\n\n'
            f"Context:\n{row['context']}\n\nOptions:\n{render_choices(row)}\n\nAnswer:"
        ),
        'generation_target_text': f"Answer: {row['answer_label']}",
        'metadata': base_metadata(row, 'hellaswag'),
    }


def build_hellaswag_decoupled(row):
    return {
        'id': row['id'],
        'benchmark': 'hellaswag',
        'lane': 'decoupled_reasoner',
        'split': row['split'],
        'condition_text': (
            'Choose the most plausible continuation. '
            'Return only the option letter.\n\n'
            f"Context:\n{row['context']}\n\nOptions:\n{render_choices(row)}\n\nAnswer:"
        ),
        'talker_target': f"Answer: {row['answer_label']}",
        'metadata': base_metadata(row, 'hellaswag'),
    }


def build_mmlu_lm(row):
    prompt = (
        'Answer the following multiple-choice question. '
        'Return only the option letter.\n\n'
        f"Question:\n{row['question']}\n\nChoices:\n{render_choices(row)}\n\nAnswer:"
    )
    return {
        'id': row['id'],
        'benchmark': 'mmlu',
        'lane': 'lm_question_to_label',
        'split': row['split'],
        'input_text': prompt,
        'target_text': f"Answer: {row['answer_label']}",
        'metadata': base_metadata(row, 'mmlu'),
    }


def build_mmlu_coupled(row):
    return {
        'id': row['id'],
        'benchmark': 'mmlu',
        'lane': 'coupled_question_correct_option',
        'split': row['split'],
        'view_a_name': 'question',
        'view_a_text': f"Question:\n{row['question']}\n\nChoices:\n{render_choices(row)}",
        'view_b_name': 'correct_option',
        'view_b_text': f"Correct option ({row['answer_label']}): {row['answer_text']}",
        'generation_prompt_text': (
            'Answer the following multiple-choice question. '
            'Return only the option letter.\n\n'
            f"Question:\n{row['question']}\n\nChoices:\n{render_choices(row)}\n\nAnswer:"
        ),
        'generation_target_text': f"Answer: {row['answer_label']}",
        'metadata': base_metadata(row, 'mmlu'),
    }


def build_mmlu_decoupled(row):
    return {
        'id': row['id'],
        'benchmark': 'mmlu',
        'lane': 'decoupled_reasoner',
        'split': row['split'],
        'condition_text': (
            'Answer the following multiple-choice question. '
            'Return only the option letter.\n\n'
            f"Question:\n{row['question']}\n\nChoices:\n{render_choices(row)}\n\nAnswer:"
        ),
        'talker_target': f"Answer: {row['answer_label']}",
        'metadata': base_metadata(row, 'mmlu'),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--benchmark', choices=sorted(ROOTS), required=True)
    args = ap.parse_args()

    root = ROOTS[args.benchmark]
    out_root = OUT_ROOTS[args.benchmark]

    if args.benchmark == 'arc_challenge':
        builders = {
            'lm_question_to_label': build_arc_lm,
            'coupled_question_correct_option': build_arc_coupled,
            'decoupled_reasoner': build_arc_decoupled,
        }
    elif args.benchmark == 'hellaswag':
        builders = {
            'lm_context_to_label': build_hellaswag_lm,
            'coupled_context_correct_ending': build_hellaswag_coupled,
            'decoupled_reasoner': build_hellaswag_decoupled,
        }
    else:
        builders = {
            'lm_question_to_label': build_mmlu_lm,
            'coupled_question_correct_option': build_mmlu_coupled,
            'decoupled_reasoner': build_mmlu_decoupled,
        }

    manifest = {'benchmark': args.benchmark, 'lanes': {}}
    for lane, builder in builders.items():
        manifest['lanes'][lane] = {}
        for split in SPLITS:
            rows = load_jsonl(root / f'{split}.jsonl')
            out_rows = [builder(r) for r in rows]
            out_path = out_root / lane / f'{split}.jsonl'
            write_jsonl(out_path, out_rows)
            manifest['lanes'][lane][split] = {
                'path': str(out_path.resolve()),
                'count': len(out_rows),
            }

    with (out_root / 'manifest.json').open('w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
