#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path('/root/workspace/jepa/data/regexeval/phase1')
OUT = Path('/root/workspace/jepa/data/regexeval/phase1_views')
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


def render_prompt(prompt):
    return f"Requirement:\n{prompt}"


def render_examples(row):
    positives = '\n'.join(f"+ {x}" for x in row['matches'])
    negatives = '\n'.join(f"- {x}" for x in row['non_matches'])
    return f"Should match:\n{positives}\n\nShould not match:\n{negatives}"


def build_lm_refined(row):
    prompt = (
        "Write a regular expression that satisfies the requirement below. "
        "Return only the regex.\n\n"
        f"{render_prompt(row['refined_prompt'])}\n\nRegex:"
    )
    return {
        'id': row['id'],
        'benchmark': 'regexeval',
        'lane': 'lm_refined_to_expression',
        'input_text': prompt,
        'target_text': row['expression'],
        'metadata': {
            'prompt_length_bucket': row['prompt_length_bucket'],
            'regex_length_bucket': row['regex_length_bucket'],
            'refined_prompt_word_count': row['refined_prompt_word_count'],
            'regex_char_count': row['regex_char_count'],
        },
    }


def build_lm_raw(row):
    prompt = (
        "Write a regular expression that satisfies the requirement below. "
        "Return only the regex.\n\n"
        f"{render_prompt(row['raw_prompt'])}\n\nRegex:"
    )
    return {
        'id': row['id'],
        'benchmark': 'regexeval',
        'lane': 'lm_raw_to_expression',
        'input_text': prompt,
        'target_text': row['expression'],
        'metadata': {
            'prompt_length_bucket': row['prompt_length_bucket'],
            'regex_length_bucket': row['regex_length_bucket'],
            'refined_prompt_word_count': row['refined_prompt_word_count'],
            'regex_char_count': row['regex_char_count'],
        },
    }


def build_coupled(row):
    return {
        'id': row['id'],
        'benchmark': 'regexeval',
        'lane': 'coupled_prompt_expression',
        'view_a_name': 'refined_prompt',
        'view_a_text': render_prompt(row['refined_prompt']),
        'view_b_name': 'expression',
        'view_b_text': row['expression'],
        'aux_view_name': 'raw_prompt',
        'aux_view_text': render_prompt(row['raw_prompt']),
        'generation_prompt_text': (
            "Write a regular expression that satisfies the requirement below. "
            "Return only the regex.\n\n"
            f"{render_prompt(row['refined_prompt'])}\n\nRegex:"
        ),
        'generation_target_text': row['expression'],
        'behavioral_examples_text': render_examples(row),
        'metadata': {
            'prompt_length_bucket': row['prompt_length_bucket'],
            'regex_length_bucket': row['regex_length_bucket'],
            'refined_prompt_word_count': row['refined_prompt_word_count'],
            'regex_char_count': row['regex_char_count'],
        },
    }


def build_decoupled(row):
    return {
        'id': row['id'],
        'benchmark': 'regexeval',
        'lane': 'decoupled_reasoner',
        'condition_text': render_prompt(row['refined_prompt']),
        'aux_condition_text': render_prompt(row['raw_prompt']),
        'talker_target': row['expression'],
        'behavioral_examples_text': render_examples(row),
        'metadata': {
            'prompt_length_bucket': row['prompt_length_bucket'],
            'regex_length_bucket': row['regex_length_bucket'],
            'refined_prompt_word_count': row['refined_prompt_word_count'],
            'regex_char_count': row['regex_char_count'],
        },
    }


def main():
    manifest = {'lanes': {}}
    builders = {
        'lm_refined_to_expression': build_lm_refined,
        'lm_raw_to_expression': build_lm_raw,
        'coupled_prompt_expression': build_coupled,
        'decoupled_reasoner': build_decoupled,
    }
    for lane, builder in builders.items():
        manifest['lanes'][lane] = {}
        for split in SPLITS:
            rows = load_jsonl(ROOT / f'{split}.jsonl')
            out_rows = [builder(r) for r in rows]
            out_path = OUT / lane / f'{split}.jsonl'
            write_jsonl(out_path, out_rows)
            manifest['lanes'][lane][split] = {
                'path': str(out_path.resolve()),
                'count': len(out_rows),
            }
    with (OUT / 'manifest.json').open('w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
