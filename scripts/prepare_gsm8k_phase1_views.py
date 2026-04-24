#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path('/root/workspace/jepa/data/gsm8k/phase1')
OUT = Path('/root/workspace/jepa/data/gsm8k/phase1_views')
SPLITS = ['train_small', 'val_small', 'dev_analysis', 'test_official_full']


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


def render_question(row):
    return f"Question:\n{row['question']}"


def render_rationale(row):
    return f"Solution:\n{row['solution_rationale']}"


def render_final_answer(row):
    return f"Final answer: {row['final_answer_normalized']}"


def build_lm_answer_only(row):
    prompt = (
        "Solve the following grade-school math problem. "
        "Return only the final numeric answer.\n\n"
        f"{render_question(row)}\n\nFinal answer:"
    )
    return {
        'id': row['id'],
        'benchmark': 'gsm8k',
        'lane': 'lm_answer_only',
        'split': row['split'],
        'input_text': prompt,
        'target_text': row['final_answer_normalized'],
        'metadata': {
            'length_bucket': row['length_bucket'],
            'answer_type': row['answer_type'],
            'question_word_count': row['question_word_count'],
            'rationale_word_count': row['rationale_word_count'],
        },
    }


def build_lm_rationale_answer(row):
    prompt = (
        "Solve the following grade-school math problem. "
        "Show the solution and end with the final numeric answer.\n\n"
        f"{render_question(row)}\n\nSolution:"
    )
    target = f"{row['solution_rationale']}\n\nFinal answer: {row['final_answer_normalized']}"
    return {
        'id': row['id'],
        'benchmark': 'gsm8k',
        'lane': 'lm_rationale_answer',
        'split': row['split'],
        'input_text': prompt,
        'target_text': target,
        'metadata': {
            'length_bucket': row['length_bucket'],
            'answer_type': row['answer_type'],
            'question_word_count': row['question_word_count'],
            'rationale_word_count': row['rationale_word_count'],
        },
    }


def build_coupled(row):
    return {
        'id': row['id'],
        'benchmark': 'gsm8k',
        'lane': 'coupled_question_rationale',
        'split': row['split'],
        'view_a_name': 'question',
        'view_a_text': render_question(row),
        'view_b_name': 'solution_rationale',
        'view_b_text': render_rationale(row),
        'generation_target_text': row['final_answer_normalized'],
        'generation_prompt_text': (
            "Solve the following grade-school math problem. "
            "Use the latent alignment objective to improve reasoning, but output only the final numeric answer.\n\n"
            f"{render_question(row)}\n\nFinal answer:"
        ),
        'metadata': {
            'length_bucket': row['length_bucket'],
            'answer_type': row['answer_type'],
            'question_word_count': row['question_word_count'],
            'rationale_word_count': row['rationale_word_count'],
        },
    }


def build_decoupled(row):
    return {
        'id': row['id'],
        'benchmark': 'gsm8k',
        'lane': 'decoupled_reasoner',
        'split': row['split'],
        'condition_text': render_question(row),
        'talker_rationale_target': row['solution_rationale'],
        'talker_answer_target': row['final_answer_normalized'],
        'talker_joint_target': f"{row['solution_rationale']}\n\nFinal answer: {row['final_answer_normalized']}",
        'evaluation_answer': row['final_answer_normalized'],
        'metadata': {
            'length_bucket': row['length_bucket'],
            'answer_type': row['answer_type'],
            'question_word_count': row['question_word_count'],
            'rationale_word_count': row['rationale_word_count'],
        },
    }


def main():
    manifest = {'lanes': {}}
    builders = {
        'lm_answer_only': build_lm_answer_only,
        'lm_rationale_answer': build_lm_rationale_answer,
        'coupled_question_rationale': build_coupled,
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
