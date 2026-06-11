#!/usr/bin/env python3
"""Unit tests for repo-relative path resolution (issue #4).

Runs with plain `python tests/test_path_resolution.py` (also pytest-compatible).
No datasets and no ML stack required.
"""
from __future__ import annotations

import json
import sys
import tempfile
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))


def _install_stubs():
    if 'peft' not in sys.modules:
        peft = types.ModuleType('peft')
        peft.LoraConfig = object
        peft.get_peft_model = lambda model, cfg: model
        sys.modules['peft'] = peft
    if 'transformers' not in sys.modules:
        transformers = types.ModuleType('transformers')

        class _Unavailable:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                raise RuntimeError('real checkpoints are not available in unit tests')

        transformers.AutoModelForCausalLM = _Unavailable
        transformers.AutoTokenizer = _Unavailable
        sys.modules['transformers'] = transformers


_install_stubs()

from jepa_phase1.configs import load_run_config  # noqa: E402
from jepa_phase1.evaluations import derive_raw_phase1_path  # noqa: E402


def _write_config(payload: dict) -> Path:
    handle = tempfile.NamedTemporaryFile('w', suffix='.json', delete=False, encoding='utf-8')
    json.dump(payload, handle)
    handle.close()
    return Path(handle.name)


BASE_PAYLOAD = {
    'run_name': 'test_run',
    'benchmark': 'gsm8k',
    'backbone_checkpoint': 'tiny-test',
    'training': {},
    'evaluation': {},
}


def test_relative_data_paths_resolve_against_repo_root():
    cfg_path = _write_config({
        **BASE_PAYLOAD,
        'data': {
            'train_path': 'data/gsm8k/phase1_views/lm_answer_only/train_small.jsonl',
            'val_path': 'data/gsm8k/phase1_views/lm_answer_only/val_small.jsonl',
        },
    })
    cfg = load_run_config(cfg_path)
    expected = REPO_ROOT / 'data/gsm8k/phase1_views/lm_answer_only/train_small.jsonl'
    assert Path(cfg.data['train_path']) == expected, cfg.data['train_path']
    assert Path(cfg.data['train_path']).is_absolute()


def test_absolute_data_paths_pass_through():
    cfg_path = _write_config({
        **BASE_PAYLOAD,
        'data': {'train_path': '/tmp/somewhere/train.jsonl', 'val_path': '/tmp/somewhere/val.jsonl'},
    })
    cfg = load_run_config(cfg_path)
    assert cfg.data['train_path'] == '/tmp/somewhere/train.jsonl'


def test_derive_raw_path_from_views_directory():
    view = REPO_ROOT / 'data/gsm8k/phase1_views/lm_answer_only/val_small.jsonl'
    derived = derive_raw_phase1_path(view, 'gsm8k')
    assert derived == REPO_ROOT / 'data/gsm8k/phase1/val_small.jsonl', derived


def test_derive_raw_path_fallback_stays_inside_repo_data_root():
    view = REPO_ROOT / 'data/gsm8k/some_other_layout/val_small.jsonl'
    derived = derive_raw_phase1_path(view, 'gsm8k')
    assert '/root/workspace' not in str(derived), derived
    assert derived == REPO_ROOT / 'data/gsm8k/phase1/val_small.jsonl', derived


def test_derive_raw_path_fallback_without_data_anchor_uses_repo_root():
    view = Path('/srv/elsewhere/val_small.jsonl')
    derived = derive_raw_phase1_path(view, 'regexeval')
    assert '/root/workspace' not in str(derived), derived
    assert derived == REPO_ROOT / 'data/regexeval/phase1/val_small.jsonl', derived


def test_derive_raw_path_ignores_data_dirs_above_the_checkout():
    view = Path('/mnt/data/projects/checkout/data/gsm8k/custom_layout/val_small.jsonl')
    derived = derive_raw_phase1_path(view, 'gsm8k')
    expected = Path('/mnt/data/projects/checkout/data/gsm8k/phase1/val_small.jsonl')
    assert derived == expected, derived


def test_nested_path_and_dir_fields_resolve_anywhere_in_payload():
    cfg_path = _write_config({
        **BASE_PAYLOAD,
        'data': {'train_path': 'data/gsm8k/phase1/train_small.jsonl'},
        'training': {'log_dir': 'runs/logs/test_run', 'max_steps': 10},
        'evaluation': {'report_path': 'runs/reports/test_run.json'},
    })
    cfg = load_run_config(cfg_path)
    assert Path(cfg.training['log_dir']) == REPO_ROOT / 'runs/logs/test_run'
    assert Path(cfg.evaluation['report_path']) == REPO_ROOT / 'runs/reports/test_run.json'
    assert cfg.training['max_steps'] == 10
    assert cfg.backbone_checkpoint == 'tiny-test', 'checkpoint ids must not be treated as paths'


def main() -> int:
    tests = [fn for name, fn in sorted(globals().items()) if name.startswith('test_') and callable(fn)]
    failures = 0
    for fn in tests:
        try:
            fn()
            print(f'PASS  {fn.__name__}')
        except AssertionError as exc:
            failures += 1
            print(f'FAIL  {fn.__name__}: {exc}')
        except Exception as exc:  # noqa: BLE001 - report unexpected errors as failures
            failures += 1
            print(f'ERROR {fn.__name__}: {type(exc).__name__}: {exc}')
    total = len(tests)
    print(f'{total - failures}/{total} passed')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
