#!/usr/bin/env python3
"""Unit tests for the eval-pipeline fixes (issue #2).

Runs with plain `python tests/test_eval_fixes.py` (also pytest-compatible).
Only torch is required: `peft` and `transformers` are stubbed so the pure
helpers in jepa_phase1 stay testable on machines without the full ML stack.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch

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

from jepa_phase1.evaluations import generate_predictions  # noqa: E402
from jepa_phase1.models import DecoupledJepaReasonerWrapper, pooled_last_hidden  # noqa: E402

CPU = torch.device('cpu')


# ---------------------------------------------------------------------------
# pooled_last_hidden: must select the last real token under either padding side
# ---------------------------------------------------------------------------

def _right_padded_case():
    batch, seq_len, hidden = 3, 5, 4
    states = torch.arange(batch * seq_len * hidden, dtype=torch.float32).reshape(batch, seq_len, hidden)
    lengths = [5, 3, 1]
    mask = torch.zeros(batch, seq_len, dtype=torch.long)
    for i, n in enumerate(lengths):
        mask[i, :n] = 1
    return states, mask, lengths


def test_pooled_last_hidden_right_padding():
    states, mask, lengths = _right_padded_case()
    pooled = pooled_last_hidden(states, mask)
    for i, n in enumerate(lengths):
        assert torch.equal(pooled[i], states[i, n - 1]), f'row {i}: expected hidden at position {n - 1}'


def test_pooled_last_hidden_matches_legacy_on_right_padding():
    states, mask, _ = _right_padded_case()
    legacy_idx = (mask.long().sum(dim=1) - 1).clamp(min=0)
    legacy = states[torch.arange(states.size(0)), legacy_idx]
    assert torch.equal(pooled_last_hidden(states, mask), legacy)


def test_pooled_last_hidden_left_padding():
    states, _, lengths = _right_padded_case()
    batch, seq_len, _ = states.shape
    left_states = torch.zeros_like(states)
    left_mask = torch.zeros(batch, seq_len, dtype=torch.long)
    for i, n in enumerate(lengths):
        left_mask[i, seq_len - n:] = 1
        left_states[i, seq_len - n:] = states[i, :n]
    pooled = pooled_last_hidden(left_states, left_mask)
    for i, n in enumerate(lengths):
        assert torch.equal(pooled[i], states[i, n - 1]), (
            f'row {i}: with left padding the last real token must be selected, '
            f'not a padding position'
        )


def test_pooled_last_hidden_all_pad_row_is_safe():
    states, mask, _ = _right_padded_case()
    mask[2, :] = 0
    pooled = pooled_last_hidden(states, mask)
    assert torch.equal(pooled[2], states[2, 0])


# ---------------------------------------------------------------------------
# generate_predictions: no prompt echo with left padding, chunked batching
# ---------------------------------------------------------------------------

class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False, truncation=True, max_length=None):
        ids = [ord(c) for c in text]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return ids

    def decode(self, ids, skip_special_tokens=True):
        if torch.is_tensor(ids):
            ids = ids.tolist()
        return ''.join(chr(i) for i in ids if i not in (self.pad_token_id, self.eos_token_id))


class FakeCausalModel:
    """Mimics the HF decoder-only contract: returns [input | generated]."""

    def __init__(self, completion: str):
        self.completion = completion
        self.batch_sizes: list[int] = []

    def generate(self, input_ids, attention_mask, max_new_tokens, pad_token_id, eos_token_id=None):
        self.batch_sizes.append(input_ids.size(0))
        new = torch.tensor(
            [[ord(c) for c in self.completion][:max_new_tokens]] * input_ids.size(0),
            dtype=input_ids.dtype,
        )
        return torch.cat([input_ids, new], dim=1)


class FakeDecoupledModel:
    """Mimics the decoupled contract: returns only the generated tokens."""

    def generate(self, input_ids, attention_mask, max_new_tokens, pad_token_id, eos_token_id=None):
        new = torch.tensor([[ord('o'), ord('k')]] * input_ids.size(0), dtype=input_ids.dtype)
        return new


MIXED_LENGTH_PROMPTS = ['hi', 'a noticeably longer prompt', 'mid-size one']


def test_generated_text_has_no_prompt_echo():
    model = FakeCausalModel(completion='42')
    texts = generate_predictions(
        model, FakeTokenizer(), MIXED_LENGTH_PROMPTS,
        max_input_tokens=64, max_new_tokens=8, device=CPU, kind='lm',
    )
    assert texts == ['42', '42', '42'], (
        f'predictions must contain only generated tokens, got {texts!r}'
    )


def test_generation_runs_in_chunks():
    model = FakeCausalModel(completion='42')
    texts = generate_predictions(
        model, FakeTokenizer(), MIXED_LENGTH_PROMPTS,
        max_input_tokens=64, max_new_tokens=8, device=CPU, kind='lm',
        batch_size=2,
    )
    assert texts == ['42', '42', '42']
    assert model.batch_sizes == [2, 1], f'expected chunks of 2 then 1, got {model.batch_sizes}'


def test_decoupled_decoding_keeps_full_sequence():
    texts = generate_predictions(
        FakeDecoupledModel(), FakeTokenizer(), MIXED_LENGTH_PROMPTS,
        max_input_tokens=64, max_new_tokens=8, device=CPU, kind='decoupled',
    )
    assert texts == ['ok', 'ok', 'ok']


# ---------------------------------------------------------------------------
# Decoupled talker start token: generation must reuse the training start token
# ---------------------------------------------------------------------------

def test_resolve_start_token_prefers_bos():
    fake = SimpleNamespace(backbone=SimpleNamespace(
        generation_config=SimpleNamespace(bos_token_id=7, eos_token_id=9, pad_token_id=3),
        config=None,
    ))
    assert DecoupledJepaReasonerWrapper.resolve_start_token_id(fake) == 7


def test_resolve_start_token_falls_back_to_eos_then_pad():
    no_bos = SimpleNamespace(backbone=SimpleNamespace(
        generation_config=None,
        config=SimpleNamespace(bos_token_id=None, eos_token_id=9, pad_token_id=3),
    ))
    assert DecoupledJepaReasonerWrapper.resolve_start_token_id(no_bos) == 9
    only_pad = SimpleNamespace(backbone=SimpleNamespace(
        generation_config=None,
        config=SimpleNamespace(bos_token_id=None, eos_token_id=None, pad_token_id=3),
    ))
    assert DecoupledJepaReasonerWrapper.resolve_start_token_id(only_pad) == 3


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
