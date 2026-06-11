#!/usr/bin/env python3
"""Unit tests for the decoupled talker modes (issue #6).

Runs with plain `python tests/test_latent_prefix_talker.py` (also
pytest-compatible). A tiny fake causal LM is injected through the
`transformers` stub so no checkpoint download is needed.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))

HIDDEN = 16
VOCAB = 50
PAD_ID, EOS_ID, BOS_ID = 0, 1, 2


class TinyCausalLM(nn.Module):
    """Minimal stand-in for a HF causal LM: embeddings, a causal cumulative-mean
    mixer (so earlier positions influence later logits, like attention does),
    a linear 'trunk', tied lm_head, hidden_states output, and a concat-based
    kv cache."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, HIDDEN)
        self.trunk = nn.Linear(HIDDEN, HIDDEN)
        ids = SimpleNamespace(bos_token_id=BOS_ID, eos_token_id=EOS_ID, pad_token_id=PAD_ID)
        self.config = ids
        self.generation_config = ids

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None,
                output_hidden_states=False, past_key_values=None, use_cache=False,
                labels=None, return_dict=True):
        if inputs_embeds is None:
            inputs_embeds = self.embed(input_ids)
        if past_key_values is not None:
            inputs_embeds = torch.cat([past_key_values, inputs_embeds], dim=1)
        positions = torch.arange(1, inputs_embeds.size(1) + 1, device=inputs_embeds.device)
        causal_mix = inputs_embeds.cumsum(dim=1) / positions.view(1, -1, 1)
        hidden = torch.tanh(self.trunk(causal_mix))
        logits = hidden @ self.embed.weight.T
        return SimpleNamespace(
            logits=logits,
            hidden_states=(hidden,) if output_hidden_states else None,
            past_key_values=inputs_embeds if use_cache else None,
            loss=None,
        )


def _install_stubs():
    peft = types.ModuleType('peft')
    peft.LoraConfig = object
    peft.get_peft_model = lambda model, cfg: model
    sys.modules['peft'] = peft

    transformers = types.ModuleType('transformers')

    class _AutoModel:
        @staticmethod
        def from_pretrained(checkpoint, **kwargs):
            return TinyCausalLM()

    transformers.AutoModelForCausalLM = _AutoModel
    transformers.AutoTokenizer = _AutoModel
    sys.modules['transformers'] = transformers


_install_stubs()

from jepa_phase1.models import DecoupledJepaReasonerWrapper  # noqa: E402


def _arch(mode: str) -> dict:
    return {
        'latent_rollout': 'short',
        'target_encoder_mode': 'shared_backbone_stopgrad',
        'talker_mode': mode,
        'latent_prefix_tokens': 4,
    }


def _batch(batch_size: int = 3, cond_len: int = 5, tgt_len: int = 6) -> dict:
    g = torch.Generator().manual_seed(0)
    cond = torch.randint(3, VOCAB, (batch_size, cond_len), generator=g)
    tgt = torch.randint(3, VOCAB, (batch_size, tgt_len), generator=g)
    tgt[:, -1] = EOS_ID
    labels = tgt.clone()
    return {
        'condition_input_ids': cond,
        'condition_attention_mask': torch.ones_like(cond),
        'target_input_ids': tgt,
        'target_attention_mask': torch.ones_like(tgt),
        'target_labels': labels,
    }


def _freeze_non_talker(model):
    for module in (model.reasoner, model.condition_proj, model.backbone):
        for p in module.parameters():
            p.requires_grad = False


def test_latent_prefix_mode_builds_projector_and_no_gru_talker():
    model = DecoupledJepaReasonerWrapper('tiny-test', None, _arch('latent_prefix'), stage='stage2')
    assert hasattr(model, 'prefix_projector')
    assert getattr(model, 'talker', None) is None, 'GRU talker must not be instantiated in latent_prefix mode'
    prefix = model._build_prefix(torch.zeros(2, HIDDEN))
    assert prefix.shape == (2, 4, HIDDEN), prefix.shape


def test_gru_mode_keeps_existing_modules():
    model = DecoupledJepaReasonerWrapper('tiny-test', None, _arch('gru'), stage='stage2')
    assert getattr(model, 'prefix_projector', None) is None
    assert model.talker is not None and model.latent_prefix is not None


def test_stage2_latent_prefix_trains_projector_only():
    model = DecoupledJepaReasonerWrapper('tiny-test', None, _arch('latent_prefix'), stage='stage2')
    _freeze_non_talker(model)
    out = model(_batch())
    assert torch.isfinite(out.loss), out.loss
    out.loss.backward()
    assert model.prefix_projector.weight.grad is not None
    assert model.prefix_projector.weight.grad.abs().sum() > 0
    for name, param in model.backbone.named_parameters():
        assert param.grad is None, f'backbone param {name} received gradient in stage 2'


def test_stage2_gru_regression_still_works():
    model = DecoupledJepaReasonerWrapper('tiny-test', None, _arch('gru'), stage='stage2')
    _freeze_non_talker(model)
    out = model(_batch())
    assert torch.isfinite(out.loss), out.loss
    out.loss.backward()
    grads = [p.grad for p in model.talker.parameters()]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_stage3_latent_prefix_joint_loss_reaches_reasoner():
    model = DecoupledJepaReasonerWrapper('tiny-test', None, _arch('latent_prefix'), stage='stage3')
    out = model(_batch())
    assert torch.isfinite(out.loss)
    assert 'talker_loss' in out.metrics and 'reasoner_loss' in out.metrics
    out.loss.backward()
    reasoner_grads = [p.grad for p in model.reasoner.parameters()]
    assert any(g is not None and g.abs().sum() > 0 for g in reasoner_grads)
    assert model.prefix_projector.weight.grad is not None
    assert model.prefix_projector.weight.grad.abs().sum() > 0


def test_generate_latent_prefix_shapes_and_vocab():
    model = DecoupledJepaReasonerWrapper('tiny-test', None, _arch('latent_prefix'), stage='stage2')
    batch = _batch()
    tokens = model.generate(
        batch['condition_input_ids'], batch['condition_attention_mask'],
        max_new_tokens=5, pad_token_id=PAD_ID, eos_token_id=EOS_ID,
    )
    assert tokens.dim() == 2 and tokens.size(0) == 3
    assert 1 <= tokens.size(1) <= 5, tokens.shape
    assert int(tokens.min()) >= 0 and int(tokens.max()) < VOCAB


def test_generate_gru_regression():
    model = DecoupledJepaReasonerWrapper('tiny-test', None, _arch('gru'), stage='stage2')
    batch = _batch()
    tokens = model.generate(
        batch['condition_input_ids'], batch['condition_attention_mask'],
        max_new_tokens=5, pad_token_id=PAD_ID, eos_token_id=EOS_ID,
    )
    assert tokens.dim() == 2 and tokens.size(0) == 3 and tokens.size(1) <= 5


def test_talker_modules_helper_reports_mode_specific_modules():
    prefix_model = DecoupledJepaReasonerWrapper('tiny-test', None, _arch('latent_prefix'), stage='stage2')
    assert prefix_model.talker_modules() == [prefix_model.prefix_projector]
    gru_model = DecoupledJepaReasonerWrapper('tiny-test', None, _arch('gru'), stage='stage2')
    assert gru_model.talker_modules() == [gru_model.latent_prefix, gru_model.talker]


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
