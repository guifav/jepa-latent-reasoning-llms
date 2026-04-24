from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .configs import RunConfig
from .data import CoupledCollator, DecoupledCollator, JsonlDataset, LMCollator
from .evaluations import evaluate_benchmark
from .models import BaselineLMWrapper, CoupledLLMJepaWrapper, DecoupledJepaReasonerWrapper


def detect_run_kind(cfg: RunConfig) -> str:
    payload = cfg.payload
    if 'architecture' in payload:
        return 'decoupled'
    if 'jepa' in payload:
        return 'coupled'
    return 'lm'


def build_tokenizer(checkpoint: str):
    tok = AutoTokenizer.from_pretrained(checkpoint)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def build_dataloaders(cfg: RunConfig, tokenizer):
    kind = detect_run_kind(cfg)
    train_ds = JsonlDataset(cfg.data['train_path'])
    val_ds = JsonlDataset(cfg.data['val_path'])
    if kind == 'lm':
        collator = LMCollator(tokenizer, cfg.training['max_input_tokens'], cfg.training['max_target_tokens'])
    elif kind == 'coupled':
        collator = CoupledCollator(tokenizer, cfg.training['max_packed_input_tokens'], cfg.training['max_generation_target_tokens'])
    else:
        collator = DecoupledCollator(tokenizer, cfg.training['max_condition_tokens'], cfg.training['max_talker_target_tokens'])
    train_loader = DataLoader(train_ds, batch_size=cfg.training['per_device_train_batch_size'], shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=cfg.training['per_device_eval_batch_size'], shuffle=False, collate_fn=collator)
    return train_loader, val_loader


def build_model(cfg: RunConfig, stage: str | None = None):
    kind = detect_run_kind(cfg)
    adaptation = cfg.payload.get('adaptation')
    if kind == 'lm':
        return BaselineLMWrapper(cfg.backbone_checkpoint, adaptation)
    if kind == 'coupled':
        return CoupledLLMJepaWrapper(cfg.backbone_checkpoint, adaptation, cfg.payload['jepa'])
    backbone_adapt = cfg.payload.get('adaptation', {}).get('backbone', cfg.payload.get('adaptation'))
    return DecoupledJepaReasonerWrapper(cfg.backbone_checkpoint, backbone_adapt, cfg.payload['architecture'], stage=stage or 'stage1')


def move_batch(batch: dict[str, torch.Tensor], device: torch.device):
    return {k: v.to(device) for k, v in batch.items()}


def optimizer_for(model: torch.nn.Module, cfg: RunConfig, lr_key: str = 'learning_rate'):
    lr = cfg.training.get(lr_key) or cfg.training.get('learning_rate_backbone') or 1e-4
    wd = cfg.training.get('weight_decay', 0.01)
    params = [p for p in model.parameters() if p.requires_grad]
    return AdamW(params, lr=lr, weight_decay=wd)


def evaluate_loss(model, loader, device):
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            out = model(batch)
            total += out.loss.item()
            count += 1
    return total / max(1, count)


def train_one_stage(model, cfg: RunConfig, train_loader, val_loader, device, max_steps: int, lr_key: str = 'learning_rate'):
    model.to(device)
    model.train()
    optimizer = optimizer_for(model, cfg, lr_key=lr_key)
    logging_steps = int(cfg.training.get('logging_steps', 25))
    eval_every = int(cfg.training.get('eval_every_steps', 100))
    best_val = math.inf
    best_state = None
    step = 0
    history = []
    grad_acc = int(cfg.training.get('gradient_accumulation_steps', 1))

    optimizer.zero_grad(set_to_none=True)
    while step < max_steps:
        for batch in train_loader:
            batch = move_batch(batch, device)
            out = model(batch)
            loss = out.loss / grad_acc
            loss.backward()
            if hasattr(model, 'update_target_encoder') and model.training:
                try:
                    model.update_target_encoder()
                except Exception:
                    pass
            if (step + 1) % grad_acc == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if step % logging_steps == 0:
                history.append({'step': step, 'train_loss': float(out.loss.detach().cpu())})
            if step > 0 and step % eval_every == 0:
                val_loss = evaluate_loss(model, val_loader, device)
                history.append({'step': step, 'val_loss': float(val_loss)})
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            step += 1
            if step >= max_steps:
                break
        else:
            continue
        break
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    return history, best_val


def run_training(cfg: RunConfig, output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = build_tokenizer(cfg.backbone_checkpoint)
    train_loader, val_loader = build_dataloaders(cfg, tokenizer)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kind = detect_run_kind(cfg)

    summary: dict[str, Any] = {
        'run_name': cfg.run_name,
        'kind': kind,
        'device': str(device),
        'stages': [],
    }

    if kind in {'lm', 'coupled'}:
        model = build_model(cfg)
        history, best_val = train_one_stage(model, cfg, train_loader, val_loader, device, int(cfg.training['max_steps']))
        torch.save(model.state_dict(), output_dir / 'model.pt')
        summary['stages'].append({'name': 'main', 'best_val_loss': best_val, 'history_tail': history[-10:]})
        summary['benchmark_eval'] = evaluate_benchmark(model, tokenizer, cfg, kind, device)
    else:
        model = build_model(cfg, stage='stage1')
        history1, best_val1 = train_one_stage(model, cfg, train_loader, val_loader, device, int(cfg.training['stage_1_max_steps']), lr_key='learning_rate_backbone')
        torch.save(model.state_dict(), output_dir / 'stage1_reasoner.pt')
        summary['stages'].append({'name': 'stage1_reasoner', 'best_val_loss': best_val1, 'history_tail': history1[-10:]})

        for module in (model.reasoner, model.condition_proj, model.backbone):
            for p in module.parameters():
                p.requires_grad = False
        model.stage = 'stage2'
        history2, best_val2 = train_one_stage(model, cfg, train_loader, val_loader, device, int(cfg.training['stage_2_max_steps']), lr_key='learning_rate_new_modules')
        torch.save(model.state_dict(), output_dir / 'stage2_talker.pt')
        summary['stages'].append({'name': 'stage2_talker', 'best_val_loss': best_val2, 'history_tail': history2[-10:]})
        summary['benchmark_eval'] = evaluate_benchmark(model, tokenizer, cfg, kind, device)

    with (output_dir / 'summary.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary
