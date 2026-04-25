from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM


@dataclass
class ModelOutput:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]


def pooled_last_hidden(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    lengths = attention_mask.long().sum(dim=1) - 1
    lengths = lengths.clamp(min=0)
    batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
    return hidden_states[batch_idx, lengths]


def latent_geometry_metrics(latents: torch.Tensor, prefix: str) -> dict[str, torch.Tensor]:
    """Cheap batch-level diagnostics for latent collapse/geometry.

    Uses the sample Gram matrix instead of a hidden_size x hidden_size covariance
    matrix, so it remains inexpensive for Gemma-sized hidden states. These are
    diagnostics, not training losses.
    """
    z = latents.detach().float()
    norm = z.norm(dim=-1)
    metrics: dict[str, torch.Tensor] = {
        f'{prefix}_norm_mean': norm.mean(),
        f'{prefix}_norm_std': norm.std(unbiased=False) if z.size(0) > 1 else torch.zeros((), device=z.device),
    }
    if z.size(0) < 2:
        zero = torch.zeros((), device=z.device)
        metrics[f'{prefix}_effective_rank'] = zero
        metrics[f'{prefix}_isotropy_deviation'] = zero
        return metrics

    z = z - z.mean(dim=0, keepdim=True)
    gram = z @ z.T / max(1, z.size(1))
    eigvals = torch.linalg.eigvalsh(gram).clamp_min(0)
    eig_sum = eigvals.sum().clamp_min(1e-12)
    probs = eigvals / eig_sum
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum()
    effective_rank = entropy.exp()
    uniform = torch.full_like(probs, 1.0 / probs.numel())
    isotropy_deviation = torch.mean(torch.abs(probs - uniform))
    metrics[f'{prefix}_effective_rank'] = effective_rank
    metrics[f'{prefix}_isotropy_deviation'] = isotropy_deviation
    return metrics


def resolve_lora_target_modules(model: nn.Module, requested_targets: list[str]) -> list[str]:
    resolved = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if 'vision_tower' in name or 'audio_tower' in name:
            continue
        if any(name.endswith(target) for target in requested_targets):
            resolved.append(name)
    return resolved or requested_targets


def maybe_apply_lora(model: nn.Module, adaptation_cfg: dict[str, Any] | None):
    if not adaptation_cfg or adaptation_cfg.get('type') != 'lora':
        return model
    target_modules = resolve_lora_target_modules(model, adaptation_cfg['target_modules'])
    lora_cfg = LoraConfig(
        r=adaptation_cfg['rank'],
        lora_alpha=adaptation_cfg['alpha'],
        lora_dropout=adaptation_cfg.get('dropout', 0.0),
        target_modules=target_modules,
        bias='none',
        task_type='CAUSAL_LM',
    )
    return get_peft_model(model, lora_cfg)


class BaselineLMWrapper(nn.Module):
    def __init__(self, checkpoint: str, adaptation_cfg: dict[str, Any] | None = None):
        super().__init__()
        self.backbone = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype='auto', low_cpu_mem_usage=True)
        self.backbone = maybe_apply_lora(self.backbone, adaptation_cfg)

    def forward(self, batch: dict[str, torch.Tensor]) -> ModelOutput:
        out = self.backbone(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
            return_dict=True,
        )
        return ModelOutput(loss=out.loss, metrics={'ce_loss': out.loss.detach()})

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, max_new_tokens: int, pad_token_id: int, eos_token_id: int | None = None) -> torch.Tensor:
        return self.backbone.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )


class CoupledLLMJepaWrapper(nn.Module):
    def __init__(self, checkpoint: str, adaptation_cfg: dict[str, Any] | None, jepa_cfg: dict[str, Any]):
        super().__init__()
        self.backbone = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype='auto', low_cpu_mem_usage=True)
        self.backbone = maybe_apply_lora(self.backbone, adaptation_cfg)
        hidden_size = self.backbone.get_input_embeddings().embedding_dim
        depth = int(jepa_cfg.get('predictor_depth', 1))
        layers = []
        for _ in range(max(1, depth)):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.GELU()])
        layers.append(nn.Linear(hidden_size, hidden_size))
        self.predictor = nn.Sequential(*layers)
        model_dtype = self.backbone.get_input_embeddings().weight.dtype
        self.predictor = self.predictor.to(dtype=model_dtype)
        self.lambda_jepa = float(jepa_cfg.get('lambda', 0.1))
        self.loss_type = str(jepa_cfg.get('loss_type', 'cosine_plus_infonce'))
        self.temperature = float(jepa_cfg.get('contrastive_temperature', 0.07))
        self.cosine_weight = float(jepa_cfg.get('cosine_weight', 1.0))
        self.contrastive_weight = float(jepa_cfg.get('contrastive_weight', 1.0))

    def encode_view(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        return pooled_last_hidden(out.hidden_states[-1], attention_mask)

    def forward(self, batch: dict[str, torch.Tensor]) -> ModelOutput:
        view_a = self.encode_view(batch['view_a_input_ids'], batch['view_a_attention_mask'])
        with torch.no_grad():
            view_b = self.encode_view(batch['view_b_input_ids'], batch['view_b_attention_mask'])
        pred_b = self.predictor(view_a)
        cosine_loss = (1.0 - F.cosine_similarity(pred_b, view_b, dim=-1)).mean()
        pred_norm = F.normalize(pred_b.float(), dim=-1)
        target_norm = F.normalize(view_b.float(), dim=-1)
        logits = pred_norm @ target_norm.T / max(self.temperature, 1e-6)
        labels = torch.arange(logits.size(0), device=logits.device)
        contrastive_loss = F.cross_entropy(logits, labels)
        if self.loss_type in {'contrastive', 'infonce'}:
            jepa_loss = contrastive_loss
        elif self.loss_type in {'cosine_plus_infonce', 'cosine_plus_contrastive'}:
            jepa_loss = self.cosine_weight * cosine_loss + self.contrastive_weight * contrastive_loss
        else:
            jepa_loss = cosine_loss

        gen_out = self.backbone(
            input_ids=batch['generation_input_ids'],
            attention_mask=batch['generation_attention_mask'],
            labels=batch['generation_labels'],
            return_dict=True,
        )
        total_loss = gen_out.loss + self.lambda_jepa * jepa_loss
        return ModelOutput(
            loss=total_loss,
            metrics={
                'ce_loss': gen_out.loss.detach(),
                'jepa_loss': jepa_loss.detach(),
                'cosine_loss': cosine_loss.detach(),
                'contrastive_loss': contrastive_loss.detach(),
                'contrastive_pos_logit': torch.diag(logits).mean().detach(),
                'contrastive_max_neg_logit': logits.masked_fill(torch.eye(logits.size(0), device=logits.device, dtype=torch.bool), float('-inf')).max(dim=1).values.mean().detach() if logits.size(0) > 1 else torch.zeros((), device=logits.device),
                'total_loss': total_loss.detach(),
            },
        )

    @torch.no_grad()
    def latent_diagnostics(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        view_a = self.encode_view(batch['view_a_input_ids'], batch['view_a_attention_mask'])
        view_b = self.encode_view(batch['view_b_input_ids'], batch['view_b_attention_mask'])
        pred_b = self.predictor(view_a)
        metrics = latent_geometry_metrics(view_a, 'view_a')
        metrics.update(latent_geometry_metrics(view_b, 'view_b'))
        metrics.update(latent_geometry_metrics(pred_b, 'pred_b'))
        metrics['alignment_cosine_mean'] = F.cosine_similarity(pred_b, view_b, dim=-1).float().mean()
        if pred_b.size(0) > 1:
            logits = F.normalize(pred_b.float(), dim=-1) @ F.normalize(view_b.float(), dim=-1).T
            eye = torch.eye(logits.size(0), device=logits.device, dtype=torch.bool)
            metrics['alignment_diag_minus_max_negative'] = (torch.diag(logits) - logits.masked_fill(eye, float('-inf')).max(dim=1).values).mean()
        return metrics

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, max_new_tokens: int, pad_token_id: int, eos_token_id: int | None = None) -> torch.Tensor:
        return self.backbone.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )


class SmallLatentReasoner(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.gru = nn.GRUCell(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)

    def forward(self, init_state: torch.Tensor, steps: int) -> tuple[torch.Tensor, torch.Tensor]:
        h = init_state
        states = []
        x = torch.zeros_like(init_state)
        for _ in range(steps):
            h = self.gru(x, h)
            states.append(h)
            x = h
        stacked = torch.stack(states, dim=1)
        return stacked, self.out(stacked[:, -1])


class SmallTalker(nn.Module):
    """Tiny autoregressive decoder over backbone token embeddings.

    The decoupled model does not call the causal-LM backbone to decode tokens.
    The backbone supplies the shared embedding matrix and vocabulary projection;
    the GRU talker autoregressively maps previous token embeddings plus the
    latent reasoner state to next-token logits.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, token_embeds: torch.Tensor, init_state: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.gru(token_embeds, init_state.unsqueeze(0))
        return outputs


class DecoupledJepaReasonerWrapper(nn.Module):
    def __init__(self, checkpoint: str, adaptation_cfg: dict[str, Any] | None, arch_cfg: dict[str, Any], stage: str = 'stage1'):
        super().__init__()
        self.stage = stage
        self.backbone = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype='auto', low_cpu_mem_usage=True)
        self.backbone = maybe_apply_lora(self.backbone, adaptation_cfg)
        hidden_size = self.backbone.get_input_embeddings().embedding_dim
        self.reasoner = SmallLatentReasoner(hidden_size)
        self.condition_proj = nn.Linear(hidden_size, hidden_size)
        self.latent_prefix = nn.Linear(hidden_size, hidden_size)
        self.talker = SmallTalker(hidden_size)
        model_dtype = self.backbone.get_input_embeddings().weight.dtype
        self.reasoner = self.reasoner.to(dtype=model_dtype)
        self.condition_proj = self.condition_proj.to(dtype=model_dtype)
        self.latent_prefix = self.latent_prefix.to(dtype=model_dtype)
        self.talker = self.talker.to(dtype=model_dtype)
        self.rollout_steps = 4 if arch_cfg.get('latent_rollout', 'short') == 'short' else 8
        self.target_encoder_mode = arch_cfg.get('target_encoder_mode', 'shared_backbone_stopgrad')
        self.momentum = float(arch_cfg.get('target_encoder_momentum', 0.98))
        self.target_encoder = None
        if self.target_encoder_mode == 'ema_copy':
            import copy
            self.target_encoder = copy.deepcopy(self.backbone)
            for p in self.target_encoder.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def update_target_encoder(self):
        if self.target_encoder is None:
            return
        for p_t, p_o in zip(self.target_encoder.parameters(), self.backbone.parameters()):
            p_t.data.mul_(self.momentum).add_(p_o.data, alpha=1.0 - self.momentum)

    def encode_with_model(self, model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor, grad_enabled: bool = True) -> torch.Tensor:
        with torch.set_grad_enabled(grad_enabled):
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        return pooled_last_hidden(out.hidden_states[-1], attention_mask)

    def encode_target_latent(self, target_input_ids: torch.Tensor, target_attention_mask: torch.Tensor) -> torch.Tensor:
        target_model = self.target_encoder if self.target_encoder is not None else self.backbone
        target_latent = self.encode_with_model(target_model, target_input_ids, target_attention_mask, grad_enabled=False)
        return target_latent.detach()

    def forward_stage1(self, batch: dict[str, torch.Tensor]) -> ModelOutput:
        cond = self.encode_with_model(self.backbone, batch['condition_input_ids'], batch['condition_attention_mask'], grad_enabled=True)
        init_state = self.condition_proj(cond)
        _, pred_latent = self.reasoner(init_state, self.rollout_steps)
        target_latent = self.encode_target_latent(batch['target_input_ids'], batch['target_attention_mask'])
        latent_loss = (1.0 - F.cosine_similarity(pred_latent, target_latent, dim=-1)).mean()
        metrics = {'reasoner_loss': latent_loss.detach()}
        if self.target_encoder is None:
            metrics['target_encoder_mode'] = torch.tensor(0.0, device=latent_loss.device)
        return ModelOutput(loss=latent_loss, metrics=metrics)

    def resolve_start_token_id(self) -> int:
        generation_config = getattr(self.backbone, 'generation_config', None)
        for source in (generation_config, self.backbone.config):
            if source is None:
                continue
            for attr in ('bos_token_id', 'eos_token_id', 'pad_token_id', 'eoa_token_id'):
                value = getattr(source, attr, None)
                if value is not None:
                    return int(value)
        return 0

    def forward_stage2(self, batch: dict[str, torch.Tensor]) -> ModelOutput:
        with torch.no_grad():
            cond = self.encode_with_model(self.backbone, batch['condition_input_ids'], batch['condition_attention_mask'], grad_enabled=False)
            init_state = self.condition_proj(cond)
            _, pred_latent = self.reasoner(init_state, self.rollout_steps)
            embed_layer = self.backbone.get_input_embeddings()
            start_token_id = self.resolve_start_token_id()
            decoder_input_ids = batch['target_input_ids'].clone()
            decoder_input_ids[:, 1:] = batch['target_input_ids'][:, :-1]
            decoder_input_ids[:, 0] = start_token_id
            token_embeds = embed_layer(decoder_input_ids)
            vocab_weight = embed_layer.weight

        talker_init = self.latent_prefix(pred_latent)
        talker_hidden = self.talker(token_embeds, talker_init)
        logits = F.linear(talker_hidden, vocab_weight)
        loss = F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            batch['target_labels'].reshape(-1),
            ignore_index=-100,
        )
        return ModelOutput(loss=loss, metrics={'talker_loss': loss.detach()})

    def forward_stage3(self, batch: dict[str, torch.Tensor]) -> ModelOutput:
        cond = self.encode_with_model(self.backbone, batch['condition_input_ids'], batch['condition_attention_mask'], grad_enabled=True)
        init_state = self.condition_proj(cond)
        _, pred_latent = self.reasoner(init_state, self.rollout_steps)
        target_latent = self.encode_target_latent(batch['target_input_ids'], batch['target_attention_mask'])
        latent_loss = (1.0 - F.cosine_similarity(pred_latent, target_latent, dim=-1)).mean()

        embed_layer = self.backbone.get_input_embeddings()
        start_token_id = self.resolve_start_token_id()
        decoder_input_ids = batch['target_input_ids'].clone()
        decoder_input_ids[:, 1:] = batch['target_input_ids'][:, :-1]
        decoder_input_ids[:, 0] = start_token_id
        token_embeds = embed_layer(decoder_input_ids)
        talker_init = self.latent_prefix(pred_latent)
        talker_hidden = self.talker(token_embeds, talker_init)
        logits = F.linear(talker_hidden, embed_layer.weight)
        talker_loss = F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            batch['target_labels'].reshape(-1),
            ignore_index=-100,
        )
        lambda_latent = 0.1
        total_loss = talker_loss + lambda_latent * latent_loss
        return ModelOutput(
            loss=total_loss,
            metrics={
                'joint_loss': total_loss.detach(),
                'talker_loss': talker_loss.detach(),
                'reasoner_loss': latent_loss.detach(),
            },
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> ModelOutput:
        if self.stage == 'stage1':
            return self.forward_stage1(batch)
        if self.stage == 'stage3':
            return self.forward_stage3(batch)
        return self.forward_stage2(batch)

    @torch.no_grad()
    def latent_diagnostics(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        cond = self.encode_with_model(self.backbone, batch['condition_input_ids'], batch['condition_attention_mask'], grad_enabled=False)
        init_state = self.condition_proj(cond)
        rollout, pred_latent = self.reasoner(init_state, self.rollout_steps)
        target_latent = self.encode_target_latent(batch['target_input_ids'], batch['target_attention_mask'])
        metrics = latent_geometry_metrics(cond, 'condition_latent')
        metrics.update(latent_geometry_metrics(pred_latent, 'pred_latent'))
        metrics.update(latent_geometry_metrics(target_latent, 'target_latent'))
        metrics.update(latent_geometry_metrics(rollout.reshape(-1, rollout.size(-1)), 'rollout_latent'))
        metrics['alignment_cosine_mean'] = F.cosine_similarity(pred_latent, target_latent, dim=-1).float().mean()
        return metrics

    @torch.no_grad()
    def generate(self, condition_input_ids: torch.Tensor, condition_attention_mask: torch.Tensor, max_new_tokens: int, pad_token_id: int, eos_token_id: int | None = None) -> torch.Tensor:
        cond = self.encode_with_model(self.backbone, condition_input_ids, condition_attention_mask, grad_enabled=False)
        init_state = self.condition_proj(cond)
        _, pred_latent = self.reasoner(init_state, self.rollout_steps)
        hidden = self.latent_prefix(pred_latent).unsqueeze(0)
        embed_layer = self.backbone.get_input_embeddings()
        vocab_weight = embed_layer.weight
        start_id = eos_token_id if eos_token_id is not None else pad_token_id
        prev_tokens = torch.full((condition_input_ids.size(0), 1), start_id, dtype=condition_input_ids.dtype, device=condition_input_ids.device)
        generated = []
        ended = torch.zeros(condition_input_ids.size(0), dtype=torch.bool, device=condition_input_ids.device)
        rnn_state = hidden
        for _ in range(max_new_tokens):
            step_embed = embed_layer(prev_tokens)
            step_out, rnn_state = self.talker.gru(step_embed, rnn_state)
            logits = F.linear(step_out[:, -1, :], vocab_weight)
            next_tokens = logits.argmax(dim=-1, keepdim=True)
            generated.append(next_tokens)
            prev_tokens = next_tokens
            if eos_token_id is not None:
                ended = ended | (next_tokens.squeeze(1) == eos_token_id)
                if bool(ended.all()):
                    break
        if not generated:
            return prev_tokens
        return torch.cat(generated, dim=1)
