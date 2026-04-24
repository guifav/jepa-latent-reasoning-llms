from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

IGNORE_INDEX = -100


@dataclass
class ExampleBatch:
    tensors: dict[str, torch.Tensor]


class JsonlDataset(Dataset):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.rows = []
        with self.path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def pad_to_length(seqs: list[list[int]], pad_id: int, side: str = 'right') -> torch.Tensor:
    max_len = max(len(s) for s in seqs)
    out = []
    for s in seqs:
        pad = [pad_id] * (max_len - len(s))
        out.append(s + pad if side == 'right' else pad + s)
    return torch.tensor(out, dtype=torch.long)


def mask_prompt_labels(input_ids: list[int], prompt_len: int) -> list[int]:
    labels = input_ids[:]
    for i in range(prompt_len):
        labels[i] = IGNORE_INDEX
    return labels


class LMCollator:
    def __init__(self, tokenizer, max_input_tokens: int, max_target_tokens: int):
        self.tokenizer = tokenizer
        self.max_input_tokens = max_input_tokens
        self.max_target_tokens = max_target_tokens
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        sequences = []
        labels = []
        for row in rows:
            prompt_ids = self.tokenizer.encode(row['input_text'], add_special_tokens=False, truncation=True, max_length=self.max_input_tokens)
            target_ids = self.tokenizer.encode(row['target_text'], add_special_tokens=False, truncation=True, max_length=self.max_target_tokens)
            input_ids = prompt_ids + target_ids + [self.tokenizer.eos_token_id]
            sequences.append(input_ids)
            labels.append(mask_prompt_labels(input_ids, len(prompt_ids)))
        input_ids = pad_to_length(sequences, self.tokenizer.pad_token_id)
        label_ids = pad_to_length(labels, IGNORE_INDEX)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': label_ids}


class CoupledCollator:
    def __init__(self, tokenizer, max_packed_input_tokens: int, max_generation_target_tokens: int):
        self.tokenizer = tokenizer
        self.max_packed_input_tokens = max_packed_input_tokens
        self.max_generation_target_tokens = max_generation_target_tokens
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _encode(self, text: str, max_len: int) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=max_len)

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        view_a = [self._encode(r['view_a_text'], self.max_packed_input_tokens // 2) for r in rows]
        view_b = [self._encode(r['view_b_text'], self.max_packed_input_tokens // 2) for r in rows]
        gen_prompt = [self._encode(r['generation_prompt_text'], self.max_packed_input_tokens) for r in rows]
        gen_target = [self._encode(r['generation_target_text'], self.max_generation_target_tokens) + [self.tokenizer.eos_token_id] for r in rows]

        gen_full = [p + t for p, t in zip(gen_prompt, gen_target)]
        gen_labels = [mask_prompt_labels(seq, len(prompt)) for seq, prompt in zip(gen_full, gen_prompt)]

        batch = {
            'view_a_input_ids': pad_to_length(view_a, self.tokenizer.pad_token_id),
            'view_b_input_ids': pad_to_length(view_b, self.tokenizer.pad_token_id),
            'generation_input_ids': pad_to_length(gen_full, self.tokenizer.pad_token_id),
            'generation_labels': pad_to_length(gen_labels, IGNORE_INDEX),
        }
        batch['view_a_attention_mask'] = (batch['view_a_input_ids'] != self.tokenizer.pad_token_id).long()
        batch['view_b_attention_mask'] = (batch['view_b_input_ids'] != self.tokenizer.pad_token_id).long()
        batch['generation_attention_mask'] = (batch['generation_input_ids'] != self.tokenizer.pad_token_id).long()
        return batch


class DecoupledCollator:
    def __init__(self, tokenizer, max_condition_tokens: int, max_talker_target_tokens: int):
        self.tokenizer = tokenizer
        self.max_condition_tokens = max_condition_tokens
        self.max_talker_target_tokens = max_talker_target_tokens
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _encode(self, text: str, max_len: int) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=max_len)

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        condition = [self._encode(r['condition_text'], self.max_condition_tokens) for r in rows]
        if 'talker_joint_target' in rows[0]:
            target_text_key = 'talker_joint_target'
        elif 'talker_target' in rows[0]:
            target_text_key = 'talker_target'
        else:
            target_text_key = 'talker_answer_target'
        targets = [self._encode(r[target_text_key], self.max_talker_target_tokens) + [self.tokenizer.eos_token_id] for r in rows]

        batch = {
            'condition_input_ids': pad_to_length(condition, self.tokenizer.pad_token_id),
            'target_input_ids': pad_to_length(targets, self.tokenizer.pad_token_id),
        }
        batch['condition_attention_mask'] = (batch['condition_input_ids'] != self.tokenizer.pad_token_id).long()
        batch['target_attention_mask'] = (batch['target_input_ids'] != self.tokenizer.pad_token_id).long()
        batch['target_labels'] = batch['target_input_ids'].clone()
        batch['target_labels'][batch['target_labels'] == self.tokenizer.pad_token_id] = IGNORE_INDEX
        return batch
