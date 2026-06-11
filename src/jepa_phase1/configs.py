from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

PATH_KEY_SUFFIXES = ('_path', '_dir')


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _resolve_path_fields(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            key: str(resolve_repo_path(value))
            if isinstance(key, str) and key.endswith(PATH_KEY_SUFFIXES) and isinstance(value, str) and value
            else _resolve_path_fields(value)
            for key, value in obj.items()
        }
    if isinstance(obj, list):
        return [_resolve_path_fields(item) for item in obj]
    return obj


@dataclass
class RunConfig:
    path: Path
    payload: dict[str, Any]

    @property
    def run_name(self) -> str:
        return self.payload['run_name']

    @property
    def benchmark(self) -> str:
        return self.payload['benchmark']

    @property
    def backbone_checkpoint(self) -> str:
        return self.payload['backbone_checkpoint']

    @property
    def data(self) -> dict[str, str]:
        return self.payload['data']

    @property
    def training(self) -> dict[str, Any]:
        return self.payload['training']

    @property
    def evaluation(self) -> dict[str, Any]:
        return self.payload['evaluation']


def load_run_config(path: str | Path) -> RunConfig:
    path = Path(path)
    with path.open('r', encoding='utf-8') as f:
        payload = json.load(f)
    payload = _resolve_path_fields(payload)
    return RunConfig(path=path, payload=payload)
