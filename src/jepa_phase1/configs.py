from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
    return RunConfig(path=path, payload=payload)
