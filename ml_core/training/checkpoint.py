"""Checkpoint save/load utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


@dataclass
class CheckpointPayload:
    state_dict: dict
    epoch: int
    val_metric: float
    config: dict[str, Any]


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    epoch: int,
    val_metric: float,
    config: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "epoch": int(epoch),
            "val_metric": float(val_metric),
            "config": dict(config or {}),
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    map_location: str | torch.device | None = None,
    strict: bool = True,
) -> CheckpointPayload:
    blob = torch.load(str(path), map_location=map_location)
    model.load_state_dict(blob["state_dict"], strict=strict)
    return CheckpointPayload(
        state_dict=blob["state_dict"],
        epoch=int(blob.get("epoch", -1)),
        val_metric=float(blob.get("val_metric", float("nan"))),
        config=dict(blob.get("config", {})),
    )
