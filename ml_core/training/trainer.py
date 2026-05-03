"""Generic Trainer: AdamW + CE + early stop on val macro F1.

Device-agnostic: prefers CUDA (Colab T4), then MPS (M-series Macs), else CPU.
Same script runs on M4 dev box and Colab GPU without changes.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from ..evaluation.metrics import compute_classification_metrics
from .callbacks import NoOpCallback
from .checkpoint import save_checkpoint


def pick_device(prefer: str | None = None) -> torch.device:
    """Pick best available device. Override via ``prefer in {"cuda","mps","cpu"}``."""
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_global_seed(seed: int) -> None:
    """Seed torch, numpy, python random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainConfig:
    n_epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    early_stop_patience: int = 15
    seed: int = 42
    grad_clip_norm: float | None = 1.0
    use_class_weights: bool = True
    cosine_lr: bool = True
    device: str | None = None  # None -> auto pick
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _class_weights(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    total  = counts.sum()
    weights = total / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


class Trainer:
    """Minimal training loop with early stopping on validation macro F1."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainConfig,
        *,
        callback=None,
        ckpt_path: str | Path | None = None,
        n_classes: int = 3,
        train_label_array: np.ndarray | None = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.callback = callback or NoOpCallback()
        self.ckpt_path = Path(ckpt_path) if ckpt_path else None
        self.n_classes = n_classes
        self.device = pick_device(config.device)
        self.model.to(self.device)

        if config.use_class_weights and train_label_array is not None:
            self.class_weights = _class_weights(train_label_array, n_classes).to(self.device)
        else:
            self.class_weights = None

        self.optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.scheduler = (
            CosineAnnealingLR(self.optimizer, T_max=config.n_epochs)
            if config.cosine_lr
            else None
        )

        self.best_val: float = -math.inf
        self.best_epoch: int = -1
        self._patience: int = 0

    # ------------------------------------------------------------------ steps
    def _train_one_epoch(self) -> float:
        self.model.train(True)
        total_loss, n = 0.0, 0
        for batch in self.train_loader:
            x, y = batch[0].to(self.device), batch[1].to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(x)
            loss = F.cross_entropy(logits, y, weight=self.class_weights)
            loss.backward()
            if self.config.grad_clip_norm:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
            self.optimizer.step()
            total_loss += float(loss.item()) * x.size(0)
            n += x.size(0)
        return total_loss / max(n, 1)

    @torch.no_grad()
    def _run_validation(self, loader: DataLoader) -> dict[str, float]:
        self.model.train(False)
        all_logits, all_y = [], []
        total_loss, n = 0.0, 0
        for batch in loader:
            x, y = batch[0].to(self.device), batch[1].to(self.device)
            logits = self.model(x)
            loss = F.cross_entropy(logits, y, weight=self.class_weights)
            total_loss += float(loss.item()) * x.size(0)
            n += x.size(0)
            all_logits.append(logits.detach().cpu())
            all_y.append(y.detach().cpu())
        logits = torch.cat(all_logits).numpy()
        y_true = torch.cat(all_y).numpy()
        preds = logits.argmax(axis=1)
        metrics = compute_classification_metrics(y_true, preds, n_classes=self.n_classes)
        metrics["loss"] = total_loss / max(n, 1)
        return metrics

    # ------------------------------------------------------------------ main
    def fit(self, run_params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        params = dict(run_params or {})
        params.update({"train_config": self.config.to_dict(), "device": str(self.device)})
        self.callback.start(params)
        try:
            stopped_epoch = 0
            for epoch in range(1, self.config.n_epochs + 1):
                stopped_epoch = epoch
                tr_loss = self._train_one_epoch()
                val_metrics = self._run_validation(self.val_loader)
                if self.scheduler is not None:
                    self.scheduler.step()

                step_metrics = {
                    "train/loss": tr_loss,
                    "val/loss": val_metrics["loss"],
                    "val/acc": val_metrics["accuracy"],
                    "val/macro_f1": val_metrics["macro_f1"],
                    "val/balanced_acc": val_metrics["balanced_accuracy"],
                }
                self.callback.log_metrics(step_metrics, step=epoch)

                if val_metrics["macro_f1"] > self.best_val:
                    self.best_val = val_metrics["macro_f1"]
                    self.best_epoch = epoch
                    self._patience = 0
                    if self.ckpt_path is not None:
                        save_checkpoint(
                            self.ckpt_path,
                            self.model,
                            epoch=epoch,
                            val_metric=self.best_val,
                            config=params,
                        )
                else:
                    self._patience += 1
                    if self._patience >= self.config.early_stop_patience:
                        break

            summary = {
                "best_val_macro_f1": self.best_val,
                "best_epoch": self.best_epoch,
                "stopped_epoch": stopped_epoch,
            }
            self.callback.log_metrics(
                {"summary/best_val_macro_f1": self.best_val, "summary/best_epoch": self.best_epoch},
                step=stopped_epoch,
            )
            return summary
        finally:
            self.callback.end()

    @torch.no_grad()
    def evaluate_on(self, loader: DataLoader) -> dict[str, float]:
        """Public eval entrypoint (e.g. for the test set)."""
        return self._run_validation(loader)
