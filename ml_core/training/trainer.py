from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ml_core.evaluation.metrics import compute_metrics
from ml_core.training.checkpoint import save_checkpoint


@dataclass
class TrainResult:
    best_metric: float
    best_checkpoint: Path
    history: list[dict[str, float]]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer_mps: bool = True) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_mps and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def class_weights_from_labels(labels: list[int], n_classes: int = 3) -> torch.Tensor | None:
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=n_classes).astype(np.float32)
    mean = counts[counts > 0].mean() if np.any(counts > 0) else 0
    if mean == 0 or np.all(counts >= 0.8 * mean):
        return None
    weights = np.where(counts > 0, mean / np.maximum(counts, 1), 0.0)
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, list[int], list[int], list[str]]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    n_items = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    subject_ids: list[str] = []

    for x, y, metas in loader:
        x = x.to(device)
        y = y.to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        if is_train:
            loss.backward()
            optimizer.step()
        batch_size = y.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        n_items += batch_size
        y_true.extend(y.detach().cpu().tolist())
        y_pred.extend(torch.argmax(logits.detach(), dim=1).cpu().tolist())
        subject_ids.extend([meta.subject_id for meta in metas])

    return total_loss / max(n_items, 1), y_true, y_pred, subject_ids


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    config: dict[str, Any],
    checkpoint_dir: str | Path,
    mlflow_logger=None,
) -> TrainResult:
    seed_everything(int(config.get("seed", 42)))
    device = get_device()
    model.to(device)

    labels = []
    if hasattr(train_loader.dataset, "df"):
        labels = train_loader.dataset.df["label_code"].astype(int).tolist()
    weights = class_weights_from_labels(labels)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device) if weights is not None else None)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("lr", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(config.get("n_epochs", 50)))
    )
    checkpoint_dir = Path(checkpoint_dir)
    best_path = checkpoint_dir / "best.pt"
    best_metric = -1.0
    patience = int(config.get("patience", 15))
    stale_epochs = 0
    history: list[dict[str, float]] = []

    if mlflow_logger:
        mlflow_logger.log_params(config)

    for epoch in range(1, int(config.get("n_epochs", 50)) + 1):
        train_loss, _, _, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, y_true, y_pred, _ = run_epoch(model, val_loader, criterion, device)
        metrics = compute_metrics(y_true, y_pred, prefix="val/")
        epoch_metrics = {
            "train/loss": train_loss,
            "val/loss": val_loss,
            "val/macro_f1": float(metrics["val/macro_f1"]),
            "val/acc": float(metrics["val/acc"]),
            "val/balanced_acc": float(metrics["val/balanced_acc"]),
        }
        history.append(epoch_metrics)
        if mlflow_logger:
            mlflow_logger.log_metrics(epoch_metrics, step=epoch)

        if epoch_metrics["val/macro_f1"] > best_metric:
            best_metric = epoch_metrics["val/macro_f1"]
            stale_epochs = 0
            save_checkpoint(best_path, model, optimizer, epoch, epoch_metrics, config)
        else:
            stale_epochs += 1

        scheduler.step()
        if stale_epochs >= patience:
            break

    return TrainResult(best_metric=best_metric, best_checkpoint=best_path, history=history)
