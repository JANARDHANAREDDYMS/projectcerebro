"""End-to-end smoke test of trainer + dataset + checkpoint round-trip."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml_core.data import EpochDataset, compute_norm_stats, subject_split
from ml_core.models import ShallowConvNet
from ml_core.training import (
    TrainConfig,
    Trainer,
    load_checkpoint,
    set_global_seed,
)


def test_one_epoch_smoke(tmp_path, synthetic_epochs_df):
    set_global_seed(0)
    train_df, val_df, _test_df, _ = subject_split(synthetic_epochs_df, seed=0)
    stats = compute_norm_stats(train_df)
    train_ds = EpochDataset(train_df, norm_stats=stats)
    val_ds = EpochDataset(val_df, norm_stats=stats)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8)

    model = ShallowConvNet(n_classes=3)
    cfg = TrainConfig(n_epochs=2, lr=1e-3, batch_size=8, early_stop_patience=10, device="cpu")
    ckpt = tmp_path / "best.pt"
    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        cfg,
        ckpt_path=ckpt,
        train_label_array=np.asarray(train_ds._labels),  # type: ignore[attr-defined]
    )
    summary = trainer.fit()
    assert summary["stopped_epoch"] >= 1
    assert ckpt.exists()


def test_checkpoint_round_trip(tmp_path, synthetic_epochs_df):
    set_global_seed(0)
    train_df, val_df, _test_df, _ = subject_split(synthetic_epochs_df, seed=0)
    stats = compute_norm_stats(train_df)
    train_ds = EpochDataset(train_df, norm_stats=stats)
    val_ds = EpochDataset(val_df, norm_stats=stats)
    train_loader = DataLoader(train_ds, batch_size=8)
    val_loader = DataLoader(val_ds, batch_size=8)

    model = ShallowConvNet(n_classes=3)
    cfg = TrainConfig(n_epochs=1, batch_size=8, early_stop_patience=10, device="cpu")
    ckpt = tmp_path / "best.pt"
    Trainer(
        model,
        train_loader,
        val_loader,
        cfg,
        ckpt_path=ckpt,
        train_label_array=np.asarray(train_ds._labels),  # type: ignore[attr-defined]
    ).fit()

    fresh = ShallowConvNet(n_classes=3)
    payload = load_checkpoint(ckpt, fresh, map_location="cpu")
    assert payload.epoch >= 1

    # Same input -> same logits across the original and reloaded model.
    fresh.train(False)
    model.train(False)
    x = torch.randn(2, 1, 5, 512)
    with torch.no_grad():
        a = fresh(x).cpu().numpy()
        b = model.cpu()(x).cpu().numpy()
    np.testing.assert_allclose(a, b, atol=1e-5)
