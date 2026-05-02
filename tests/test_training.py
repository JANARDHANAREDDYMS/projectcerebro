from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from ml_core.data.dataset import EpochDataset, collate_epoch_batch
from ml_core.data.normalize import compute_train_stats
from ml_core.data.splits import make_subject_split, split_dataframe
from ml_core.models.shallowconv import ShallowConvNet
from ml_core.training.checkpoint import load_checkpoint
from ml_core.training.trainer import train_model


def test_one_epoch_training_and_checkpoint_reload(tmp_path, synthetic_epoch_df):
    split = make_subject_split(synthetic_epoch_df["subject_id"].tolist(), seed=42)
    parts = split_dataframe(synthetic_epoch_df, split)
    stats = compute_train_stats(parts["train"])
    train_loader = DataLoader(
        EpochDataset(parts["train"], stats=stats),
        batch_size=8,
        shuffle=False,
        collate_fn=collate_epoch_batch,
    )
    val_loader = DataLoader(
        EpochDataset(parts["val"], stats=stats),
        batch_size=8,
        shuffle=False,
        collate_fn=collate_epoch_batch,
    )
    config = {"seed": 42, "n_epochs": 1, "patience": 1, "lr": 1e-3, "weight_decay": 1e-4}
    model = ShallowConvNet()
    result = train_model(model, train_loader, val_loader, config, tmp_path)
    assert result.best_checkpoint.exists()
    reloaded = ShallowConvNet()
    payload = load_checkpoint(result.best_checkpoint, reloaded)
    assert payload["epoch"] == 1
    with torch.no_grad():
        assert reloaded(torch.randn(2, 1, 5, 512)).shape == (2, 3)
