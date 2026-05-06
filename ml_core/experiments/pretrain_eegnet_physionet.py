"""EEGNet pretraining on PhysioNet only.

Usage:
    python -m ml_core.experiments.pretrain_eegnet_physionet \
        --delta-path delta_lake/epochs_mi_v1_ch5_sr128_bp8_30 \
        --filter-version bp_8_30_v1 \
        --out-dir artifacts/checkpoints/eegnet_physionet
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..models import EEGNet
from ..training import TrainConfig, Trainer, set_global_seed
from ._common import add_common_args, build_callback, build_loaders, configure_logging


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="EEGNet PhysioNet pretrain")
    add_common_args(parser)
    args = parser.parse_args()
    if args.datasets is None:
        args.datasets = ["physionet"]
    set_global_seed(args.seed)

    train_loader, val_loader, _test_loader, _stats, train_labels, info = build_loaders(args)
    model = EEGNet(n_classes=3, embed_dim=128)

    cfg = TrainConfig(
        n_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        early_stop_patience=args.patience,
        seed=args.seed,
        device=args.device,
    )
    cb = build_callback(
        args,
        run_name="eegnet_pretrain_physionet",
        tags={"phase": "pretrain", "model": "EEGNet", "dataset": "physionet"},
    )
    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        cfg,
        callback=cb,
        ckpt_path=Path(args.out_dir) / "best.pt",
        train_label_array=train_labels,
    )
    summary = trainer.fit({"info": str(info)})
    print(summary)


if __name__ == "__main__":
    main()
