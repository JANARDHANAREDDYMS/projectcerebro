"""Tiny ShallowConvNet smoke run: confirms loader + model + trainer integrate.

Usage:
    python -m ml_core.experiments.train_smoke \
        --delta-path delta_lake/epochs_mi_v1_ch5_sr128_bp8_30 \
        --out-dir artifacts/checkpoints/smoke \
        --epochs 5 --limit-rows 200
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..models import ShallowConvNet
from ..training import TrainConfig, Trainer, set_global_seed
from ._common import add_common_args, build_callback, build_loaders, configure_logging


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="ShallowConvNet smoke trainer")
    add_common_args(parser)
    args = parser.parse_args()
    set_global_seed(args.seed)

    train_loader, val_loader, _test_loader, _stats, train_labels, info = build_loaders(args)
    model = ShallowConvNet(n_classes=3)

    cfg = TrainConfig(
        n_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        early_stop_patience=args.patience,
        seed=args.seed,
        grad_clip_norm=args.grad_clip_norm,
        use_class_weights=not args.no_class_weights,
        device=args.device,
    )
    cb = build_callback(
        args,
        run_name="shallow_smoke",
        tags={"phase": "smoke", "model": "ShallowConvNet"},
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
