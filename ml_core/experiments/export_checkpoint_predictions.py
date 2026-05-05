"""Export per-epoch predictions for an existing checkpoint.

Usage:
    python -m ml_core.experiments.export_checkpoint_predictions \
        --model eegnet \
        --checkpoint artifacts/checkpoints/eegnet_bci/best.pt \
        --delta-path delta_lake/epochs_mi_v1_ch5_sr128_bp8_30 \
        --filter-version bp_8_30_v1 \
        --datasets bci_iv_2a \
        --out-dir artifacts/checkpoints/eegnet_bci \
        --dropout 0.5 \
        --use-ea
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..models import EEGConformer, EEGNet, ShallowConvNet
from ..training import TrainConfig, Trainer, load_checkpoint, set_global_seed
from ._common import add_common_args, build_loaders, configure_logging
from .train_shallow_baseline import save_test_predictions


def _build_model(args):
    if args.model == "eegnet":
        return EEGNet(n_classes=3, embed_dim=args.embed_dim, dropout=args.dropout)
    if args.model == "shallow":
        return ShallowConvNet(n_classes=3, dropout=args.dropout)
    if args.model == "conformer":
        return EEGConformer(
            n_classes=3,
            embed_dim=args.embed_dim,
            conv_filters=args.conv_filters,
            n_heads=args.n_heads,
            depth=args.depth,
            ff_dim=args.ff_dim,
            dropout=args.dropout,
        )
    raise ValueError(f"Unsupported model: {args.model}")


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Export predictions for an existing checkpoint.")
    add_common_args(parser)
    parser.add_argument("--model", choices=["eegnet", "shallow", "conformer"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--conv-filters", type=int, default=40)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=256)
    args = parser.parse_args()

    set_global_seed(args.seed)
    _train_loader, val_loader, test_loader, _stats, train_labels, _info = build_loaders(args)
    model = _build_model(args)
    load_checkpoint(args.checkpoint, model, map_location="cpu", strict=True)
    cfg = TrainConfig(
        n_epochs=1,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        early_stop_patience=args.patience,
        seed=args.seed,
        grad_clip_norm=args.grad_clip_norm,
        use_class_weights=not args.no_class_weights,
        device=args.device,
    )
    trainer = Trainer(
        model,
        _train_loader,
        val_loader,
        cfg,
        train_label_array=train_labels,
    )
    out = Path(args.out_dir)
    save_test_predictions(trainer, test_loader, out / "test_predictions.jsonl")
    print({"wrote": str(out / "test_predictions.jsonl"), "checkpoint": args.checkpoint})


if __name__ == "__main__":
    main()
