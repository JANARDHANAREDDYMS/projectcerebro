"""EEGNet fine-tune on BCI IV-2a, optionally loading PhysioNet pretrained weights.

Usage:
    python -m ml_core.experiments.finetune_eegnet_bci \
        --delta-path delta_lake/epochs_mi_v1_ch5_sr128_bp8_30 \
        --filter-version bp_8_30_v1 \
        --pretrained artifacts/checkpoints/eegnet_physionet/best.pt \
        --out-dir artifacts/checkpoints/eegnet_bci
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..evaluation.metrics import save_classification_report
from ..evaluation.subject_eval import per_subject_metrics
from ..models import EEGNet
from ..training import TrainConfig, Trainer, load_checkpoint, set_global_seed
from ._common import add_common_args, build_callback, build_loaders, configure_logging
from .train_shallow_baseline import save_test_predictions


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="EEGNet BCI IV-2a fine-tune")
    add_common_args(parser)
    parser.add_argument(
        "--pretrained",
        default=None,
        help="Path to PhysioNet pretrained checkpoint (best.pt). Optional.",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze EEGNet feature blocks; train only the embedding/classifier heads.",
    )
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument(
        "--backbone-lr-mult",
        type=float,
        default=1.0,
        help="Multiplier for EEGNet block1/block2 LR during fine-tuning. Use 0.1 for gentler transfer.",
    )
    args = parser.parse_args()
    if args.datasets is None:
        args.datasets = ["bci_iv_2a"]
    set_global_seed(args.seed)

    train_loader, val_loader, test_loader, _stats, train_labels, info = build_loaders(args)

    model = EEGNet(n_classes=3, embed_dim=args.embed_dim, dropout=args.dropout)
    if args.pretrained:
        load_checkpoint(args.pretrained, model, map_location="cpu", strict=False)
    if args.freeze_backbone:
        for p in model.block1.parameters():
            p.requires_grad = False
        for p in model.block2.parameters():
            p.requires_grad = False
    param_groups = None
    if args.backbone_lr_mult != 1.0:
        backbone_params = list(model.block1.parameters()) + list(model.block2.parameters())
        head_params = list(model.embed.parameters()) + list(model.classifier.parameters())
        param_groups = [
            {"params": backbone_params, "lr": args.lr * args.backbone_lr_mult},
            {"params": head_params, "lr": args.lr},
        ]

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
        run_name="eegnet_finetune_bci",
        tags={
            "phase": "finetune",
            "model": "EEGNet",
            "dataset": "bci_iv_2a",
            "pretrained": str(args.pretrained or "none"),
            "dropout": str(args.dropout),
            "embed_dim": str(args.embed_dim),
            "backbone_lr_mult": str(args.backbone_lr_mult),
        },
    )
    out = Path(args.out_dir)
    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        cfg,
        callback=cb,
        ckpt_path=out / "best.pt",
        train_label_array=train_labels,
        param_groups=param_groups,
    )
    summary = trainer.fit(
        {
            "info": str(info),
            "embed_dim": args.embed_dim,
            "dropout": args.dropout,
            "backbone_lr_mult": args.backbone_lr_mult,
        }
    )
    trainer.restore_best_checkpoint()

    preds, y_true, sids = save_test_predictions(trainer, test_loader, out / "test_predictions.jsonl")
    overall = trainer.evaluate_on(test_loader)
    save_classification_report(overall, out / "test_overall.json")
    by_subject = per_subject_metrics(y_true, preds, sids, n_classes=3)
    (out / "test_by_subject.json").write_text(json.dumps(by_subject, indent=2))

    print({"summary": summary, "test_overall": overall})


if __name__ == "__main__":
    main()
