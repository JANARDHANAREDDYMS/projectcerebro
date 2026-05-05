"""Experiment 3: train EEGNet jointly on PhysioNet and BCI IV-2a.

Usage:
    python -m ml_core.experiments.train_joint_physionet_bci \
        --delta-path delta_lake/epochs_mi_v1_ch5_sr128_bp8_30 \
        --filter-version bp_8_30_v1 \
        --out-dir artifacts/checkpoints/eegnet_joint \
        --use-ea
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..evaluation.metrics import save_classification_report
from ..evaluation.subject_eval import per_subject_metrics
from ..models import EEGNet
from ..training import TrainConfig, Trainer, set_global_seed
from ._common import add_common_args, build_callback, build_loaders, configure_logging
from .train_shallow_baseline import save_test_predictions


def main() -> None:
    """Train one EEGNet on both datasets and evaluate held-out subjects."""
    configure_logging()
    parser = argparse.ArgumentParser(description="EEGNet joint PhysioNet + BCI training")
    add_common_args(parser)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.25)
    args = parser.parse_args()
    if args.datasets is None:
        args.datasets = ["physionet", "bci_iv_2a"]
    set_global_seed(args.seed)

    train_loader, val_loader, test_loader, _stats, train_labels, info = build_loaders(args)
    model = EEGNet(n_classes=3, embed_dim=args.embed_dim, dropout=args.dropout)

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
        run_name="eegnet_joint_physionet_bci",
        tags={"phase": "joint", "model": "EEGNet", "dataset": "physionet+bci_iv_2a"},
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
    )
    summary = trainer.fit({"info": str(info), "embed_dim": args.embed_dim, "dropout": args.dropout})
    trainer.restore_best_checkpoint()

    preds, y_true, sids = save_test_predictions(trainer, test_loader, out / "test_predictions.jsonl")
    overall = trainer.evaluate_on(test_loader)
    save_classification_report(overall, out / "test_overall.json")
    by_subject = per_subject_metrics(y_true, preds, sids, n_classes=3)
    (out / "test_by_subject.json").write_text(json.dumps(by_subject, indent=2))

    print({"summary": summary, "test_overall": overall})


if __name__ == "__main__":
    main()
