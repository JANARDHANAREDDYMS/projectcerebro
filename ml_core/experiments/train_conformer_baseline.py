"""Train EEG Conformer baseline on Stage 2 epochs.

Usage:
    python -m ml_core.experiments.train_conformer_baseline \
        --delta-path delta_lake/epochs_mi_v1_ch5_sr128_bp8_30 \
        --filter-version bp_8_30_v1 \
        --out-dir artifacts/checkpoints/conformer_bp8_30 \
        --use-ea
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..evaluation.metrics import save_classification_report
from ..evaluation.subject_eval import per_subject_metrics
from ..models import EEGConformer
from ..training import TrainConfig, Trainer, set_global_seed
from ._common import add_common_args, build_callback, build_loaders, configure_logging
from .train_shallow_baseline import _collect_test_preds


def main() -> None:
    """Train EEG Conformer and write overall plus per-subject reports."""
    configure_logging()
    parser = argparse.ArgumentParser(description="EEG Conformer baseline")
    add_common_args(parser)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--conv-filters", type=int, default=40)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.25)
    args = parser.parse_args()
    if args.datasets is None:
        args.datasets = ["physionet", "bci_iv_2a"]
    set_global_seed(args.seed)

    train_loader, val_loader, test_loader, _stats, train_labels, info = build_loaders(args)
    model = EEGConformer(
        n_classes=3,
        embed_dim=args.embed_dim,
        conv_filters=args.conv_filters,
        n_heads=args.n_heads,
        depth=args.depth,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    )

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
        run_name="eeg_conformer_baseline",
        tags={"phase": "baseline", "model": "EEGConformer", "filter_version": args.filter_version or "all"},
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
    summary = trainer.fit(
        {
            "info": str(info),
            "embed_dim": args.embed_dim,
            "conv_filters": args.conv_filters,
            "n_heads": args.n_heads,
            "depth": args.depth,
            "ff_dim": args.ff_dim,
            "dropout": args.dropout,
        }
    )

    preds, y_true, sids = _collect_test_preds(trainer, test_loader)
    overall = trainer.evaluate_on(test_loader)
    save_classification_report(overall, out / "test_overall.json")
    by_subject = per_subject_metrics(y_true, preds, sids, n_classes=3)
    (out / "test_by_subject.json").write_text(json.dumps(by_subject, indent=2))

    print({"summary": summary, "test_overall": overall})


if __name__ == "__main__":
    main()
