"""Leave-one-subject-out EEGNet fine-tuning on BCI IV-2a.

For each BCI IV-2a subject, the fold uses that subject as test, the next BCI
subject as validation, and all remaining BCI subjects plus all PhysioNet
subjects as training data. This keeps validation/test subjects disjoint while
still using PhysioNet as transfer-training support in every fold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from ..data import EpochDataset, EuclideanAligner, SplitManifest, compute_norm_stats, read_epochs
from ..evaluation.metrics import save_classification_report
from ..evaluation.subject_eval import per_subject_metrics
from ..models import EEGNet
from ..training import TrainConfig, Trainer, load_checkpoint, set_global_seed
from ._common import add_common_args, build_callback, configure_logging
from .train_shallow_baseline import save_test_predictions


BCI_SUBJECTS = [f"A{i:02d}" for i in range(1, 10)]


def _mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0}


def _build_fold_loaders(args, df, fold_dir: Path, test_subject: str, val_subject: str):
    """Build train/val/test loaders for one LOSO fold."""
    bci_df = df[df["dataset"] == "bci_iv_2a"].copy()
    physio_df = df[df["dataset"] == "physionet"].copy()

    train_bci_subjects = [sid for sid in BCI_SUBJECTS if sid not in {test_subject, val_subject}]
    train_bci_df = bci_df[bci_df["subject_id"].astype(str).isin(train_bci_subjects)]
    train_df = pd.concat([train_bci_df, physio_df], ignore_index=True).reset_index(drop=True)
    val_df = bci_df[bci_df["subject_id"].astype(str) == val_subject].reset_index(drop=True)
    test_df = bci_df[bci_df["subject_id"].astype(str) == test_subject].reset_index(drop=True)

    if args.limit_rows is not None:
        train_df = train_df.head(args.limit_rows).reset_index(drop=True)
        val_df = val_df.head(max(8, args.limit_rows // 4)).reset_index(drop=True)
        test_df = test_df.head(max(8, args.limit_rows // 4)).reset_index(drop=True)

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            f"Empty LOSO split for test={test_subject}, val={val_subject}: "
            f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )

    fold_dir.mkdir(parents=True, exist_ok=True)
    manifest = SplitManifest(
        train_subjects=sorted(train_df["subject_id"].astype(str).unique().tolist()),
        val_subjects=[val_subject],
        test_subjects=[test_subject],
        seed=args.seed,
        ratios=(0.0, 0.0, 0.0),
    )
    manifest.to_json(fold_dir / "split_manifest.json")

    aligner = None
    if args.use_ea:
        aligner = EuclideanAligner().fit(train_df)
        aligner.save(fold_dir / "euclidean_aligner.npz")
    stats = compute_norm_stats(aligner.transform(train_df) if aligner else train_df)
    stats.to_json(fold_dir / "norm_stats.json")

    train_ds = EpochDataset(train_df, aligner=aligner, norm_stats=stats, shape_mode="bcnt")
    val_ds = EpochDataset(val_df, aligner=aligner, norm_stats=stats, shape_mode="bcnt")
    test_ds = EpochDataset(
        test_df, aligner=aligner, norm_stats=stats, shape_mode="bcnt", return_meta=True
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    train_labels = np.asarray(train_ds._labels)  # type: ignore[attr-defined]
    info = {
        "test_subject": test_subject,
        "val_subject": val_subject,
        "train_bci_subjects": train_bci_subjects,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "class_counts_train": train_ds.class_counts(),
        "use_ea": bool(args.use_ea),
    }
    return train_loader, val_loader, test_loader, train_labels, info


def _fold_subjects(subjects: list[str], fold_idx: int) -> tuple[str, str]:
    test_subject = subjects[fold_idx]
    val_subject = subjects[(fold_idx + 1) % len(subjects)]
    return test_subject, val_subject


def _fold_report_row(test_subject: str, overall: dict[str, Any]) -> dict[str, Any]:
    return {
        "test_subject": test_subject,
        "accuracy": float(overall["accuracy"]),
        "macro_f1": float(overall["macro_f1"]),
        "balanced_accuracy": float(overall["balanced_accuracy"]),
        "per_class_recall": overall["per_class_recall"],
        "n_samples": int(overall["n_samples"]),
    }


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="LOSO EEGNet fine-tune on BCI IV-2a.")
    add_common_args(parser)
    parser.add_argument("--pretrained", required=True, help="PhysioNet EEGNet checkpoint path.")
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze EEGNet block1/block2 and train only embed/classifier in each LOSO fold.",
    )
    parser.add_argument(
        "--backbone-lr-mult",
        type=float,
        default=1.0,
        help="Multiplier for EEGNet block1/block2 LR. Ignored when --freeze-backbone is set.",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=BCI_SUBJECTS,
        help="BCI subject IDs to rotate through, default A01-A09.",
    )
    args = parser.parse_args()
    set_global_seed(args.seed)

    df = read_epochs(
        args.delta_path,
        datasets=["physionet", "bci_iv_2a"],
        filter_version=args.filter_version,
        drop_synthetic_rest=getattr(args, "drop_synthetic_rest", False),
    )

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    fold_rows = []
    all_by_subject = {}

    for fold_idx, _subject in enumerate(args.subjects):
        test_subject, val_subject = _fold_subjects(args.subjects, fold_idx)
        fold_dir = out_root / f"test_{test_subject}"
        print(
            f"LOSO fold {fold_idx + 1}/{len(args.subjects)}: "
            f"test={test_subject} val={val_subject}",
            flush=True,
        )

        train_loader, val_loader, test_loader, train_labels, info = _build_fold_loaders(
            args, df, fold_dir, test_subject, val_subject
        )
        model = EEGNet(n_classes=3, embed_dim=args.embed_dim, dropout=args.dropout)
        load_checkpoint(args.pretrained, model, map_location="cpu", strict=False)
        if args.freeze_backbone:
            for p in model.block1.parameters():
                p.requires_grad = False
            for p in model.block2.parameters():
                p.requires_grad = False
        param_groups = None
        if not args.freeze_backbone and args.backbone_lr_mult != 1.0:
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
            run_name=f"loso_eegnet_test_{test_subject}",
            tags={
                "phase": "loso_finetune",
                "model": "EEGNet",
                "test_subject": test_subject,
                "val_subject": val_subject,
                "dropout": str(args.dropout),
                "freeze_backbone": str(args.freeze_backbone),
                "backbone_lr_mult": str(args.backbone_lr_mult),
            },
        )
        trainer = Trainer(
            model,
            train_loader,
            val_loader,
            cfg,
            callback=cb,
            ckpt_path=fold_dir / "best.pt",
            train_label_array=train_labels,
            param_groups=param_groups,
        )
        summary = trainer.fit(
            {
                "info": json.dumps(info, sort_keys=True),
                "dropout": args.dropout,
                "freeze_backbone": args.freeze_backbone,
                "backbone_lr_mult": args.backbone_lr_mult,
            }
        )
        trainer.restore_best_checkpoint()

        preds, y_true, sids = save_test_predictions(
            trainer, test_loader, fold_dir / "test_predictions.jsonl"
        )
        overall = trainer.evaluate_on(test_loader)
        save_classification_report(overall, fold_dir / "test_overall.json")
        by_subject = per_subject_metrics(y_true, preds, sids, n_classes=3)
        (fold_dir / "test_by_subject.json").write_text(json.dumps(by_subject, indent=2))
        (fold_dir / "summary.json").write_text(json.dumps(summary, indent=2))

        row = _fold_report_row(test_subject, overall)
        row["val_subject"] = val_subject
        row["best_val_macro_f1"] = float(summary["best_val_macro_f1"])
        row["best_epoch"] = int(summary["best_epoch"])
        row["stopped_epoch"] = int(summary["stopped_epoch"])
        fold_rows.append(row)
        all_by_subject[test_subject] = row
        print({"fold": row}, flush=True)

    summary_report = {
        "n_folds": len(fold_rows),
        "subjects": args.subjects,
        "accuracy": _mean_std([row["accuracy"] for row in fold_rows]),
        "macro_f1": _mean_std([row["macro_f1"] for row in fold_rows]),
        "balanced_accuracy": _mean_std([row["balanced_accuracy"] for row in fold_rows]),
        "folds": fold_rows,
    }
    (out_root / "loso_per_subject.json").write_text(json.dumps(all_by_subject, indent=2))
    (out_root / "loso_summary.json").write_text(json.dumps(summary_report, indent=2))
    print(summary_report)


if __name__ == "__main__":
    main()
