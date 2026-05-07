"""Leave-one-subject-out (LOSO) evaluation for ShallowConvNet or EEGNet.

Per held-out subject, train a fresh model on the rest and report per-subject
metrics. Aggregate mean ± std across folds. This is the honest cross-subject
generalization number for motor-imagery EEG.

Example
-------
::

    python -m ml_core.experiments.loso_eval \
        --model shallowconvnet \
        --delta-path delta_lake/epochs_mi_v1_ch5_sr128_bp8_30 \
        --filter-version bp_8_30_v1 \
        --holdout-dataset bci_iv_2a \
        --out-dir artifacts/reports/loso_shallow_bci \
        --epochs 30 --patience 10
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data import (
    EpochDataset,
    compute_norm_stats,
    loso_iter,
    read_epochs,
)
from ..evaluation.metrics import compute_classification_metrics
from ..evaluation.subject_eval import aggregate_loso
from ..models import build_model
from ..training import TrainConfig, Trainer, set_global_seed
from ._common import build_callback, configure_logging

log = logging.getLogger(__name__)


def _add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, choices=["shallowconvnet", "eegnet"])
    parser.add_argument("--delta-path", required=True)
    parser.add_argument("--filter-version", default=None)
    parser.add_argument(
        "--holdout-dataset",
        default="bci_iv_2a",
        help="Dataset to LOSO over. All other datasets are kept in train.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Smoke convenience: cap LOSO folds (skip remaining held-out subjects).",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--mlflow-experiment", default=None)
    parser.add_argument(
        "--mlflow-uri",
        default="sqlite:///artifacts/mlruns/mlflow.db",
        help="sqlite avoids the file-store race that hit our earlier runs.",
    )


def _collect_test(trainer: Trainer, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    trainer.model.train(False)
    preds_all, y_all = [], []
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(trainer.device), batch[1]
            logits = trainer.model(x)
            preds_all.append(logits.argmax(dim=1).cpu().numpy())
            y_all.append(y.numpy())
    return np.concatenate(preds_all), np.concatenate(y_all)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="LOSO evaluation harness")
    _add_args(parser)
    args = parser.parse_args()
    set_global_seed(args.seed)

    df = read_epochs(args.delta_path, filter_version=args.filter_version)
    log.info(
        "Loaded %d epochs across %d subjects; LOSO over dataset=%s",
        len(df),
        df["subject_id"].nunique(),
        args.holdout_dataset,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_fold: dict[str, dict] = {}
    fold_idx = 0
    started = time.time()

    for train_df, val_df, test_df, manifest in loso_iter(
        df,
        seed=args.seed,
        holdout_dataset=args.holdout_dataset,
        val_fraction=args.val_fraction,
        dataset_for_manifest=args.holdout_dataset,
        filter_version=args.filter_version,
    ):
        held = manifest.test_subjects[0]
        fold_idx += 1
        if args.max_subjects is not None and fold_idx > args.max_subjects:
            break
        log.info(
            "[fold %d] held=%s n_train=%d n_val=%d n_test=%d",
            fold_idx,
            held,
            len(train_df),
            len(val_df),
            len(test_df),
        )
        if len(test_df) == 0:
            log.warning("Held-out subject %s has zero epochs; skipping.", held)
            continue

        # Per-fold normalization computed on training subjects only.
        stats = compute_norm_stats(train_df)
        train_ds = EpochDataset(train_df, norm_stats=stats, shape_mode="bcnt")
        val_ds = EpochDataset(val_df, norm_stats=stats, shape_mode="bcnt")
        test_ds = EpochDataset(test_df, norm_stats=stats, shape_mode="bcnt")

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False
        )
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        model = build_model(args.model, n_classes=3)

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
            run_name=f"loso_{args.model}_held{held}",
            tags={
                "phase": "loso",
                "model": args.model,
                "held_out_subject": str(held),
                "holdout_dataset": args.holdout_dataset,
            },
        )
        ckpt = out_dir / f"fold_{held}.pt"
        trainer = Trainer(
            model,
            train_loader,
            val_loader,
            cfg,
            callback=cb,
            ckpt_path=ckpt,
            train_label_array=np.asarray(train_ds._labels),  # type: ignore[attr-defined]
        )
        summary = trainer.fit({"held_out_subject": held})
        preds, y_true = _collect_test(trainer, test_loader)
        metrics = compute_classification_metrics(y_true, preds, n_classes=3)
        metrics["best_val_macro_f1"] = float(summary["best_val_macro_f1"])
        metrics["stopped_epoch"] = int(summary["stopped_epoch"])
        per_fold[held] = metrics

        # Save per-fold split manifest for reproducibility.
        manifest.to_json(out_dir / f"fold_{held}_manifest.json")
        log.info(
            "[fold %d] held=%s acc=%.3f macro_f1=%.3f",
            fold_idx,
            held,
            metrics["accuracy"],
            metrics["macro_f1"],
        )

    summary = aggregate_loso(per_fold)
    summary["model"] = args.model
    summary["holdout_dataset"] = args.holdout_dataset
    summary["filter_version"] = args.filter_version
    summary["n_seconds"] = time.time() - started
    (out_dir / "loso_summary.json").write_text(json.dumps(summary, indent=2))
    log.info(
        "LOSO done. n_folds=%d acc=%.3f ± %.3f macro_f1=%.3f ± %.3f",
        summary["n_folds"],
        summary["mean"].get("accuracy", float("nan")),
        summary["std"].get("accuracy", float("nan")),
        summary["mean"].get("macro_f1", float("nan")),
        summary["std"].get("macro_f1", float("nan")),
    )


if __name__ == "__main__":
    main()
