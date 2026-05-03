"""Shared helpers for experiment entrypoints."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from ..data import (
    EpochDataset,
    NormStats,
    compute_norm_stats,
    read_epochs,
    subject_split,
)
from ..training.callbacks import MLflowCallback, NoOpCallback


def configure_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", level=logging.INFO
    )


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--delta-path", required=True, help="Path to a Delta epochs table.")
    parser.add_argument("--filter-version", default=None, help="e.g. bp_8_30_v1")
    parser.add_argument("--datasets", nargs="*", default=None, help="Subset of datasets to load.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--out-dir", required=True, help="Output dir for ckpt + reports.")
    parser.add_argument("--mlflow-experiment", default=None, help="Skip MLflow if absent.")
    parser.add_argument(
        "--mlflow-uri", default="file://./artifacts/mlruns", help="MLflow tracking URI."
    )
    parser.add_argument("--device", default=None, help="cuda|mps|cpu (auto if absent)")
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=None,
        help="Smoke-test convenience: keep only first N rows after split.",
    )


def build_callback(args, run_name: str, tags: dict[str, str]):
    if args.mlflow_experiment:
        return MLflowCallback(
            args.mlflow_experiment,
            tracking_uri=args.mlflow_uri,
            run_name=run_name,
            tags=tags,
        )
    return NoOpCallback()


def build_loaders(
    args,
    *,
    drop_synthetic_rest: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader, NormStats, np.ndarray, dict]:
    """Read Delta -> split by subject -> normalize -> torch loaders."""
    df = read_epochs(
        args.delta_path,
        datasets=args.datasets,
        filter_version=args.filter_version,
        drop_synthetic_rest=drop_synthetic_rest,
    )
    train_df, val_df, test_df, manifest = subject_split(df, seed=args.seed)

    if args.limit_rows is not None:
        train_df = train_df.head(args.limit_rows).reset_index(drop=True)
        val_df = val_df.head(max(8, args.limit_rows // 4)).reset_index(drop=True)
        test_df = test_df.head(max(8, args.limit_rows // 4)).reset_index(drop=True)

    stats = compute_norm_stats(train_df)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    stats.to_json(Path(args.out_dir) / "norm_stats.json")
    manifest.to_json(Path(args.out_dir) / "split_manifest.json")

    train_ds = EpochDataset(train_df, norm_stats=stats, shape_mode="bcnt")
    val_ds = EpochDataset(val_df, norm_stats=stats, shape_mode="bcnt")
    test_ds = EpochDataset(test_df, norm_stats=stats, shape_mode="bcnt", return_meta=True)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    train_labels = np.asarray(train_ds._labels)  # type: ignore[attr-defined]
    info = {
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "class_counts_train": train_ds.class_counts(),
        "split_manifest": manifest,
    }
    return train_loader, val_loader, test_loader, stats, train_labels, info
