"""Few-shot subject adaptation for EEGNet motor imagery classification.

For each BCI IV-2a subject, this experiment samples a small balanced
calibration set from that subject, fits preprocessing statistics from the
calibration trials only, freezes the pretrained EEGNet backbone, fine-tunes
only the classifier head, and evaluates on the subject's remaining epochs.
It also reports a zero-shot baseline on the same evaluation epochs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ..data import EpochDataset, EuclideanAligner, NormStats, compute_norm_stats, read_epochs
from ..evaluation.metrics import compute_classification_metrics
from ..models import EEGNet
from ..training import load_checkpoint, pick_device, set_global_seed


BCI_SUBJECTS = ["A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08", "A09"]
DEFAULT_SHOTS = [5, 10, 20, 50]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Few-shot EEGNet subject adaptation.")
    parser.add_argument("--delta-path", required=True, help="Path to Delta/Parquet epoch table.")
    parser.add_argument("--filter-version", default="bp_8_30_v1")
    parser.add_argument("--pretrained", required=True, help="Path to pretrained EEGNet checkpoint.")
    parser.add_argument("--out-dir", required=True, help="Directory for JSON reports.")
    parser.add_argument("--shots-per-class", type=int, default=10)
    parser.add_argument("--adapt-epochs", type=int, default=10)
    parser.add_argument("--adapt-lr", type=float, default=1e-4)
    parser.add_argument("--adapt-weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="cuda|mps|cpu. Auto-selected if omitted.")
    parser.add_argument("--all-shots", action="store_true", help="Run shots 5, 10, 20, and 50.")
    parser.add_argument(
        "--unfreeze-last",
        action="store_true",
        help="Unfreeze last conv block in addition to classifier head.",
    )
    return parser.parse_args()


def load_pretrained_model(pretrained_path: str | Path, device: torch.device) -> EEGNet:
    """Load a fresh EEGNet from a checkpoint."""
    model = EEGNet(n_classes=3, n_channels=5, n_samples=512)
    load_checkpoint(pretrained_path, model, map_location=device, strict=False)
    model.to(device)
    return model


def sample_calibration_set(
    subject_df: pd.DataFrame,
    *,
    shots_per_class: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split one subject into balanced calibration and remaining evaluation epochs."""
    cal_rows: list[pd.DataFrame] = []
    for label in [0, 1, 2]:
        class_df = subject_df[subject_df["label_code"].astype(int) == label]
        if len(class_df) <= shots_per_class:
            sampled = class_df
        else:
            sampled = class_df.sample(n=shots_per_class, random_state=seed)
        cal_rows.append(sampled)

    calibration_df = pd.concat(cal_rows).sort_index()
    evaluation_df = subject_df.drop(index=calibration_df.index).sort_index()
    return calibration_df.reset_index(drop=True), evaluation_df.reset_index(drop=True)


def fit_calibration_preprocessors(
    calibration_df: pd.DataFrame,
) -> tuple[EuclideanAligner, NormStats]:
    """Fit EA and z-score normalization from calibration epochs only."""
    aligner = EuclideanAligner().fit(calibration_df)
    norm_stats = compute_norm_stats(aligner.transform(calibration_df))
    return aligner, norm_stats


def freeze_model(model: EEGNet, unfreeze_last: bool = False) -> EEGNet:
    """Freeze EEGNet except classifier and optionally the final convolution block."""
    for param in model.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True

    if unfreeze_last:
        for name, module in model.named_modules():
            if any(key in name.lower() for key in ["separable", "block2", "conv2", "depthwise2"]):
                for param in module.parameters():
                    param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {trainable:,}", flush=True)
    return model


def build_loader(
    df: pd.DataFrame,
    *,
    aligner: EuclideanAligner,
    norm_stats: NormStats,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Create a DataLoader for epochs using calibration-fitted preprocessing."""
    dataset = EpochDataset(df, aligner=aligner, norm_stats=norm_stats)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


@torch.no_grad()
def evaluate_model(model: EEGNet, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    """Evaluate a model and return classification metrics."""
    model.eval()
    preds_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    for x, y in loader:
        logits = model(x.to(device))
        preds_all.append(logits.argmax(dim=1).cpu().numpy())
        labels_all.append(y.numpy())

    y_true = np.concatenate(labels_all)
    y_pred = np.concatenate(preds_all)
    return compute_classification_metrics(y_true, y_pred, n_classes=3)


def adapt_classifier(
    model: EEGNet,
    calibration_loader: DataLoader,
    *,
    device: torch.device,
    adapt_epochs: int,
    adapt_lr: float,
    weight_decay: float,
    unfreeze_last: bool,
) -> None:
    """Fine-tune only the classifier layer on calibration trials."""
    model = freeze_model(model, unfreeze_last=unfreeze_last)
    if unfreeze_last and adapt_lr > 1e-4:
        adapt_lr = 1e-4
        print(f"  Auto-reduced adapt_lr to {adapt_lr} for unfreeze-last mode", flush=True)
    optimizer = AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=adapt_lr,
        weight_decay=weight_decay,
    )

    model.train()
    for _epoch in range(adapt_epochs):
        for x, y in calibration_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()


def metric_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Return mean/std summary for accuracy, macro F1, and balanced accuracy."""
    out: dict[str, float] = {}
    for key in ["accuracy", "macro_f1", "balanced_accuracy"]:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        out[f"mean_{key}"] = float(values.mean())
        out[f"std_{key}"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return out


def subject_result_row(
    *,
    subject_id: str,
    n_calibration: int,
    n_evaluation: int,
    metrics: dict[str, Any],
    zero_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the per-subject JSON row for one evaluation."""
    row = {
        "subject_id": subject_id,
        "n_calibration": int(n_calibration),
        "n_evaluation": int(n_evaluation),
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "per_class_precision": metrics["per_class_precision"],
        "per_class_recall": metrics["per_class_recall"],
        "per_class_f1": metrics["per_class_f1"],
    }
    if zero_metrics is not None:
        row["improvement_accuracy"] = float(metrics["accuracy"] - zero_metrics["accuracy"])
        row["improvement_macro_f1"] = float(metrics["macro_f1"] - zero_metrics["macro_f1"])
    return row


def run_subject_adaptation(
    *,
    subject_id: str,
    subject_df: pd.DataFrame,
    shots_per_class: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run zero-shot and few-shot evaluation for one subject."""
    calibration_df, evaluation_df = sample_calibration_set(
        subject_df, shots_per_class=shots_per_class, seed=args.seed
    )
    if evaluation_df.empty:
        raise ValueError(f"No evaluation epochs left for {subject_id} at {shots_per_class} shots.")

    aligner, norm_stats = fit_calibration_preprocessors(calibration_df)
    calibration_loader = build_loader(
        calibration_df,
        aligner=aligner,
        norm_stats=norm_stats,
        batch_size=min(args.batch_size, max(1, len(calibration_df))),
        shuffle=True,
    )
    evaluation_loader = build_loader(
        evaluation_df,
        aligner=aligner,
        norm_stats=norm_stats,
        batch_size=args.batch_size,
        shuffle=False,
    )

    zero_model = load_pretrained_model(args.pretrained, device)
    zero_metrics = evaluate_model(zero_model, evaluation_loader, device)

    few_model = load_pretrained_model(args.pretrained, device)
    adapt_classifier(
        few_model,
        calibration_loader,
        device=device,
        adapt_epochs=args.adapt_epochs,
        adapt_lr=args.adapt_lr,
        weight_decay=args.adapt_weight_decay,
        unfreeze_last=args.unfreeze_last,
    )
    few_metrics = evaluate_model(few_model, evaluation_loader, device)

    zero_row = subject_result_row(
        subject_id=subject_id,
        n_calibration=0,
        n_evaluation=len(evaluation_df),
        metrics=zero_metrics,
    )
    few_row = subject_result_row(
        subject_id=subject_id,
        n_calibration=len(calibration_df),
        n_evaluation=len(evaluation_df),
        metrics=few_metrics,
        zero_metrics=zero_metrics,
    )

    print(f"\n{'=' * 50}", flush=True)
    print(f"Subject {subject_id}", flush=True)
    print(f"  Calibration: {len(calibration_df)} trials", flush=True)
    print(f"  Evaluation:  {len(evaluation_df)} trials", flush=True)
    print(
        f"  Zero-shot:   acc={zero_metrics['accuracy']:.3f} "
        f"f1={zero_metrics['macro_f1']:.3f}",
        flush=True,
    )
    print(
        f"  {shots_per_class}-shot:    acc={few_metrics['accuracy']:.3f} "
        f"f1={few_metrics['macro_f1']:.3f}",
        flush=True,
    )
    print(
        f"  Improvement: acc={few_row['improvement_accuracy']:+.3f} "
        f"f1={few_row['improvement_macro_f1']:+.3f}",
        flush=True,
    )
    return zero_row, few_row


def run_for_shots(
    *,
    df: pd.DataFrame,
    shots_per_class: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    """Run few-shot adaptation for all BCI subjects at one shot count."""
    zero_rows: list[dict[str, Any]] = []
    few_rows: list[dict[str, Any]] = []

    for subject_idx, subject_id in enumerate(BCI_SUBJECTS):
        subject_df = df[df["subject_id"].astype(str) == subject_id].reset_index(drop=True)
        if subject_df.empty:
            print(f"Skipping missing subject {subject_id}", flush=True)
            continue
        print(f"\n{'=' * 50}", flush=True)
        print(f"Subject {subject_id} ({subject_idx + 1}/{len(BCI_SUBJECTS)})", flush=True)
        zero_row, few_row = run_subject_adaptation(
            subject_id=subject_id,
            subject_df=subject_df,
            shots_per_class=shots_per_class,
            args=args,
            device=device,
        )
        zero_rows.append(zero_row)
        few_rows.append(few_row)

    zero_summary = metric_summary(zero_rows)
    few_summary = metric_summary(few_rows)

    print(f"\n{'=' * 50}", flush=True)
    print("FEW-SHOT ADAPTATION SUMMARY", flush=True)
    print(f"Shots per class: {shots_per_class}", flush=True)
    print(
        f"Zero-shot:  acc={zero_summary['mean_accuracy']:.3f}"
        f"±{zero_summary['std_accuracy']:.3f}  "
        f"f1={zero_summary['mean_macro_f1']:.3f}±{zero_summary['std_macro_f1']:.3f}",
        flush=True,
    )
    print(
        f"Few-shot:   acc={few_summary['mean_accuracy']:.3f}"
        f"±{few_summary['std_accuracy']:.3f}  "
        f"f1={few_summary['mean_macro_f1']:.3f}±{few_summary['std_macro_f1']:.3f}",
        flush=True,
    )
    print(
        f"Improvement: acc={few_summary['mean_accuracy'] - zero_summary['mean_accuracy']:+.3f}  "
        f"f1={few_summary['mean_macro_f1'] - zero_summary['mean_macro_f1']:+.3f}",
        flush=True,
    )

    return {
        "shots_per_class": shots_per_class,
        "zero_shot": {**zero_summary, "per_subject": zero_rows},
        "few_shot": {**few_summary, "shots_per_class": shots_per_class, "per_subject": few_rows},
    }


def main() -> None:
    """Run few-shot subject adaptation and save JSON reports."""
    args = parse_args()
    set_global_seed(args.seed)
    device = pick_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_epochs(
        args.delta_path,
        filter_version=args.filter_version,
        dataset="bci_iv_2a",
        drop_synthetic_rest=False,
    )

    shot_values = DEFAULT_SHOTS if args.all_shots else [args.shots_per_class]
    shot_reports = [
        run_for_shots(df=df, shots_per_class=shots, args=args, device=device)
        for shots in shot_values
    ]

    config = {
        "delta_path": args.delta_path,
        "filter_version": args.filter_version,
        "pretrained": args.pretrained,
        "shots_per_class": shot_values if args.all_shots else args.shots_per_class,
        "adapt_epochs": args.adapt_epochs,
        "adapt_lr": args.adapt_lr,
        "adapt_weight_decay": args.adapt_weight_decay,
        "seed": args.seed,
        "device": str(device),
    }

    if args.all_shots:
        payload: dict[str, Any] = {"config": config, "results_by_shots": shot_reports}
    else:
        report = shot_reports[0]
        payload = {
            "config": config,
            "zero_shot": report["zero_shot"],
            "few_shot": report["few_shot"],
        }

    (out_dir / "few_shot_results.json").write_text(json.dumps(payload, indent=2))
    print({"wrote": str(out_dir / "few_shot_results.json")}, flush=True)


if __name__ == "__main__":
    main()
