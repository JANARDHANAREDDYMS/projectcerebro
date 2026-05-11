"""A09 50-shot calibration comparison for EEGNet/ShallowConvNet ensembles.

Runs four setups with true BCI labels:

1. 50-shot ShallowConvNet only
2. 50-shot EEGNet + 50-shot ShallowConvNet ensemble
3. 50-shot ShallowConvNet + zero-shot EEGNet ensemble
4. 50-shot EEGNet + zero-shot ShallowConvNet ensemble
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ml_core.data import EpochDataset, EuclideanAligner, compute_norm_stats
from ml_core.evaluation.metrics import compute_classification_metrics
from ml_core.models.eegnet import EEGNet
from ml_core.models.shallowconv import ShallowConvNet
from ml_core.training import load_checkpoint, pick_device, set_global_seed


STREAM_COLUMNS = [
    "epoch_id",
    "dataset",
    "subject_id",
    "label_code",
    "label_name",
    "features",
    "epoch_start_sec",
    "epoch_end_sec",
    "filter_version",
    "preprocessing_version",
    "is_rest_synthetic",
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="A09 50-shot calibration comparison.")
    parser.add_argument("--delta-path", default="delta_lake/epochs_mi_v1_ch5_sr128_bp8_30")
    parser.add_argument("--filter-version", default="bp_8_30_v1")
    parser.add_argument("--subject", default="A09")
    parser.add_argument("--pretrained-eegnet", default="artifacts/checkpoints/eegnet_physionet_seed42/best.pt")
    parser.add_argument("--pretrained-shallow", default="artifacts/checkpoints/shallow_physionet_only_seed42/best.pt")
    parser.add_argument("--out-dir", default="artifacts/checkpoints/a09_50shot_compare_seed42")
    parser.add_argument("--shots-per-class", type=int, default=50)
    parser.add_argument("--adapt-epochs", type=int, default=20)
    parser.add_argument("--adapt-lr", type=float, default=1e-3)
    parser.add_argument("--adapt-weight-decay", type=float, default=0.0)
    parser.add_argument("--ensemble-weight-eegnet", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def has_parquet_magic(path: Path) -> bool:
    """Return true when a materialized file is valid parquet."""
    stat = path.stat()
    if stat.st_size < 8 or getattr(stat, "st_blocks", 1) == 0:
        return False
    with path.open("rb") as handle:
        head = handle.read(4)
        handle.seek(-4, 2)
        tail = handle.read(4)
    return head == b"PAR1" and tail == b"PAR1"


def load_subject_df(delta_path: str | Path, *, subject: str, filter_version: str) -> pd.DataFrame:
    """Load one BCI subject with direct parquet reads and placeholder filtering."""
    path = Path(delta_path)
    frames: list[pd.DataFrame] = []
    for parquet_path in sorted(path.glob("*.parquet")):
        if parquet_path.name.startswith(".") or not has_parquet_magic(parquet_path):
            continue
        frame = pd.read_parquet(parquet_path, columns=STREAM_COLUMNS)
        frame = frame[
            (frame["dataset"] == "bci_iv_2a")
            & (frame["filter_version"] == filter_version)
            & (frame["subject_id"].astype(str) == subject)
        ]
        if len(frame):
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No epochs found for {subject} in {delta_path}")
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["epoch_id"], keep="first")
    return df.sort_values(["epoch_start_sec", "epoch_id"], kind="stable").reset_index(drop=True)


def sample_calibration(
    df: pd.DataFrame,
    *,
    shots_per_class: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample a stratified calibration set with true labels."""
    cal_parts: list[pd.DataFrame] = []
    for label in [0, 1, 2]:
        class_df = df[df["label_code"].astype(int) == label]
        if len(class_df) < shots_per_class:
            raise ValueError(f"Class {label} has only {len(class_df)} epochs, need {shots_per_class}.")
        cal_parts.append(class_df.sample(n=shots_per_class, random_state=seed))
    calibration_df = pd.concat(cal_parts).sort_index()
    evaluation_df = df.drop(index=calibration_df.index).sort_index()
    return calibration_df.reset_index(drop=True), evaluation_df.reset_index(drop=True)


def fit_preprocessors(calibration_df: pd.DataFrame) -> tuple[EuclideanAligner, Any]:
    """Fit EA and norm stats on calibration epochs only."""
    aligner = EuclideanAligner().fit(calibration_df)
    norm_stats = compute_norm_stats(aligner.transform(calibration_df))
    return aligner, norm_stats


def make_loader(
    df: pd.DataFrame,
    *,
    aligner: EuclideanAligner,
    norm_stats: Any,
    batch_size: int,
    shuffle: bool,
    return_meta: bool = False,
) -> DataLoader:
    """Build a calibration-normalized loader."""
    dataset = EpochDataset(df, aligner=aligner, norm_stats=norm_stats, return_meta=return_meta)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def load_eegnet(path: str | Path, device: torch.device) -> EEGNet:
    """Load a fresh EEGNet."""
    model = EEGNet(n_classes=3, n_channels=5, n_samples=512)
    load_checkpoint(path, model, map_location=device, strict=False)
    model.to(device)
    return model


def load_shallow(path: str | Path, device: torch.device) -> ShallowConvNet:
    """Load a fresh ShallowConvNet."""
    model = ShallowConvNet(n_classes=3, n_channels=5, n_samples=512)
    load_checkpoint(path, model, map_location=device, strict=False)
    model.to(device)
    return model


def freeze_except_classifier(model: nn.Module) -> None:
    """Freeze all model parameters except the classifier head."""
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True


def adapt_classifier(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> float:
    """Adapt only the classifier head."""
    freeze_except_classifier(model)
    optimizer = AdamW(model.classifier.parameters(), lr=lr, weight_decay=weight_decay)
    final_loss = 0.0
    model.train()
    for _ in range(epochs):
        losses: list[float] = []
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        final_loss = float(np.mean(losses)) if losses else 0.0
    model.eval()
    return final_loss


@torch.no_grad()
def predict_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Return probabilities and labels."""
    model.eval()
    probs_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    for x, y, meta in loader:
        logits = model(x.to(device))
        probs_all.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels_all.append(y.numpy())
    return np.concatenate(probs_all), np.concatenate(labels_all)


def metrics_from_probs(y_true: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    """Compute metrics from probabilities."""
    return compute_classification_metrics(y_true, probs.argmax(axis=1), n_classes=3)


def setup_result(metrics: dict[str, Any]) -> dict[str, Any]:
    """Keep the high-signal metric fields for JSON output."""
    return {
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "per_class_precision": metrics["per_class_precision"],
        "per_class_recall": metrics["per_class_recall"],
        "per_class_f1": metrics["per_class_f1"],
        "confusion_matrix": metrics["confusion_matrix"],
    }


def write_predictions(path: Path, epoch_ids: list[str], y_true: np.ndarray, setup_probs: dict[str, np.ndarray]) -> None:
    """Write per-epoch predictions for all setups as JSONL."""
    with path.open("w", encoding="utf-8") as handle:
        for i, epoch_id in enumerate(epoch_ids):
            row = {"epoch_id": epoch_id, "true_label": int(y_true[i]), "setups": {}}
            for name, probs in setup_probs.items():
                pred = int(probs[i].argmax())
                row["setups"][name] = {
                    "pred": pred,
                    "confidence": float(probs[i, pred]),
                    "probs": [float(value) for value in probs[i]],
                }
            handle.write(json.dumps(row) + "\n")


def main() -> int:
    """Run the A09 50-shot comparison."""
    args = parse_args()
    set_global_seed(args.seed)
    device = pick_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    subject = args.subject.upper()
    print(f"Loading {subject} epochs...", flush=True)
    df = load_subject_df(args.delta_path, subject=subject, filter_version=args.filter_version)
    calibration_df, evaluation_df = sample_calibration(
        df,
        shots_per_class=args.shots_per_class,
        seed=args.seed,
    )
    print(
        f"Loaded {len(df)} subject epochs; calibration={len(calibration_df)} "
        f"evaluation={len(evaluation_df)}",
        flush=True,
    )
    print("Fitting calibration preprocessors...", flush=True)
    aligner, norm_stats = fit_preprocessors(calibration_df)
    cal_loader = make_loader(
        calibration_df,
        aligner=aligner,
        norm_stats=norm_stats,
        batch_size=min(args.batch_size, len(calibration_df)),
        shuffle=True,
    )
    eval_loader = make_loader(
        evaluation_df,
        aligner=aligner,
        norm_stats=norm_stats,
        batch_size=args.batch_size,
        shuffle=False,
        return_meta=True,
    )

    print("Running zero-shot EEGNet/ShallowConvNet...", flush=True)
    zero_eeg = load_eegnet(args.pretrained_eegnet, device)
    zero_shallow = load_shallow(args.pretrained_shallow, device)
    epoch_ids = [str(value) for value in evaluation_df["epoch_id"].tolist()]
    zero_eeg_probs, y_true = predict_probs(zero_eeg, eval_loader, device)
    zero_shallow_probs, _ = predict_probs(zero_shallow, eval_loader, device)

    print("Adapting EEGNet classifier...", flush=True)
    adapted_eeg = load_eegnet(args.pretrained_eegnet, device)
    eeg_loss = adapt_classifier(
        adapted_eeg,
        cal_loader,
        device=device,
        epochs=args.adapt_epochs,
        lr=args.adapt_lr,
        weight_decay=args.adapt_weight_decay,
    )
    print("Adapting ShallowConvNet classifier...", flush=True)
    adapted_shallow = load_shallow(args.pretrained_shallow, device)
    shallow_loss = adapt_classifier(
        adapted_shallow,
        cal_loader,
        device=device,
        epochs=args.adapt_epochs,
        lr=args.adapt_lr,
        weight_decay=args.adapt_weight_decay,
    )
    print("Evaluating adapted models and ensembles...", flush=True)
    adapted_eeg_probs, _ = predict_probs(adapted_eeg, eval_loader, device)
    adapted_shallow_probs, _ = predict_probs(adapted_shallow, eval_loader, device)

    w_eeg = float(args.ensemble_weight_eegnet)
    w_shallow = 1.0 - w_eeg
    setup_probs = {
        "zero_shot_ensemble": w_eeg * zero_eeg_probs + w_shallow * zero_shallow_probs,
        "50shot_shallow_only": adapted_shallow_probs,
        "50shot_both_ensemble": w_eeg * adapted_eeg_probs + w_shallow * adapted_shallow_probs,
        "50shot_shallow_plus_zero_eeg_ensemble": w_eeg * zero_eeg_probs + w_shallow * adapted_shallow_probs,
        "50shot_eeg_plus_zero_shallow_ensemble": w_eeg * adapted_eeg_probs + w_shallow * zero_shallow_probs,
        "50shot_eeg_only": adapted_eeg_probs,
        "zero_shot_eeg_only": zero_eeg_probs,
        "zero_shot_shallow_only": zero_shallow_probs,
    }
    results = {name: setup_result(metrics_from_probs(y_true, probs)) for name, probs in setup_probs.items()}

    calibration_counts = calibration_df["label_code"].astype(int).value_counts().sort_index().to_dict()
    evaluation_counts = evaluation_df["label_code"].astype(int).value_counts().sort_index().to_dict()
    summary = {
        "config": {
            "subject": subject,
            "delta_path": args.delta_path,
            "filter_version": args.filter_version,
            "pretrained_eegnet": args.pretrained_eegnet,
            "pretrained_shallow": args.pretrained_shallow,
            "shots_per_class": args.shots_per_class,
            "adapt_epochs": args.adapt_epochs,
            "adapt_lr": args.adapt_lr,
            "adapt_weight_decay": args.adapt_weight_decay,
            "ensemble_weight_eegnet": w_eeg,
            "ensemble_weight_shallow": w_shallow,
            "seed": args.seed,
            "device": str(device),
        },
        "data": {
            "n_subject_epochs": int(len(df)),
            "n_calibration": int(len(calibration_df)),
            "n_evaluation": int(len(evaluation_df)),
            "calibration_counts": {str(k): int(v) for k, v in calibration_counts.items()},
            "evaluation_counts": {str(k): int(v) for k, v in evaluation_counts.items()},
            "calibration_uses_true_labels": True,
        },
        "adaptation": {
            "eegnet_final_loss": eeg_loss,
            "shallow_final_loss": shallow_loss,
        },
        "results": results,
        "runtime_sec": time.time() - started,
    }

    with (out_dir / "a09_50shot_calibration_compare.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    calibration_df[["epoch_id", "label_code", "label_name", "epoch_start_sec", "epoch_end_sec"]].to_json(
        out_dir / "calibration_epochs.jsonl",
        orient="records",
        lines=True,
    )
    write_predictions(out_dir / "evaluation_predictions.jsonl", epoch_ids, y_true, setup_probs)

    print(f"\nSubject {subject} 50-shot comparison")
    print(f"  calibration={len(calibration_df)} evaluation={len(evaluation_df)}")
    for name in [
        "zero_shot_ensemble",
        "50shot_shallow_only",
        "50shot_both_ensemble",
        "50shot_shallow_plus_zero_eeg_ensemble",
        "50shot_eeg_plus_zero_shallow_ensemble",
    ]:
        metrics = results[name]
        print(
            f"  {name}: acc={metrics['accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} bal_acc={metrics['balanced_accuracy']:.4f}",
            flush=True,
        )
    print(f"\nWrote: {out_dir / 'a09_50shot_calibration_compare.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
