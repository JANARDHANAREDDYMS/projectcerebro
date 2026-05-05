"""Few-shot subject adaptation with EEGNet + ShallowConvNet ensembling."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ..data import EpochDataset, EuclideanAligner, NormStats, compute_norm_stats, read_epochs
from ..evaluation.metrics import compute_classification_metrics
from ..models.eegnet import EEGNet
from ..models.shallowconv import ShallowConvNet
from ..training import load_checkpoint, pick_device, set_global_seed


BCI_SUBJECTS = ["A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08", "A09"]
DEFAULT_SHOTS = [5, 10, 20, 50]
WEIGHT_SWEEP = [0.5, 0.6, 0.7, 0.8]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Few-shot EEGNet + ShallowConvNet ensemble.")
    parser.add_argument("--delta-path", required=True)
    parser.add_argument("--filter-version", default="bp_8_30_v1")
    parser.add_argument("--pretrained-eegnet", required=True)
    parser.add_argument("--pretrained-shallow", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--shots-per-class", type=int, default=10)
    parser.add_argument("--adapt-epochs", type=int, default=20)
    parser.add_argument("--adapt-lr", type=float, default=1e-3)
    parser.add_argument("--adapt-weight-decay", type=float, default=0.0)
    parser.add_argument("--ensemble-weight-eegnet", type=float, default=0.6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--all-shots", action="store_true")
    return parser.parse_args()


def sample_calibration_set(
    subject_df: pd.DataFrame,
    *,
    shots_per_class: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split one subject into stratified calibration and evaluation epochs."""
    cal_rows: list[pd.DataFrame] = []
    for label in [0, 1, 2]:
        class_df = subject_df[subject_df["label_code"].astype(int) == label]
        n = min(shots_per_class, len(class_df))
        if n:
            cal_rows.append(class_df.sample(n=n, random_state=seed))
    if not cal_rows:
        raise ValueError("Cannot sample calibration set from an empty subject DataFrame.")
    calibration_df = pd.concat(cal_rows).sort_index()
    evaluation_df = subject_df.drop(index=calibration_df.index).sort_index()
    return calibration_df.reset_index(drop=True), evaluation_df.reset_index(drop=True)


def fit_calibration_preprocessors(
    calibration_df: pd.DataFrame,
) -> tuple[EuclideanAligner, NormStats]:
    """Fit EA and z-score stats from calibration epochs only."""
    aligner = EuclideanAligner().fit(calibration_df)
    norm_stats = compute_norm_stats(aligner.transform(calibration_df))
    return aligner, norm_stats


def build_loader(
    df: pd.DataFrame,
    *,
    aligner: EuclideanAligner,
    norm_stats: NormStats,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Build a DataLoader using calibration-fitted preprocessing."""
    dataset = EpochDataset(df, aligner=aligner, norm_stats=norm_stats)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def load_eegnet(path: str | Path, device: torch.device) -> EEGNet:
    """Load a fresh EEGNet checkpoint."""
    model = EEGNet(n_classes=3, n_channels=5, n_samples=512)
    load_checkpoint(path, model, map_location=device, strict=False)
    model.to(device)
    return model


def load_shallow(path: str | Path, device: torch.device) -> ShallowConvNet:
    """Load a fresh ShallowConvNet checkpoint."""
    model = ShallowConvNet(n_classes=3, n_channels=5, n_samples=512)
    load_checkpoint(path, model, map_location=device, strict=False)
    model.to(device)
    return model


def print_shallow_linear_layers(model: ShallowConvNet) -> None:
    """Print ShallowConvNet linear layers so the classifier name is explicit."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            print(f"Linear layer: {name}", flush=True)


def freeze_except_classifier(model: nn.Module) -> int:
    """Freeze all parameters except `model.classifier` and return trainable count."""
    if not hasattr(model, "classifier"):
        raise AttributeError(f"{model.__class__.__name__} has no classifier attribute.")
    for param in model.parameters():
        param.requires_grad = False
    classifier = getattr(model, "classifier")
    for param in classifier.parameters():
        param.requires_grad = True
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def adapt_classifier(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    label: str,
) -> None:
    """Fine-tune only a model's classifier layer on calibration trials."""
    trainable = freeze_except_classifier(model)
    print(f"  {label} trainable parameters: {trainable}", flush=True)
    optimizer = AdamW(getattr(model, "classifier").parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for _epoch in range(epochs):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()


@torch.no_grad()
def predict_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Return softmax probabilities and labels for a loader."""
    model.eval()
    probs_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    for x, y in loader:
        logits = model(x.to(device))
        probs_all.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels_all.append(y.numpy())
    return np.concatenate(probs_all), np.concatenate(labels_all)


def metrics_from_probs(y_true: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    """Compute classification metrics from probabilities."""
    return compute_classification_metrics(y_true, probs.argmax(axis=1), n_classes=3)


def summarize(rows: list[dict[str, Any]], prefix: str = "") -> dict[str, Any]:
    """Summarize per-subject rows for one method."""
    out: dict[str, Any] = {}
    for key in ["accuracy", "macro_f1", "balanced_accuracy"]:
        row_key = f"{prefix}{key}" if prefix else key
        values = np.asarray([float(row[row_key]) for row in rows], dtype=np.float64)
        out[f"mean_{key}"] = float(values.mean())
        out[f"std_{key}"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return out


def row_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    """Flatten selected metric fields with a prefix."""
    return {
        f"{prefix}_accuracy": float(metrics["accuracy"]),
        f"{prefix}_macro_f1": float(metrics["macro_f1"]),
        f"{prefix}_balanced_accuracy": float(metrics["balanced_accuracy"]),
    }


def run_subject(
    *,
    subject_id: str,
    subject_df: pd.DataFrame,
    shots_per_class: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run zero-shot, few-shot, and ensemble evaluation for one subject."""
    calibration_df, evaluation_df = sample_calibration_set(
        subject_df, shots_per_class=shots_per_class, seed=args.seed
    )
    if evaluation_df.empty:
        raise ValueError(f"No evaluation epochs left for {subject_id}.")

    aligner, norm_stats = fit_calibration_preprocessors(calibration_df)
    cal_loader = build_loader(
        calibration_df,
        aligner=aligner,
        norm_stats=norm_stats,
        batch_size=min(args.batch_size, max(1, len(calibration_df))),
        shuffle=True,
    )
    eval_loader = build_loader(
        evaluation_df,
        aligner=aligner,
        norm_stats=norm_stats,
        batch_size=args.batch_size,
        shuffle=False,
    )

    zero_eeg = load_eegnet(args.pretrained_eegnet, device)
    zero_sh = load_shallow(args.pretrained_shallow, device)
    zero_eeg_probs, y_true = predict_probs(zero_eeg, eval_loader, device)
    zero_sh_probs, _ = predict_probs(zero_sh, eval_loader, device)
    zero_eeg_metrics = metrics_from_probs(y_true, zero_eeg_probs)
    zero_sh_metrics = metrics_from_probs(y_true, zero_sh_probs)
    zero_ens_probs = (
        args.ensemble_weight_eegnet * zero_eeg_probs
        + (1.0 - args.ensemble_weight_eegnet) * zero_sh_probs
    )
    zero_ens_metrics = metrics_from_probs(y_true, zero_ens_probs)

    eegnet = load_eegnet(args.pretrained_eegnet, device)
    shallow = load_shallow(args.pretrained_shallow, device)
    adapt_classifier(
        eegnet,
        cal_loader,
        device=device,
        epochs=args.adapt_epochs,
        lr=args.adapt_lr,
        weight_decay=args.adapt_weight_decay,
        label="EEGNet",
    )
    adapt_classifier(
        shallow,
        cal_loader,
        device=device,
        epochs=args.adapt_epochs,
        lr=args.adapt_lr,
        weight_decay=args.adapt_weight_decay,
        label="ShallowConvNet",
    )

    eeg_probs, y_true = predict_probs(eegnet, eval_loader, device)
    sh_probs, _ = predict_probs(shallow, eval_loader, device)
    eeg_metrics = metrics_from_probs(y_true, eeg_probs)
    sh_metrics = metrics_from_probs(y_true, sh_probs)

    fixed_weight = args.ensemble_weight_eegnet
    ens_probs = fixed_weight * eeg_probs + (1.0 - fixed_weight) * sh_probs
    ens_metrics = metrics_from_probs(y_true, ens_probs)

    sweep: dict[str, dict[str, Any]] = {}
    best_weight = fixed_weight
    best_metrics = ens_metrics
    for weight in WEIGHT_SWEEP:
        metrics = metrics_from_probs(y_true, weight * eeg_probs + (1.0 - weight) * sh_probs)
        sweep[f"{weight:.1f}"] = metrics
        if metrics["macro_f1"] > best_metrics["macro_f1"]:
            best_weight = weight
            best_metrics = metrics

    zero_row = {
        "subject_id": subject_id,
        "n_calibration": 0,
        "n_evaluation": int(len(evaluation_df)),
        **row_metrics("eegnet", zero_eeg_metrics),
        **row_metrics("shallow", zero_sh_metrics),
        **row_metrics("ensemble", zero_ens_metrics),
        "ensemble_weight_eegnet": fixed_weight,
        "ensemble_per_class_recall": zero_ens_metrics["per_class_recall"],
    }
    result_row = {
        "subject_id": subject_id,
        "n_calibration": int(len(calibration_df)),
        "n_evaluation": int(len(evaluation_df)),
        **row_metrics("eegnet", eeg_metrics),
        **row_metrics("shallow", sh_metrics),
        **row_metrics("ensemble", ens_metrics),
        "ensemble_weight_eegnet": fixed_weight,
        "best_weight_eegnet": float(best_weight),
        "best_weight_accuracy": float(best_metrics["accuracy"]),
        "best_weight_macro_f1": float(best_metrics["macro_f1"]),
        "per_class_recall": ens_metrics["per_class_recall"],
        "weight_sweep": {
            weight: {
                "accuracy": float(metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
            }
            for weight, metrics in sweep.items()
        },
        "improvement_vs_zero_shot": float(ens_metrics["macro_f1"] - zero_eeg_metrics["macro_f1"]),
        "improvement_accuracy_vs_zero_shot": float(ens_metrics["accuracy"] - zero_eeg_metrics["accuracy"]),
    }

    print(f"\nSubject {subject_id} | {shots_per_class}-shot", flush=True)
    print(
        f"  Zero-shot EEGNet:    acc={zero_eeg_metrics['accuracy']:.3f} "
        f"f1={zero_eeg_metrics['macro_f1']:.3f}",
        flush=True,
    )
    print(
        f"  Zero-shot Shallow:   acc={zero_sh_metrics['accuracy']:.3f} "
        f"f1={zero_sh_metrics['macro_f1']:.3f}",
        flush=True,
    )
    print(
        f"  Few-shot EEGNet:     acc={eeg_metrics['accuracy']:.3f} "
        f"f1={eeg_metrics['macro_f1']:.3f}  "
        f"Delta={eeg_metrics['macro_f1'] - zero_eeg_metrics['macro_f1']:+.3f}",
        flush=True,
    )
    print(
        f"  Few-shot Shallow:    acc={sh_metrics['accuracy']:.3f} "
        f"f1={sh_metrics['macro_f1']:.3f}  "
        f"Delta={sh_metrics['macro_f1'] - zero_sh_metrics['macro_f1']:+.3f}",
        flush=True,
    )
    print(
        f"  Ensemble ({fixed_weight:.1f}/{1.0 - fixed_weight:.1f}): "
        f"acc={ens_metrics['accuracy']:.3f} f1={ens_metrics['macro_f1']:.3f}  "
        f"Delta={ens_metrics['macro_f1'] - zero_eeg_metrics['macro_f1']:+.3f}",
        flush=True,
    )
    print(
        f"  Best sweep weight:   w_eeg={best_weight:.1f} "
        f"acc={best_metrics['accuracy']:.3f} f1={best_metrics['macro_f1']:.3f}",
        flush=True,
    )
    return zero_row, result_row


def run_for_shots(
    *,
    df: pd.DataFrame,
    shots_per_class: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run all subjects for one shots-per-class value."""
    zero_rows: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for index, subject_id in enumerate(BCI_SUBJECTS):
        subject_df = df[df["subject_id"].astype(str) == subject_id].reset_index(drop=True)
        if subject_df.empty:
            continue
        print(f"\n{'=' * 50}", flush=True)
        print(f"Subject {subject_id} ({index + 1}/9) | {shots_per_class}-shot", flush=True)
        zero_row, row = run_subject(
            subject_id=subject_id,
            subject_df=subject_df,
            shots_per_class=shots_per_class,
            args=args,
            device=device,
        )
        zero_rows.append(zero_row)
        rows.append(row)

    shot_key = f"{args.ensemble_weight_eegnet:.1f}"
    result = {
        "eegnet_only": summarize(rows, "eegnet_"),
        "shallow_only": summarize(rows, "shallow_"),
        f"ensemble_{shot_key}": summarize(rows, "ensemble_"),
        "best_weight_sweep": {
            "mean_accuracy": float(np.mean([row["best_weight_accuracy"] for row in rows])),
            "std_accuracy": float(np.std([row["best_weight_accuracy"] for row in rows], ddof=1)),
            "mean_macro_f1": float(np.mean([row["best_weight_macro_f1"] for row in rows])),
            "std_macro_f1": float(np.std([row["best_weight_macro_f1"] for row in rows], ddof=1)),
        },
        "per_subject": rows,
    }
    zero_result = {
        "eegnet": summarize(zero_rows, "eegnet_"),
        "shallow": summarize(zero_rows, "shallow_"),
        "ensemble": summarize(zero_rows, "ensemble_"),
        "per_subject": zero_rows,
    }
    return result, zero_result


def main() -> None:
    """Run few-shot ensemble adaptation and write JSON output."""
    args = parse_args()
    set_global_seed(args.seed)
    device = pick_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not np.isclose(args.ensemble_weight_eegnet + (1.0 - args.ensemble_weight_eegnet), 1.0):
        raise ValueError("Invalid ensemble weight.")

    shallow_probe = ShallowConvNet(n_classes=3, n_channels=5, n_samples=512)
    print_shallow_linear_layers(shallow_probe)

    df = read_epochs(
        args.delta_path,
        filter_version=args.filter_version,
        dataset="bci_iv_2a",
        drop_synthetic_rest=False,
    )
    shots_values = DEFAULT_SHOTS if args.all_shots else [args.shots_per_class]

    results: dict[str, Any] = {}
    zero_by_shots: dict[str, Any] = {}
    for shots in shots_values:
        shot_result, zero_result = run_for_shots(
            df=df, shots_per_class=shots, args=args, device=device
        )
        results[str(shots)] = shot_result
        zero_by_shots[str(shots)] = zero_result

    first_shot = str(shots_values[0])
    payload = {
        "config": {
            "delta_path": args.delta_path,
            "filter_version": args.filter_version,
            "pretrained_eegnet": args.pretrained_eegnet,
            "pretrained_shallow": args.pretrained_shallow,
            "shots_per_class": shots_values if args.all_shots else args.shots_per_class,
            "adapt_epochs": args.adapt_epochs,
            "adapt_lr": args.adapt_lr,
            "adapt_weight_decay": args.adapt_weight_decay,
            "ensemble_weight_eegnet": args.ensemble_weight_eegnet,
            "seed": args.seed,
            "device": str(device),
        },
        "results": results,
        "zero_shot_baseline": {
            "eegnet": zero_by_shots[first_shot]["eegnet"],
            "shallow": zero_by_shots[first_shot]["shallow"],
            "by_shots": zero_by_shots,
        },
    }
    out_path = out_dir / "few_shot_ensemble_results.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print({"wrote": str(out_path)}, flush=True)


if __name__ == "__main__":
    main()
