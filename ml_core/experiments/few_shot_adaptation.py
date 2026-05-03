"""Experiment 4: adapt a pretrained EEGNet with a few labeled trials per test subject.

The split is still subject-independent. For each held-out test subject, this
script samples a small support set from that subject, fine-tunes a fresh copy of
the pretrained model on those support trials only, and evaluates on the
remaining query trials for the same subject.

Usage:
    python -m ml_core.experiments.few_shot_adaptation \
        --delta-path delta_lake/epochs_mi_v1_ch5_sr128_bp8_30 \
        --filter-version bp_8_30_v1 \
        --pretrained artifacts/checkpoints/eegnet_physionet/best.pt \
        --out-dir artifacts/checkpoints/eegnet_few_shot \
        --shots-per-subject 10 \
        --use-ea
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ..data import (
    EpochDataset,
    EuclideanAligner,
    compute_norm_stats,
    read_epochs,
    subject_split,
)
from ..evaluation.metrics import compute_classification_metrics, save_classification_report
from ..evaluation.subject_eval import per_subject_metrics
from ..models import EEGNet
from ..training import load_checkpoint, pick_device, set_global_seed
from ._common import add_common_args, configure_logging


def _sample_support_indices(labels: np.ndarray, n_shots: int, seed: int) -> np.ndarray:
    """Sample up to `n_shots` support indices, balanced by class where possible."""
    rng = np.random.default_rng(seed)
    classes = sorted(np.unique(labels).astype(int).tolist())
    per_class = max(1, n_shots // max(len(classes), 1))
    selected: list[int] = []

    for cls in classes:
        cls_idx = np.flatnonzero(labels == cls)
        rng.shuffle(cls_idx)
        selected.extend(cls_idx[:per_class].tolist())

    if len(selected) < n_shots:
        remaining = np.asarray([i for i in range(len(labels)) if i not in set(selected)])
        rng.shuffle(remaining)
        selected.extend(remaining[: n_shots - len(selected)].tolist())

    selected = selected[: min(n_shots, len(labels) - 1)]
    return np.asarray(sorted(selected), dtype=int)


def _adapt_model(
    model: EEGNet,
    loader: DataLoader,
    device: torch.device,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> None:
    """Fine-tune one subject-specific model on the support loader."""
    model.to(device)
    model.train(True)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _epoch in range(epochs):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()


@torch.no_grad()
def _predict(model: EEGNet, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return predictions, labels, and subject IDs for a query loader."""
    model.to(device)
    model.train(False)
    preds_all, y_all, sids_all = [], [], []
    for x, y, meta in loader:
        logits = model(x.to(device))
        preds_all.append(logits.argmax(dim=1).cpu().numpy())
        y_all.append(y.numpy())
        sids_all.extend(meta["subject_id"])
    return np.concatenate(preds_all), np.concatenate(y_all), sids_all


def main() -> None:
    """Run per-subject few-shot adaptation from a pretrained EEGNet checkpoint."""
    configure_logging()
    parser = argparse.ArgumentParser(description="EEGNet few-shot subject adaptation")
    add_common_args(parser)
    parser.add_argument("--pretrained", required=True, help="Path to pretrained EEGNet checkpoint.")
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--shots-per-subject", type=int, default=10)
    parser.add_argument("--adapt-epochs", type=int, default=10)
    parser.add_argument("--adapt-lr", type=float, default=1e-4)
    parser.add_argument("--adapt-weight-decay", type=float, default=1e-4)
    args = parser.parse_args()
    if args.datasets is None:
        args.datasets = ["bci_iv_2a"]
    set_global_seed(args.seed)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = read_epochs(
        args.delta_path,
        datasets=args.datasets,
        filter_version=args.filter_version,
        drop_synthetic_rest=args.drop_synthetic_rest,
    )
    train_df, _val_df, test_df, manifest = subject_split(df, seed=args.seed)
    manifest.to_json(out / "split_manifest.json")

    aligner = EuclideanAligner().fit(train_df) if args.use_ea else None
    if aligner is not None:
        aligner.save(out / "euclidean_aligner.npz")
    stats = compute_norm_stats(aligner.transform(train_df) if aligner else train_df)
    stats.to_json(out / "norm_stats.json")

    base_model = EEGNet(n_classes=3, embed_dim=args.embed_dim)
    load_checkpoint(args.pretrained, base_model, map_location="cpu", strict=False)
    device = pick_device(args.device)

    all_preds, all_true, all_sids = [], [], []
    support_manifest: dict[str, dict[str, int]] = {}
    for subject_id, subject_df in test_df.groupby("subject_id", sort=True):
        subject_df = subject_df.reset_index(drop=True)
        if len(subject_df) <= args.shots_per_subject:
            continue

        labels = subject_df["label_code"].astype(int).to_numpy()
        support_idx = _sample_support_indices(labels, args.shots_per_subject, args.seed)
        support_mask = np.zeros(len(subject_df), dtype=bool)
        support_mask[support_idx] = True
        support_df = subject_df.loc[support_mask].reset_index(drop=True)
        query_df = subject_df.loc[~support_mask].reset_index(drop=True)
        if query_df.empty:
            continue

        support_ds = EpochDataset(support_df, aligner=aligner, norm_stats=stats)
        query_ds = EpochDataset(query_df, aligner=aligner, norm_stats=stats, return_meta=True)
        support_loader = DataLoader(support_ds, batch_size=min(args.batch_size, len(support_ds)), shuffle=True)
        query_loader = DataLoader(query_ds, batch_size=args.batch_size, shuffle=False)

        model = copy.deepcopy(base_model)
        _adapt_model(
            model,
            support_loader,
            device,
            epochs=args.adapt_epochs,
            lr=args.adapt_lr,
            weight_decay=args.adapt_weight_decay,
        )
        preds, y_true, sids = _predict(model, query_loader, device)
        all_preds.append(preds)
        all_true.append(y_true)
        all_sids.extend(sids)
        support_manifest[str(subject_id)] = {
            "n_support": int(len(support_df)),
            "n_query": int(len(query_df)),
        }

    if not all_preds:
        raise RuntimeError("No test subjects had enough trials for few-shot adaptation.")

    preds = np.concatenate(all_preds)
    y_true = np.concatenate(all_true)
    overall = compute_classification_metrics(y_true, preds, n_classes=3)
    save_classification_report(overall, out / "few_shot_overall.json")
    by_subject = per_subject_metrics(y_true, preds, all_sids, n_classes=3)
    (out / "few_shot_by_subject.json").write_text(json.dumps(by_subject, indent=2))
    (out / "support_query_manifest.json").write_text(json.dumps(support_manifest, indent=2))

    print({"few_shot_overall": overall, "subjects": len(support_manifest)})


if __name__ == "__main__":
    main()
