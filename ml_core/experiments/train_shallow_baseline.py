"""Full ShallowConvNet baseline on the bp8_30 Delta table.

Usage:
    python -m ml_core.experiments.train_shallow_baseline \
        --delta-path delta_lake/epochs_mi_v1_ch5_sr128_bp8_30 \
        --filter-version bp_8_30_v1 \
        --out-dir artifacts/checkpoints/shallow_bp8_30 \
        --mlflow-experiment cerebro_week2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..evaluation.metrics import save_classification_report
from ..evaluation.subject_eval import per_subject_metrics
from ..models import ShallowConvNet
from ..training import TrainConfig, Trainer, set_global_seed
from ._common import add_common_args, build_callback, build_loaders, configure_logging


def _collect_test_preds(trainer: Trainer, test_loader):
    """Run inference on test_loader (which yields meta) and return preds + subject ids."""
    import numpy as np
    import torch

    trainer.model.train(False)
    preds_all, y_all, sids_all = [], [], []
    with torch.no_grad():
        for x, y, meta in test_loader:
            x = x.to(trainer.device)
            logits = trainer.model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            preds_all.append(preds)
            y_all.append(y.numpy())
            sids_all.extend(meta["subject_id"])
    return np.concatenate(preds_all), np.concatenate(y_all), sids_all


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="ShallowConvNet baseline")
    add_common_args(parser)
    args = parser.parse_args()
    set_global_seed(args.seed)

    train_loader, val_loader, test_loader, _stats, train_labels, info = build_loaders(args)
    model = ShallowConvNet(n_classes=3)

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
        run_name="shallow_baseline",
        tags={"phase": "baseline", "model": "ShallowConvNet", "filter_version": args.filter_version or "all"},
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
    summary = trainer.fit({"info": str(info)})

    # Test set evaluation with per-subject breakdown.
    preds, y_true, sids = _collect_test_preds(trainer, test_loader)
    overall = trainer.evaluate_on(test_loader)
    save_classification_report(overall, out / "test_overall.json")

    by_subject = per_subject_metrics(y_true, preds, sids, n_classes=3)
    (out / "test_by_subject.json").write_text(json.dumps(by_subject, indent=2))

    print({"summary": summary, "test_overall": overall})


if __name__ == "__main__":
    main()
