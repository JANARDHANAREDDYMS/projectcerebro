"""Compare two model prediction files to estimate ensemble complementarity.

Usage:
    python -m ml_core.experiments.compare_model_errors \
        --a artifacts/checkpoints/eegnet_bci/test_predictions.jsonl \
        --b artifacts/checkpoints/shallow_bci/test_predictions.jsonl \
        --a-name EEGNet \
        --b-name ShallowConvNet \
        --out artifacts/reports/eegnet_vs_shallow_errors.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append(row)
    return rows


def _accuracy(correct: int, total: int) -> float:
    return float(correct / total) if total else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare per-epoch errors from two models.")
    parser.add_argument("--a", required=True, help="First test_predictions.jsonl file.")
    parser.add_argument("--b", required=True, help="Second test_predictions.jsonl file.")
    parser.add_argument("--a-name", default="model_a")
    parser.add_argument("--b-name", default="model_b")
    parser.add_argument("--out", default=None, help="Optional JSON report path.")
    args = parser.parse_args()

    a_loaded = _load_jsonl(args.a)
    b_loaded = _load_jsonl(args.b)
    if len(a_loaded) == len(b_loaded):
        pairs = list(zip(a_loaded, b_loaded))
        alignment = "row_order"
    else:
        a_by_id = {str(row["epoch_id"]): row for row in a_loaded}
        b_by_id = {str(row["epoch_id"]): row for row in b_loaded}
        common_ids = sorted(set(a_by_id) & set(b_by_id))
        if not common_ids:
            raise ValueError("No overlapping epoch_id values. Ensure both runs used the same test split.")
        pairs = [(a_by_id[epoch_id], b_by_id[epoch_id]) for epoch_id in common_ids]
        alignment = "unique_epoch_id"

    counts = Counter()
    class_counts: dict[str, Counter] = defaultdict(Counter)
    disagreements = Counter()
    examples: dict[str, list[dict[str, Any]]] = {
        f"{args.a_name}_only_correct": [],
        f"{args.b_name}_only_correct": [],
        "both_wrong": [],
    }

    for row_idx, (a, b) in enumerate(pairs):
        if int(a["y_true"]) != int(b["y_true"]):
            raise ValueError(f"Label mismatch at row_idx={row_idx}, epoch_id={a.get('epoch_id')}")
        if alignment == "row_order" and str(a["epoch_id"]) != str(b["epoch_id"]):
            raise ValueError(
                f"Epoch mismatch at row_idx={row_idx}: {a.get('epoch_id')} != {b.get('epoch_id')}"
            )
        y_true = int(a["y_true"])
        a_pred = int(a["y_pred"])
        b_pred = int(b["y_pred"])
        a_correct = a_pred == y_true
        b_correct = b_pred == y_true
        label_key = str(y_true)

        counts["a_correct"] += int(a_correct)
        counts["b_correct"] += int(b_correct)
        counts["disagree"] += int(a_pred != b_pred)
        counts["oracle_correct"] += int(a_correct or b_correct)

        if a_correct and b_correct:
            bucket = "both_correct"
        elif a_correct:
            bucket = f"{args.a_name}_only_correct"
        elif b_correct:
            bucket = f"{args.b_name}_only_correct"
        else:
            bucket = "both_wrong"
        counts[bucket] += 1
        class_counts[label_key][bucket] += 1
        if a_pred != b_pred:
            disagreements[f"{a_pred}->{b_pred}"] += 1
        if bucket in examples and len(examples[bucket]) < 20:
            examples[bucket].append(
                    {
                        "epoch_id": str(a["epoch_id"]),
                        "row_idx": row_idx,
                        "subject_id": a.get("subject_id"),
                    "dataset": a.get("dataset"),
                    "y_true": y_true,
                    f"{args.a_name}_pred": a_pred,
                    f"{args.b_name}_pred": b_pred,
                }
            )

    total = len(pairs)
    report = {
        "model_a": args.a_name,
        "model_b": args.b_name,
        "alignment": alignment,
        "n_common": total,
        f"{args.a_name}_accuracy": _accuracy(counts["a_correct"], total),
        f"{args.b_name}_accuracy": _accuracy(counts["b_correct"], total),
        "oracle_accuracy_if_either_model_correct": _accuracy(counts["oracle_correct"], total),
        "disagreement_rate": _accuracy(counts["disagree"], total),
        "counts": dict(counts),
        "class_counts": {label: dict(counter) for label, counter in sorted(class_counts.items())},
        "top_prediction_disagreements": disagreements.most_common(20),
        "examples": examples,
    }

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
