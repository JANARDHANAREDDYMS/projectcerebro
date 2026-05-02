from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from torch.utils.data import DataLoader

from ml_core.data.dataset import EpochDataset, collate_epoch_batch
from ml_core.data.delta_loader import read_delta
from ml_core.data.normalize import compute_train_stats, save_stats
from ml_core.data.schema import LABEL_MAP, PATH_FILTER_TO_COLUMN
from ml_core.data.splits import (
    make_pretrain_split,
    make_subject_split,
    save_split_manifest,
    split_dataframe,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def filter_to_delta_path(filter_key: str) -> Path:
    return PROJECT_ROOT / "delta_lake" / f"epochs_mi_v1_ch5_sr128_{filter_key}"


def make_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--filter", type=str, default="bp8_30", choices=sorted(PATH_FILTER_TO_COLUMN))
    parser.add_argument("--delta-path", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-synthetic-rest", action="store_true", default=True)
    parser.add_argument("--exclude-synthetic-rest", dest="include_synthetic_rest", action="store_false")
    return parser


def prepare_dataloaders(
    config: dict[str, Any],
    filter_key: str,
    delta_path: str | Path | None,
    dataset: str | None,
    pretrain: bool = False,
    limit: int | None = None,
    include_synthetic_rest: bool = True,
):
    path = Path(delta_path) if delta_path else filter_to_delta_path(filter_key)
    df = read_delta(
        path=path,
        filter_version=filter_key,
        dataset=dataset,
        include_synthetic_rest=include_synthetic_rest,
        limit=limit,
    )
    subjects = df["subject_id"].astype(str).tolist()
    split = make_pretrain_split(subjects, seed=int(config["seed"])) if pretrain else make_subject_split(subjects, seed=int(config["seed"]))
    parts = split_dataframe(df, split)
    stats = compute_train_stats(parts["train"])

    batch_size = int(config.get("batch_size", 64))
    loaders = {
        name: DataLoader(
            EpochDataset(part, stats=stats),
            batch_size=batch_size,
            shuffle=name == "train",
            collate_fn=collate_epoch_batch,
        )
        for name, part in parts.items()
        if not part.empty
    }
    return loaders, split, stats, path


def write_run_manifests(output_dir: str | Path, split, stats, config: dict[str, Any]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_split_manifest(split, output_dir / "split_manifest.json")
    save_stats(stats, output_dir / "normalization_stats.json")
    (output_dir / "label_map.json").write_text(json.dumps(LABEL_MAP, indent=2), encoding="utf-8")
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
