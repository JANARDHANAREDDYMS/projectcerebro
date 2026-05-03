"""Read Stage 2 Delta Lake Parquet files into pandas DataFrames."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .schema import STAGE2_COLUMNS, filter_valid_rows, validate_schema


def _parquet_files(delta_path: str | Path) -> list[Path]:
    """Return sorted parquet data files from a Delta/Parquet directory or file."""
    path = Path(delta_path)
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Delta path not found: {path}")
    files = [p for p in sorted(path.glob("*.parquet")) if not p.name.startswith(".")]
    files = [p for p in files if _has_parquet_magic(p)]
    if not files:
        raise FileNotFoundError(f"No parquet files found under {path}")
    return files


def _has_parquet_magic(path: Path) -> bool:
    """Return true when a file has Parquet header and footer magic bytes."""
    if path.stat().st_size < 8:
        return False
    with path.open("rb") as handle:
        head = handle.read(4)
        handle.seek(-4, 2)
        tail = handle.read(4)
    return head == b"PAR1" and tail == b"PAR1"


def read_epochs(
    delta_path: str | Path,
    *,
    filter_version: str | None = None,
    dataset: str | None = None,
    datasets: Iterable[str] | None = None,
    drop_synthetic_rest: bool = False,
) -> pd.DataFrame:
    """Read Stage 2 epoch Parquet rows and apply optional filters.

    Parameters
    ----------
    delta_path:
        Directory such as `delta_lake/epochs_mi_v1_ch5_sr128_bp8_30` or one
        parquet file.
    filter_version:
        Optional stored filter string, e.g. `bp_8_30_v1`.
    dataset / datasets:
        Optional dataset filter. `dataset` accepts one name; `datasets` accepts
        multiple names and is kept for compatibility with experiment scripts.
    drop_synthetic_rest:
        If true, remove rows where `is_rest_synthetic` is true.
    """
    if dataset is not None and datasets is not None:
        raise ValueError("Pass either dataset or datasets, not both.")

    frames = []
    for file_path in _parquet_files(delta_path):
        frame = pd.read_parquet(file_path)
        keep_cols = [col for col in STAGE2_COLUMNS if col in frame.columns]
        frames.append(frame[keep_cols] if keep_cols else frame)

    df = pd.concat(frames, ignore_index=True)

    if filter_version is not None:
        df = df[df["filter_version"] == filter_version]

    selected = [dataset] if dataset is not None else list(datasets or [])
    if selected:
        df = df[df["dataset"].isin(selected)]

    if drop_synthetic_rest and "is_rest_synthetic" in df.columns:
        df = df[~df["is_rest_synthetic"].fillna(False).astype(bool)]

    df = filter_valid_rows(df)
    validate_schema(df)
    return df.reset_index(drop=True)
