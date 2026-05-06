"""Read preprocessed epochs from Delta Lake into a pandas DataFrame.

We use `deltalake` (rust binding) for read paths because it has zero JVM/Spark
overhead. Spark is reserved for the upstream preprocessing in
`scripts/stage2_spark_preprocess.py`.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from .schema import (
    REQUIRED_COLUMNS,
    OPTIONAL_COLUMNS,
    filter_valid_rows,
    validate_schema,
)

log = logging.getLogger(__name__)


def _read_delta_pandas(path: Path) -> pd.DataFrame:
    """Read a Delta table to pandas.

    Tries `deltalake` first; falls back to PyArrow Parquet glob when the Delta
    log is unavailable (useful for partitioned snapshots copied without
    `_delta_log`).
    """
    try:
        from deltalake import DeltaTable

        return DeltaTable(str(path)).to_pandas()
    except Exception as exc:  # pragma: no cover - environment dependent
        log.warning("deltalake read failed (%s); falling back to parquet glob.", exc)
        import pyarrow.parquet as pq

        files = sorted(p for p in path.rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files under {path}") from exc
        return pq.ParquetDataset([str(f) for f in files]).read_pandas().to_pandas()


def read_epochs(
    delta_path: str | Path,
    *,
    datasets: Iterable[str] | None = None,
    filter_version: str | None = None,
    drop_synthetic_rest: bool = False,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Load epochs from a Delta table with optional filters.

    Parameters
    ----------
    delta_path: e.g. ``delta_lake/epochs_mi_v1_ch5_sr128_bp8_30``.
    datasets: subset of {"physionet", "bci_iv_2a"}; None = all.
    filter_version: stored value (e.g. ``"bp_8_30_v1"``); None = no filter.
    drop_synthetic_rest: if True and ``is_rest_synthetic`` exists, drop rows where it is True.
    columns: explicit projection. If None, returns all available columns.

    Returns
    -------
    A validated DataFrame with feature length == 2560 on every row.
    """
    path = Path(delta_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Delta path not found: {path.resolve()}. "
            "Download `delta_lake/` from the shared Drive into project root."
        )

    df = _read_delta_pandas(path)

    if datasets is not None:
        df = df[df["dataset"].isin(set(datasets))].reset_index(drop=True)
    if filter_version is not None:
        df = df[df["filter_version"] == filter_version].reset_index(drop=True)
    if drop_synthetic_rest and "is_rest_synthetic" in df.columns:
        df = df[~df["is_rest_synthetic"].astype(bool)].reset_index(drop=True)

    df = filter_valid_rows(df)
    validate_schema(df)

    if columns is not None:
        keep = [c for c in columns if c in df.columns]
        df = df[keep]

    log.info(
        "Loaded %d epochs from %s (datasets=%s, filter_version=%s)",
        len(df),
        path,
        datasets,
        filter_version,
    )
    return df


def list_known_columns(df: pd.DataFrame) -> list[str]:
    """Return required + present-optional columns in stable order."""
    present = list(REQUIRED_COLUMNS)
    present += [c for c in OPTIONAL_COLUMNS if c in df.columns]
    return present
