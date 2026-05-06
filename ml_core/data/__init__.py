"""Data layer: Delta loader, schema, splits, normalization, torch Dataset."""
from .schema import (
    REQUIRED_COLUMNS,
    EXPECTED_FEATURE_LEN,
    N_CHANNELS,
    N_SAMPLES,
    SAMPLING_RATE_HZ,
    CHANNEL_NAMES,
    LABEL_MAP,
    INV_LABEL_MAP,
    validate_schema,
)
from .delta_loader import read_epochs
from .splits import subject_split, SplitManifest, loso_iter
from .normalize import compute_norm_stats, apply_norm_stats, NormStats
from .dataset import EpochDataset

__all__ = [
    "REQUIRED_COLUMNS",
    "EXPECTED_FEATURE_LEN",
    "N_CHANNELS",
    "N_SAMPLES",
    "SAMPLING_RATE_HZ",
    "CHANNEL_NAMES",
    "LABEL_MAP",
    "INV_LABEL_MAP",
    "validate_schema",
    "read_epochs",
    "subject_split",
    "SplitManifest",
    "loso_iter",
    "compute_norm_stats",
    "apply_norm_stats",
    "NormStats",
    "EpochDataset",
]
