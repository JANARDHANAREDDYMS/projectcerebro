"""Data loading, splitting, alignment, and normalization for ML Core."""
from .dataset import EpochDataset
from .delta_loader import read_epochs
from .normalize import (
    EuclideanAligner,
    NormStats,
    apply_norm_stats,
    compute_norm_stats,
)
from .schema import (
    EXPECTED_FEATURE_LEN,
    LABEL_MAP,
    N_CHANNELS,
    N_SAMPLES,
    SchemaError,
    filter_valid_rows,
    validate_schema,
)
from .splits import SplitManifest, subject_split

__all__ = [
    "EXPECTED_FEATURE_LEN",
    "N_CHANNELS",
    "N_SAMPLES",
    "LABEL_MAP",
    "SchemaError",
    "validate_schema",
    "filter_valid_rows",
    "read_epochs",
    "SplitManifest",
    "subject_split",
    "EuclideanAligner",
    "NormStats",
    "compute_norm_stats",
    "apply_norm_stats",
    "EpochDataset",
]
