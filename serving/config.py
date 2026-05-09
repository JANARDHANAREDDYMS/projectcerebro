"""Configuration constants for the ProjectCerebro serving layer."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EEGNET_CHECKPOINT = PROJECT_ROOT / "artifacts/checkpoints/eegnet_physionet_seed42/best.pt"
EEGNET_ALIGNER = PROJECT_ROOT / "artifacts/checkpoints/eegnet_physionet_seed42/euclidean_aligner.npz"
EEGNET_NORM = PROJECT_ROOT / "artifacts/checkpoints/eegnet_physionet_seed42/norm_stats.json"

SHALLOW_CHECKPOINT = PROJECT_ROOT / "artifacts/checkpoints/shallow_physionet_only_seed42/best.pt"
SHALLOW_ALIGNER = PROJECT_ROOT / "artifacts/checkpoints/shallow_physionet_only_seed42/euclidean_aligner.npz"
SHALLOW_NORM = PROJECT_ROOT / "artifacts/checkpoints/shallow_physionet_only_seed42/norm_stats.json"

ENSEMBLE_WEIGHT_SHALLOW = 0.8
ENSEMBLE_WEIGHT_EEGNET = 0.2

N_CHANNELS = 5
N_SAMPLES = 512
N_CLASSES = 3
FEATURE_LEN = N_CHANNELS * N_SAMPLES
LABEL_MAP = {0: "left", 1: "right", 2: "rest"}
CLASS_NAMES = [LABEL_MAP[i] for i in sorted(LABEL_MAP)]
VERSION = "1.0.0"

