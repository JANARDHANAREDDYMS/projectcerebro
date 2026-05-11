"""
Build a fast Stream 1 cache for ProjectCerebro dashboard playback.

This script reads A09 cleaned FIF data directly from FIF binary tags, bypassing
MNE because MNE's FIF reader can hang on Apple Silicon. It extracts the five
motor-imagery dashboard channels, downsamples to 128 Hz with numpy interpolation,
and writes numpy cache files that FastAPI can load in milliseconds.

Usage:
    cerebro_env/bin/python scripts/read_fif_fast.py
"""
from __future__ import annotations

import json
import re
import struct
import time
import argparse
from pathlib import Path

import numpy as np

CLEANED_ROOT = Path("data_cleaned/bci_iv_2a")
CACHE_DIR = Path("artifacts/stream_cache")
TARGET_CHANNELS = ["FZ", "C3", "CZ", "C4", "PZ"]
SFREQ_TARGET = 128

# FIF tags used by this cleaned BCI IV-2a file.
FIFF_NCHAN = 200
FIFF_SFREQ = 201
FIFF_CH_INFO = 203
FIFF_DATA_BUFFER = 300
FIFFT_INT = 3
FIFFT_FLOAT = 4
FIFFT_DOUBLE = 5


def _normalize_channel(name: str) -> str:
    """Normalize EEG channel names for robust matching."""
    normalized = name.upper()
    for token in ("EEG", "REF"):
        normalized = normalized.replace(token, "")
    for char in (" ", "-", "_", "."):
        normalized = normalized.replace(char, "")
    return normalized


def _find_picks(channel_names: list[str], target_channels: list[str]) -> tuple[list[int], list[str]]:
    """Find target channel indices with exact-then-partial matching."""
    available = {_normalize_channel(ch): idx for idx, ch in enumerate(channel_names)}
    picks: list[int] = []
    found: list[str] = []
    used: set[int] = set()

    for channel in target_channels:
        key = _normalize_channel(channel)
        match = key if key in available else None
        if match is None:
            matches = [candidate for candidate in available if key in candidate]
            match = matches[0] if matches else None
        if match is None:
            continue

        idx = available[match]
        if idx in used:
            continue
        used.add(idx)
        picks.append(idx)
        found.append(channel)

    return picks, found


def _downsample_interp(data: np.ndarray, sfreq: float, target_sfreq: int) -> tuple[np.ndarray, np.ndarray]:
    """Downsample channel-first data with fast numpy linear interpolation."""
    if float(sfreq) == float(target_sfreq):
        times = np.arange(data.shape[1], dtype=np.float32) / float(target_sfreq)
        return data.astype(np.float32), times

    ratio = float(sfreq) / float(target_sfreq)
    if ratio == int(ratio):
        downsampled = data[:, :: int(ratio)].astype(np.float32)
        times = np.arange(downsampled.shape[1], dtype=np.float32) / float(target_sfreq)
        return downsampled, times

    old_times = np.arange(data.shape[1], dtype=np.float32) / float(sfreq)
    duration = float(old_times[-1])
    n_new = int(duration * target_sfreq) + 1
    new_times = np.arange(n_new, dtype=np.float32) / float(target_sfreq)

    out = np.empty((data.shape[0], n_new), dtype=np.float32)
    for channel_idx in range(data.shape[0]):
        out[channel_idx] = np.interp(new_times, old_times, data[channel_idx]).astype(np.float32)
    return out, new_times


def _channel_name_from_info(data_bytes: bytes) -> str | None:
    """Extract the channel name from a FIFF_CH_INFO tag."""
    text = data_bytes.decode("latin-1", errors="ignore").replace("\x00", " ")
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9-]{0,15}", text)
    tokens = [token for token in tokens if token.upper() not in {"EEG", "REF"}]
    return tokens[-1] if tokens else None


def resolve_fif_path(subject: str) -> Path:
    """Resolve the cleaned FIF path for a BCI IV-2a subject."""
    subject = subject.upper()
    candidates = [
        CLEANED_ROOT / subject / f"{subject}T_cleaned_raw.fif",
        CLEANED_ROOT / subject / f"{subject}_cleaned_raw.fif",
    ]
    for path in candidates:
        if path.exists():
            return path

    matches = sorted((CLEANED_ROOT / subject).glob(f"{subject}*_cleaned_raw.fif"))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"No cleaned FIF found for {subject}. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


def read_fif_binary(fif_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Read the cleaned FIF file using direct FIF binary tag parsing."""
    if not fif_path.exists():
        raise FileNotFoundError(f"FIF file not found: {fif_path}")
    stat = fif_path.stat()
    if stat.st_size == 0 or getattr(stat, "st_blocks", 1) == 0:
        raise ValueError(
            f"{fif_path} exists but appears empty/offloaded locally "
            f"(size={stat.st_size}, blocks={getattr(stat, 'st_blocks', 'unknown')}). "
            "Download or copy the actual FIF file before building this cache."
        )

    data_blocks: list[np.ndarray] = []
    channel_names: list[str] = []
    sfreq = 250.0
    n_channels = 0

    with fif_path.open("rb") as handle:
        while True:
            header = handle.read(16)
            if len(header) < 16:
                break

            kind, dtype, size, next_pos = struct.unpack(">iiii", header)
            if size < 0 or size > 512 * 1024 * 1024:
                raise ValueError(f"Invalid FIF tag size {size} for kind={kind}")

            data_bytes = handle.read(size) if size > 0 else b""

            if kind == FIFF_NCHAN and dtype == FIFFT_INT and len(data_bytes) >= 4:
                n_channels = struct.unpack(">i", data_bytes[:4])[0]

            elif kind == FIFF_SFREQ:
                if dtype == FIFFT_FLOAT and len(data_bytes) >= 4:
                    sfreq = struct.unpack(">f", data_bytes[:4])[0]
                elif dtype == FIFFT_DOUBLE and len(data_bytes) >= 8:
                    sfreq = struct.unpack(">d", data_bytes[:8])[0]

            elif kind == FIFF_CH_INFO:
                channel_name = _channel_name_from_info(data_bytes)
                if channel_name:
                    channel_names.append(channel_name)

            elif kind == FIFF_DATA_BUFFER and data_bytes and n_channels > 0:
                if dtype == FIFFT_FLOAT:
                    flat = np.frombuffer(data_bytes, dtype=">f4")
                elif dtype == FIFFT_INT:
                    flat = np.frombuffer(data_bytes, dtype=">i4").astype(np.float32)
                elif dtype == FIFFT_DOUBLE:
                    flat = np.frombuffer(data_bytes, dtype=">f8").astype(np.float32)
                else:
                    flat = np.array([], dtype=np.float32)

                if flat.size and flat.size % n_channels == 0:
                    data_blocks.append(flat.reshape(-1, n_channels).T.astype(np.float32))

            if next_pos > 0:
                handle.seek(next_pos)
            elif next_pos == -1:
                break

    if not data_blocks:
        raise ValueError("No FIFF_DATA_BUFFER tags found")
    if not channel_names:
        raise ValueError("No FIFF_CH_INFO channel names found")

    data = np.concatenate(data_blocks, axis=1)
    picks, found = _find_picks(channel_names, TARGET_CHANNELS)
    if len(picks) != len(TARGET_CHANNELS):
        raise ValueError(
            f"Only found channels {found}; wanted {TARGET_CHANNELS}. "
            f"Available sample: {channel_names[:25]}"
        )

    selected = data[picks, :]
    downsampled, times = _downsample_interp(selected, sfreq, SFREQ_TARGET)
    return downsampled.astype(np.float32), times.astype(np.float32), found


def save_cache(
    subject: str,
    fif_path: Path,
    data: np.ndarray,
    times: np.ndarray,
    channels: list[str],
    elapsed_sec: float,
) -> None:
    """Save dashboard stream cache files."""
    subject = subject.upper()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(CACHE_DIR / f"{subject}_continuous.npy", data.astype(np.float32))
    np.save(CACHE_DIR / f"{subject}_times.npy", times.astype(np.float32))

    meta = {
        "channels": channels,
        "sfreq": SFREQ_TARGET,
        "shape": list(data.shape),
        "duration": float(times[-1]),
        "working_approach": "Approach 3: FIF binary",
        "subject": subject,
        "source": str(fif_path),
        "read_time_sec": elapsed_sec,
    }
    with (CACHE_DIR / f"{subject}_continuous_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Build a ProjectCerebro Stream 1 FIF cache.")
    parser.add_argument(
        "--subject",
        default="A09",
        help="BCI IV-2a subject ID, e.g. A01 through A09. Default: A09.",
    )
    return parser.parse_args()


def main() -> int:
    """Read one subject's FIF with direct binary parsing and write stream cache."""
    args = parse_args()
    subject = args.subject.upper()
    fif_path = resolve_fif_path(subject)

    print("Running Approach 3: direct FIF binary reader", flush=True)
    print(f"Subject: {subject}", flush=True)
    print(f"Source: {fif_path}", flush=True)
    start = time.time()

    data, times, channels = read_fif_binary(fif_path)
    elapsed = time.time() - start
    save_cache(subject, fif_path, data, times, channels, elapsed)

    print(f"SUCCESS in {elapsed:.2f}s")
    print(f"Shape: {data.shape}")
    print(f"Duration: {float(times[-1]):.1f}s")
    print(f"Channels: {channels}")
    print(f"Range: min={float(data.min()):.6f} max={float(data.max()):.6f} mean={float(data.mean()):.6f}")
    print("\nSaved to artifacts/stream_cache/")
    print(f"  {subject}_continuous.npy      {data.nbytes / 1024 / 1024:.1f} MB")
    print(f"  {subject}_times.npy")
    print(f"  {subject}_continuous_meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
