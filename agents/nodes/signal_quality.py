"""Signal Quality Agent for raw EEG artifact checks."""
from __future__ import annotations

import numpy as np

from agents.state import BrainState

N_CHANNELS = 5
N_SAMPLES = 512
SAMPLING_RATE_HZ = 128.0
CHANNEL_NAMES = ["FZ", "C3", "CZ", "C4", "PZ"]


def signal_quality_node(state: BrainState) -> BrainState:
    """Score raw EEG signal quality and decide whether inference should run."""
    features = np.asarray(state["features"], dtype=np.float32)
    if features.size != N_CHANNELS * N_SAMPLES:
        return {
            **state,
            "signal_quality": "bad",
            "quality_score": 0.0,
            "bad_channels": CHANNEL_NAMES,
            "artifact_types": ["invalid_length"],
            "skip_inference": True,
        }

    epoch = features.reshape(N_CHANNELS, N_SAMPLES)
    artifact_types: list[str] = []
    bad_channels: list[str] = []
    scores: list[float] = []

    for channel_name, channel_data in zip(CHANNEL_NAMES, epoch):
        channel_score = 1.0
        amp_range = float(channel_data.max() - channel_data.min())
        variance = float(channel_data.var())

        if amp_range > 500.0:
            artifact_types.append("clipping")
            bad_channels.append(channel_name)
            channel_score -= 0.4

        if variance < 0.01:
            artifact_types.append("flat")
            bad_channels.append(channel_name)
            channel_score -= 0.5

        fft_mag = np.abs(np.fft.rfft(channel_data))
        freqs = np.fft.rfftfreq(N_SAMPLES, d=1.0 / SAMPLING_RATE_HZ)
        low_power = float(fft_mag[freqs < 30.0].sum())
        high_power = float(fft_mag[freqs >= 30.0].sum())
        high_freq_ratio = high_power / (low_power + high_power + 1e-8)
        if high_freq_ratio > 0.4:
            artifact_types.append("emg")
            bad_channels.append(channel_name)
            channel_score -= 0.3

        if abs(float(channel_data.mean())) > 100.0:
            artifact_types.append("drift")
            channel_score -= 0.2

        if np.allclose(channel_data, 0.0):
            artifact_types.append("disconnected")
            bad_channels.append(channel_name)
            channel_score = 0.0

        scores.append(max(0.0, channel_score))

    quality_score = float(np.mean(scores))
    unique_artifacts = sorted(set(artifact_types))
    unique_bad_channels = sorted(set(bad_channels))

    if quality_score >= 0.7:
        signal_quality = "good"
    elif quality_score >= 0.4:
        signal_quality = "noisy"
    else:
        signal_quality = "bad"

    print(f"[SignalQuality] epoch={state['epoch_id']} quality={signal_quality} score={quality_score:.3f}")
    return {
        **state,
        "signal_quality": signal_quality,
        "quality_score": quality_score,
        "bad_channels": unique_bad_channels,
        "artifact_types": unique_artifacts,
        "skip_inference": signal_quality == "bad",
    }

