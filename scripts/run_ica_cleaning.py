from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import mne
from mne.preprocessing import ICA

try:
    from mne.preprocessing import EOGRegression
except ImportError:
    EOGRegression = None

from mne_icalabel import label_components

mne.set_log_level("WARNING")


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PHYSIONET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "physionet"
    / "physionet.org"
    / "files"
    / "eegmmidb"
    / "1.0.0"
)

BCI_ROOT = (
    PROJECT_ROOT
    / "data"
    / "bci_iv_2a"
    / "BCICIV_2a_gdf"
)

PRIVATE_ROOT = PROJECT_ROOT / "data" / "private"

CLEANED_ROOT = PROJECT_ROOT / "data_cleaned"
CLEANED_PHYSIONET_ROOT = CLEANED_ROOT / "physionet"
CLEANED_BCI_ROOT = CLEANED_ROOT / "bci_iv_2a"
CLEANED_PRIVATE_ROOT = CLEANED_ROOT / "private"

MANIFEST_PATH = CLEANED_ROOT / "stage1_cleaning_manifest.jsonl"


# =========================================================
# CONFIG
# =========================================================

PHYSIONET_RUNS = [4, 6, 8, 10, 12, 14]
CORRUPTED_PHYSIONET_SUBJECTS = {88, 92, 100, 104, 106}

L_FREQ = 1.0
H_FREQ = 100.0

ICA_N_COMPONENTS = 0.95
ICA_METHOD = "infomax"
ICA_FIT_PARAMS = {"extended": True}
ICA_RANDOM_STATE = 97
ICA_MAX_ITER = 500


REMOVE_NON_BRAIN_LABELS = {
    "muscle artifact",
    "eye blink",
    "heart beat",
    "line noise",
    "channel noise",
    "other",
}

TEST_MODE = False
TEST_PHYSIONET = [("S001", "S001R04")]
TEST_BCI = [("A01", "A01T")]


# =========================================================
# DATA MODEL
# =========================================================

@dataclass
class CleaningRecord:
    dataset: str
    subject_id: str
    run_id: str
    input_file: str
    output_file: str
    sfreq: float
    n_channels: int
    ica_n_components: int
    excluded_components: list[int]
    excluded_labels: list[str]
    cleaned_at_utc: str


# =========================================================
# HELPERS
# =========================================================

def ensure_output_dirs() -> None:
    CLEANED_PHYSIONET_ROOT.mkdir(parents=True, exist_ok=True)
    CLEANED_BCI_ROOT.mkdir(parents=True, exist_ok=True)
    CLEANED_PRIVATE_ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)


def append_manifest(record: CleaningRecord) -> None:
    with MANIFEST_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record)) + "\n")


def attach_standard_montage(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    raw2 = raw.copy()
    montage = mne.channels.make_standard_montage("standard_1005")
    raw2.set_montage(montage, on_missing="warn")
    return raw2


# =========================================================
# DATASET ITERATORS
# =========================================================

def iter_physionet_files() -> Iterable[tuple[str, str, Path]]:
    for subject_dir in sorted(PHYSIONET_ROOT.glob("S*")):
        if not subject_dir.is_dir():
            continue

        subject_id = subject_dir.name
        subject_num = int(subject_id[1:])

        if subject_num in CORRUPTED_PHYSIONET_SUBJECTS:
            continue

        for run in PHYSIONET_RUNS:
            run_id = f"{subject_id}R{run:02d}"
            fpath = subject_dir / f"{run_id}.edf"
            if fpath.exists():
                yield subject_id, run_id, fpath


def iter_bci_files() -> Iterable[tuple[str, str, Path]]:
    for subject_num in range(1, 10):
        subject_id = f"A{subject_num:02d}"

        # Training session only for now
        run_id = f"{subject_id}T"
        fpath = BCI_ROOT / f"{run_id}.gdf"

        if fpath.exists():
            yield subject_id, run_id, fpath


# =========================================================
# CHANNEL STANDARDIZATION
# Stage 1 uses montage-compatible names for ICA/ICLabel.
# Stage 2 will do the real channel alignment to common set.
# =========================================================

BCI_RENAME_MAP = {
    "EEG-Fz": "Fz",
    "EEG-C3": "C3",
    "EEG-Cz": "Cz",
    "EEG-C4": "C4",
    "EEG-Pz": "Pz",
    "EEG-0": "FC3",
    "EEG-1": "FC1",
    "EEG-2": "FCz",
    "EEG-3": "FC2",
    "EEG-4": "FC4",
    "EEG-5": "C5",
    "EEG-6": "C1",
    "EEG-7": "C2",
    "EEG-8": "C6",
    "EEG-9": "CP3",
    "EEG-10": "CP1",
    "EEG-11": "CPz",
    "EEG-12": "CP2",
    "EEG-13": "CP4",
    "EEG-14": "P1",
    "EEG-15": "P2",
    "EEG-16": "POz",
    "EOG-left": "EOG-LEFT",
    "EOG-central": "EOG-CENTRAL",
    "EOG-right": "EOG-RIGHT",
}


def normalize_physionet_name(ch: str) -> str:
    clean = ch.strip().replace(".", "")

    # First normalize to simple title form, e.g. "Fc5" / "Fcz"
    if len(clean) > 1:
        clean = clean[:1].upper() + clean[1:].lower()
    else:
        clean = clean.upper()

    # Fix montage-sensitive z channels and multi-letter prefixes
    replacements = {
        "Fpz": "Fpz",
        "Afz": "AFz",
        "Fcz": "FCz",
        "Cpz": "CPz",
        "Poz": "POz",
        "Cz": "Cz",
        "Fz": "Fz",
        "Pz": "Pz",
        "Oz": "Oz",
        "Iz": "Iz",
    }
    clean = replacements.get(clean, clean)

    if clean.startswith("Fc") and len(clean) > 2:
        clean = "FC" + clean[2:]
    if clean.startswith("Cp") and len(clean) > 2:
        clean = "CP" + clean[2:]
    if clean.startswith("Po") and len(clean) > 2:
        clean = "PO" + clean[2:]
    if clean.startswith("Af") and len(clean) > 2:
        clean = "AF" + clean[2:]
    if clean.startswith("Ft") and len(clean) > 2:
        clean = "FT" + clean[2:]
    if clean.startswith("Tp") and len(clean) > 2:
        clean = "TP" + clean[2:]

    return clean


def standardize_physionet(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    rename_map = {ch: normalize_physionet_name(ch) for ch in raw.ch_names}
    raw2 = raw.copy().rename_channels(rename_map)
    raw2 = attach_standard_montage(raw2)
    return raw2


def standardize_bci(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    raw2 = raw.copy().rename_channels(BCI_RENAME_MAP)

    eog_map = {}
    for ch in ["EOG-LEFT", "EOG-CENTRAL", "EOG-RIGHT"]:
        if ch in raw2.ch_names:
            eog_map[ch] = "eog"

    if eog_map:
        raw2.set_channel_types(eog_map)

    raw2 = attach_standard_montage(raw2)
    return raw2


# =========================================================
# IO
# =========================================================

def read_input_file(dataset: str, fpath: Path) -> mne.io.BaseRaw:
    if dataset == "physionet":
        raw = mne.io.read_raw_edf(str(fpath), preload=True, verbose=False)
        return standardize_physionet(raw)

    if dataset == "bci_iv_2a":
        raw = mne.io.read_raw_gdf(str(fpath), preload=True, verbose=False)
        return standardize_bci(raw)

    raise ValueError(f"Unsupported dataset: {dataset}")


def get_output_path(dataset: str, subject_id: str, run_id: str) -> Path:
    if dataset == "physionet":
        out_dir = CLEANED_PHYSIONET_ROOT / subject_id
    elif dataset == "bci_iv_2a":
        out_dir = CLEANED_BCI_ROOT / subject_id
    elif dataset == "private":
        out_dir = CLEANED_PRIVATE_ROOT / subject_id
    else:
        raise ValueError(dataset)

    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{run_id}_cleaned_raw.fif"


# =========================================================
# CLEANING STEPS
# =========================================================

def prepare_cleaning_branch(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    branch = raw.copy()

    safe_h_freq = min(H_FREQ, raw.info["sfreq"] / 2.0 - 1.0)
    if safe_h_freq <= L_FREQ:
        raise ValueError(
            f"Invalid filter bounds for sfreq={raw.info['sfreq']}: "
            f"l_freq={L_FREQ}, safe_h_freq={safe_h_freq}"
        )

    print(
        f"Applying broadband filter {L_FREQ}-{safe_h_freq:.1f} Hz "
        f"for sfreq={raw.info['sfreq']}"
    )

    branch.filter(
        l_freq=L_FREQ,
        h_freq=safe_h_freq,
        method="fir",
        phase="zero",
        verbose=False,
    )
    branch.set_eeg_reference(ref_channels="average", verbose=False)
    return branch


def maybe_apply_bci_eog_regression(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    if EOGRegression is None:
        print("Warning: EOGRegression not available in this MNE version. Skipping EOG regression.")
        return raw

    eog_chs = [ch for ch in raw.ch_names if ch.startswith("EOG-")]
    if not eog_chs:
        return raw

    try:
        model = EOGRegression(picks="eeg", picks_artifact=eog_chs)
        model.fit(raw)
        return model.apply(raw.copy())
    except Exception as e:
        print(f"Warning: EOG regression failed. Continuing without it. Error: {e}")
        return raw


def fit_ica(raw_branch: mne.io.BaseRaw) -> ICA:
    ica = ICA(
        n_components=ICA_N_COMPONENTS,
        method=ICA_METHOD,
        fit_params=ICA_FIT_PARAMS,
        random_state=ICA_RANDOM_STATE,
        max_iter=ICA_MAX_ITER,
    )
    ica.fit(raw_branch, picks="eeg", reject_by_annotation=True, verbose=False)
    return ica


def choose_components_to_exclude(raw_branch: mne.io.BaseRaw, ica: ICA) -> tuple[list[int], list[str]]:
    iclabel = label_components(raw_branch, ica, method="iclabel")
    labels = iclabel["labels"]

    exclude_idx = []
    exclude_labels = []

    for i, label in enumerate(labels):
        if label in REMOVE_NON_BRAIN_LABELS:
            exclude_idx.append(i)
            exclude_labels.append(label)

    return exclude_idx, exclude_labels


def apply_cleaning_to_continuous(raw_original: mne.io.BaseRaw, ica: ICA, exclude_idx: list[int]) -> mne.io.BaseRaw:
    cleaned = raw_original.copy()
    cleaned = ica.apply(cleaned, exclude=exclude_idx)
    return cleaned


# =========================================================
# MAIN WORKER
# =========================================================

def clean_one_file(dataset: str, subject_id: str, run_id: str, fpath: Path) -> None:
    print(f"\n[{dataset}] {run_id} -> loading {fpath}")

    raw = read_input_file(dataset, fpath)
    print(f"[{dataset}] {run_id} -> loaded, sfreq={raw.info['sfreq']}, channels={len(raw.ch_names)}")

    raw_branch = prepare_cleaning_branch(raw)
    print(f"[{dataset}] {run_id} -> broadband filter + CAR done")

    if dataset == "bci_iv_2a":
        raw_branch = maybe_apply_bci_eog_regression(raw_branch)
        print(f"[{dataset}] {run_id} -> EOG regression step done")

    print(f"[{dataset}] {run_id} -> fitting ICA")
    ica = fit_ica(raw_branch)

    exclude_idx, exclude_labels = choose_components_to_exclude(raw_branch, ica)
    print(f"[{dataset}] {run_id} -> excluding {len(exclude_idx)} ICs: {exclude_labels}")

    cleaned = apply_cleaning_to_continuous(raw, ica, exclude_idx)

    out_path = get_output_path(dataset, subject_id, run_id)
    cleaned.save(str(out_path), overwrite=True, verbose=False)

    n_comp = getattr(ica, "n_components_", None)
    if n_comp is None:
        n_comp = len(exclude_idx)

    record = CleaningRecord(
        dataset=dataset,
        subject_id=subject_id,
        run_id=run_id,
        input_file=str(fpath),
        output_file=str(out_path),
        sfreq=float(cleaned.info["sfreq"]),
        n_channels=len(cleaned.ch_names),
        ica_n_components=int(n_comp),
        excluded_components=exclude_idx,
        excluded_labels=exclude_labels,
        cleaned_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    append_manifest(record)

    print(f"[{dataset}] {run_id} -> saved to {out_path}")


# =========================================================
# ENTRYPOINT
# =========================================================

def main() -> None:
    ensure_output_dirs()

    if not PHYSIONET_ROOT.exists():
        raise FileNotFoundError(f"PhysioNet root not found: {PHYSIONET_ROOT}")

    if not BCI_ROOT.exists():
        raise FileNotFoundError(f"BCI root not found: {BCI_ROOT}")

    physionet_jobs = [("physionet", s, r, p) for s, r, p in iter_physionet_files()]
    bci_jobs = [("bci_iv_2a", s, r, p) for s, r, p in iter_bci_files()]

    if TEST_MODE:
        physionet_jobs = [
            job for job in physionet_jobs
            if (job[1], job[2]) in TEST_PHYSIONET
        ]
        bci_jobs = [
            job for job in bci_jobs
            if (job[1], job[2]) in TEST_BCI
        ]

    all_jobs = physionet_jobs + bci_jobs

    print(f"PhysioNet root: {PHYSIONET_ROOT}")
    print(f"BCI root:       {BCI_ROOT}")
    print(f"Test mode:      {TEST_MODE}")
    print(f"PhysioNet jobs: {len(physionet_jobs)}")
    print(f"BCI jobs:       {len(bci_jobs)}")
    print(f"Total jobs:     {len(all_jobs)}")

    for dataset, subject_id, run_id, fpath in all_jobs:
        try:
            clean_one_file(dataset, subject_id, run_id, fpath)
        except Exception as e:
            print(f"FAILED [{dataset}] {run_id}: {e}")

    print("\nStage 1 ICA cleaning complete.")


if __name__ == "__main__":
    main()