"""
ProjectCerebro — Upload Cho 2017 Parquet to Hugging Face
=========================================================
Uploads the processed Cho 2017 parquet files from
parquet_export/ to the HuggingFace dataset repo.

Usage:
    python scripts/upload_cho2017_to_hf.py
    python scripts/upload_cho2017_to_hf.py --token <token>
    python scripts/upload_cho2017_to_hf.py --repo divyanshmaurya1/BCI_Data_new
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, DatasetCard


HF_TOKEN    = os.environ.get("HF_TOKEN", "")   # set via: export HF_TOKEN=<your_token>
HF_REPO_ID  = "divyanshmaurya1/BCI_Data_new"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARQUET_ROOT = PROJECT_ROOT / "parquet_export"


DATASET_CARD_CONTENT = """---
license: cc-by-4.0
task_categories:
  - other
tags:
  - eeg
  - bci
  - motor-imagery
  - neuroscience
  - brain-computer-interface
pretty_name: ProjectCerebro Cho 2017 EEG Motor Imagery
size_categories:
  - 10K<n<100K
---

# ProjectCerebro — Cho 2017 EEG Motor Imagery Dataset (BCI_Data_new)

## Dataset Description

Preprocessed EEG motor imagery epochs from the **Cho 2017** dataset
([GigaDB 100295](http://gigadb.org/dataset/100295)),
processed through the ProjectCerebro pipeline.

**Original paper:**
> Cho, H., et al. (2017). EEG datasets for motor imagery brain–computer interface.
> *GigaScience*, 6(7), 1–8. https://doi.org/10.1093/gigascience/gix034

## Dataset Structure

Each row is one 4-second EEG epoch with these columns:

| Column | Type | Description |
|--------|------|-------------|
| epoch_id | string | Unique identifier |
| dataset | string | Always `cho2017` |
| subject_id | string | Subject ID (s01–s52) |
| label_code | int | 0=left, 1=right, 2=rest |
| label_name | string | `left`, `right`, or `rest` |
| features | float32[] | Flattened EEG (5 ch × 512 samples = 2560 floats) |
| n_channels | int | 5 |
| n_samples | int | 512 (4s @ 128Hz) |
| channel_names | string[] | [FZ, C3, CZ, C4, PZ] |
| sampling_rate_hz | float | 128.0 |
| filter_version | string | `bp_8_30_v1` or `bp_4_38_v1` |
| preprocessing_version | string | `v1.1.0` |

## Preprocessing Pipeline

1. **Stage 1** — ICA artifact removal (infomax + ICLabel)
   - 20 ICA components fitted on 1–100 Hz broadband signal
   - Eye blink, muscle, line noise components excluded (>0.8 confidence)
   - Output: cleaned `.fif` files

2. **Stage 2** — Epoch extraction & standardization
   - 5 common channels selected: FZ, C3, CZ, C4, PZ
   - Bandpass filter: 8–30 Hz (primary) or 4–38 Hz (ablation)
   - Resampled to 128 Hz
   - Epochs: -1s to +3s around trial onset → 512 samples
   - Baseline correction: subtract mean of pre-stimulus period

## Files

| File | Description |
|------|-------------|
| `bp8_30/*.parquet` | Primary filter (8–30 Hz) — motor imagery band |
| `bp4_38/*.parquet` | Ablation filter (4–38 Hz) — broader band |

## Source

GitHub: [JANARDHANAREDDYMS/projectcerebro](https://github.com/JANARDHANAREDDYMS/projectcerebro/tree/janardhan)
"""


def create_dataset_card(api: HfApi, repo_id: str) -> None:
    print("Creating dataset card...")
    card = DatasetCard(DATASET_CARD_CONTENT)
    card.push_to_hub(repo_id, token=HF_TOKEN)
    print("Dataset card uploaded.")


def upload_parquet_folder(
    api: HfApi,
    repo_id: str,
    local_path: Path,
    hf_folder: str,
) -> None:
    parquet_files = sorted(local_path.rglob("*.parquet"))
    if not parquet_files:
        print(f"  No parquet files found in {local_path}")
        return

    print(f"  Uploading {len(parquet_files)} parquet file(s) to {hf_folder}/")
    for pf in parquet_files:
        rel = pf.relative_to(local_path)
        path_in_repo = f"{hf_folder}/{rel}"
        print(f"    -> {path_in_repo}")
        api.upload_file(
            path_or_fileobj=str(pf),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            token=HF_TOKEN,
        )
    print(f"  Done uploading {hf_folder}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", default=HF_TOKEN)
    parser.add_argument("--repo",  default=HF_REPO_ID)
    args = parser.parse_args()

    api = HfApi(token=args.token)
    print(f"Logged in as: {api.whoami()['name']}")
    print(f"Target repo:  {args.repo}\n")

    api.create_repo(
        repo_id=args.repo,
        repo_type="dataset",
        private=False,
        exist_ok=True,
    )

    create_dataset_card(api, args.repo)

    bp8_30_path = PARQUET_ROOT / "cho2017_epochs_ch5_sr128_bp8_30"
    if bp8_30_path.exists():
        print("\nUploading bp8_30 (8-30 Hz) parquet files...")
        upload_parquet_folder(api, args.repo, bp8_30_path, "bp8_30")
    else:
        print(f"bp8_30 path not found: {bp8_30_path}")

    bp4_38_path = PARQUET_ROOT / "cho2017_epochs_ch5_sr128_bp4_38"
    if bp4_38_path.exists():
        print("\nUploading bp4_38 (4-38 Hz) parquet files...")
        upload_parquet_folder(api, args.repo, bp4_38_path, "bp4_38")
    else:
        print(f"bp4_38 path not found: {bp4_38_path}")

    print(f"\nUpload complete!")
    print(f"  Dataset URL: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
