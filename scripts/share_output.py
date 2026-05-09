"""
Share the parquet output files.

Options:
  1. transfer.sh  — public anonymous upload, 14-day link (no auth needed)
  2. HuggingFace  — requires HF token (set HF_TOKEN env var)

Usage:
  # Anonymous transfer.sh upload (easiest):
  python scripts/share_output.py --transfer

  # Hugging Face dataset (persistent, requires token):
  python scripts/share_output.py --hf --token YOUR_HF_TOKEN --repo yourname/projectcerebro-eeg
"""

from __future__ import annotations
import argparse
import os
import subprocess
from pathlib import Path

PARQUET_ROOT = Path(__file__).resolve().parent.parent / "parquet_output"


def upload_catbox(file_path: Path) -> str:
    """Upload to catbox.moe (anonymous, permanent) and return download URL."""
    size_mb = file_path.stat().st_size / 1024 / 1024
    print(f"Uploading {file_path.name} ({size_mb:.1f} MB) to catbox.moe...")
    result = subprocess.run(
        ["curl", "-F", "reqtype=fileupload",
         "-F", f"fileToUpload=@{file_path}",
         "https://catbox.moe/user/api.php"],
        capture_output=True, text=True, timeout=3600
    )
    if result.returncode != 0 or not result.stdout.startswith("http"):
        raise RuntimeError(f"Upload failed: {result.stdout} {result.stderr}")
    url = result.stdout.strip()
    print(f"  Download URL: {url}")
    return url


def upload_huggingface(file_path: Path, repo_id: str, token: str) -> str:
    """Upload to Hugging Face dataset."""
    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=token)
    try:
        create_repo(repo_id, repo_type="dataset", token=token, exist_ok=True)
        print(f"  Uploading to HuggingFace: {repo_id}")
    except Exception as e:
        print(f"  Repo may already exist: {e}")

    api.upload_file(
        path_or_fileobj=str(file_path),
        path_in_repo=file_path.name,
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
    )
    url = f"https://huggingface.co/datasets/{repo_id}/blob/main/{file_path.name}"
    print(f"  HuggingFace URL: {url}")
    return url


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transfer", action="store_true", help="Upload to catbox.moe")
    parser.add_argument("--hf",       action="store_true", help="Upload to HuggingFace")
    parser.add_argument("--token",    default=os.environ.get("HF_TOKEN"), help="HuggingFace token")
    parser.add_argument("--repo",     default="projectcerebro/eeg-motor-imagery", help="HF repo id")
    args = parser.parse_args()

    parquet_files = list(PARQUET_ROOT.glob("*.parquet"))
    if not parquet_files:
        print(f"No parquet files found in {PARQUET_ROOT}")
        return

    print(f"Found {len(parquet_files)} parquet file(s):")
    for f in parquet_files:
        print(f"  {f.name}  ({f.stat().st_size / 1024 / 1024:.1f} MB)")

    for fpath in parquet_files:
        if args.transfer:
            url = upload_catbox(fpath)
            print(f"\nShare this URL: {url}")
            print("(Permanent — catbox.moe)")

        if args.hf:
            if not args.token:
                print("ERROR: --token required for HuggingFace upload")
                return
            url = upload_huggingface(fpath, args.repo, args.token)
            print(f"\nShare this URL: {url}")


if __name__ == "__main__":
    main()
