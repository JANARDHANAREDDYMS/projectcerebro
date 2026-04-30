"""
ProjectCerebro — Google Drive Upload Script
============================================
Uploads preprocessed data to Google Drive for team sharing.

Uploads:
  delta_lake/     -> Google Drive (primary, ~400MB)
  data_cleaned/   -> Google Drive (optional, ~3.6GB)

Setup:
  pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
  Follow OAuth setup instructions below.

Usage:
  # Upload Delta Lake only (recommended first)
  python scripts/upload_to_drive.py --delta-only

  # Upload everything
  python scripts/upload_to_drive.py --all

  # Upload specific folder
  python scripts/upload_to_drive.py --folder delta_lake
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# =========================================================
# SETUP INSTRUCTIONS
# =========================================================
# 1. Go to https://console.cloud.google.com/
# 2. Create a new project called "ProjectCerebro"
# 3. Enable Google Drive API
# 4. Create OAuth 2.0 credentials (Desktop app)
# 5. Download credentials JSON as:
#    projectcerebro/credentials.json
# 6. Run this script once - it will open browser for auth
# 7. Token saved to token.json for future runs
# =========================================================

PROJECT_ROOT   = Path(__file__).resolve().parent.parent
CREDENTIALS    = PROJECT_ROOT / "credentials.json"
TOKEN_PATH     = PROJECT_ROOT / "token.json"

SCOPES = ["https://www.googleapis.com/auth/drive.file"]

# Google Drive folder name for the team
DRIVE_FOLDER_NAME = "ProjectCerebro-Shared"

# Folders to upload
UPLOAD_TARGETS = {
    "delta_lake":   PROJECT_ROOT / "delta_lake",
    "data_cleaned": PROJECT_ROOT / "data_cleaned",
    "docs":         PROJECT_ROOT / "docs",
}


# =========================================================
# AUTH
# =========================================================

def get_drive_service():
    """Authenticate and return Google Drive service."""
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError:
        print("Missing packages. Install with:")
        print("pip install google-api-python-client "
              "google-auth-httplib2 google-auth-oauthlib")
        sys.exit(1)

    if not CREDENTIALS.exists():
        print(f"credentials.json not found at {CREDENTIALS}")
        print("\nSetup steps:")
        print("1. Go to https://console.cloud.google.com/")
        print("2. Create project 'ProjectCerebro'")
        print("3. Enable Google Drive API")
        print("4. Create OAuth 2.0 credentials (Desktop app)")
        print(f"5. Download as: {CREDENTIALS}")
        sys.exit(1)

    creds = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(
            str(TOKEN_PATH), SCOPES
        )

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(CREDENTIALS), SCOPES
            )
            creds = flow.run_local_server(port=0)
        with open(TOKEN_PATH, "w") as f:
            f.write(creds.to_json())

    service = build("drive", "v3", credentials=creds)
    return service


# =========================================================
# DRIVE HELPERS
# =========================================================

def get_or_create_folder(service, name: str, parent_id: str = None) -> str:
    """Get existing folder ID or create new one."""
    query = (
        f"name='{name}' and mimeType='application/vnd.google-apps.folder'"
        f" and trashed=false"
    )
    if parent_id:
        query += f" and '{parent_id}' in parents"

    results = service.files().list(
        q=query,
        fields="files(id, name)"
    ).execute()

    files = results.get("files", [])
    if files:
        return files[0]["id"]

    # Create folder
    metadata = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
    }
    if parent_id:
        metadata["parents"] = [parent_id]

    folder = service.files().create(
        body=metadata,
        fields="id"
    ).execute()

    return folder["id"]


def upload_file(
    service,
    local_path: Path,
    parent_id: str,
) -> str:
    """Upload a single file to Google Drive."""
    from googleapiclient.http import MediaFileUpload

    file_size = local_path.stat().st_size
    size_mb   = file_size / 1024 / 1024

    print(f"  Uploading {local_path.name} ({size_mb:.1f} MB)")

    # Check if file already exists
    query = (
        f"name='{local_path.name}' and '{parent_id}' in parents"
        f" and trashed=false"
    )
    results = service.files().list(
        q=query, fields="files(id, name)"
    ).execute()
    existing = results.get("files", [])

    media = MediaFileUpload(
        str(local_path),
        resumable=True,
    )

    if existing:
        # Update existing file
        file_id = existing[0]["id"]
        service.files().update(
            fileId=file_id,
            media_body=media,
        ).execute()
        print(f"  Updated: {local_path.name}")
    else:
        # Create new file
        metadata = {
            "name": local_path.name,
            "parents": [parent_id],
        }
        file = service.files().create(
            body=metadata,
            media_body=media,
            fields="id",
        ).execute()
        file_id = file["id"]
        print(f"  Uploaded: {local_path.name}")

    return file_id


def upload_folder(
    service,
    local_folder: Path,
    parent_drive_id: str,
    folder_name: str = None,
) -> str:
    """Recursively upload a local folder to Google Drive."""
    name = folder_name or local_folder.name

    print(f"\nCreating/finding Drive folder: {name}")
    folder_id = get_or_create_folder(service, name, parent_drive_id)

    files   = [f for f in local_folder.iterdir() if f.is_file()]
    subdirs = [d for d in local_folder.iterdir() if d.is_dir()]

    # Skip Delta Lake transaction log internals
    skip_names = {"__pycache__", ".ipynb_checkpoints"}
    subdirs = [d for d in subdirs if d.name not in skip_names]

    print(f"  {len(files)} files, {len(subdirs)} subdirectories")

    for f in sorted(files):
        try:
            upload_file(service, f, folder_id)
        except Exception as e:
            print(f"  WARNING: Failed to upload {f.name}: {e}")

    for d in sorted(subdirs):
        upload_folder(service, d, folder_id)

    return folder_id


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload ProjectCerebro data to Google Drive"
    )
    parser.add_argument(
        "--delta-only",
        action="store_true",
        help="Upload Delta Lake only (~400MB, recommended)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Upload all folders including data_cleaned (~4GB)",
    )
    parser.add_argument(
        "--folder",
        choices=list(UPLOAD_TARGETS.keys()),
        help="Upload specific folder",
    )
    parser.add_argument(
        "--docs-only",
        action="store_true",
        help="Upload docs folder only",
    )
    args = parser.parse_args()

    if not any([args.delta_only, args.all, args.folder, args.docs_only]):
        parser.print_help()
        print("\nExample: python scripts/upload_to_drive.py --delta-only")
        sys.exit(0)

    print("Authenticating with Google Drive...")
    service = get_drive_service()
    print("Authenticated successfully\n")

    # Create or find root shared folder
    print(f"Setting up Drive folder: {DRIVE_FOLDER_NAME}")
    root_id = get_or_create_folder(service, DRIVE_FOLDER_NAME)
    print(f"Root folder ID: {root_id}")
    print(f"Share this folder with teammates in Google Drive\n")

    # Determine what to upload
    to_upload = []

    if args.delta_only:
        to_upload = ["delta_lake"]
    elif args.all:
        to_upload = ["delta_lake", "data_cleaned", "docs"]
    elif args.docs_only:
        to_upload = ["docs"]
    elif args.folder:
        to_upload = [args.folder]

    # Upload
    for target in to_upload:
        local_path = UPLOAD_TARGETS[target]
        if not local_path.exists():
            print(f"WARNING: {local_path} does not exist, skipping")
            continue

        print(f"\n{'='*50}")
        print(f"Uploading: {target}")
        print(f"From: {local_path}")
        print(f"{'='*50}")

        folder_id = upload_folder(service, local_path, root_id)
        print(f"Done: {target} -> Drive folder ID: {folder_id}")

    print(f"\n{'='*50}")
    print("Upload complete!")
    print(f"\nTo share with teammates:")
    print(f"1. Open Google Drive")
    print(f"2. Find folder: {DRIVE_FOLDER_NAME}")
    print(f"3. Right click -> Share -> Add teammates emails")
    print(f"4. Give them Viewer access")
    print(f"\nTeammates download with:")
    print(f"  pip install gdown")
    print(f"  gdown --folder <shared_folder_url>")


if __name__ == "__main__":
    main()