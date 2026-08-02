"""
Google Drive Uploader Utility.

Handles authentication and automated uploading of exported video files to Google Drive.

Supports three modes:
1. Local OAuth (credentials.json + token.json) - for manual runs.
2. Service account (service_account.json) - for GitHub Actions.
3. Service account from environment variables - for GitHub Actions secrets.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class GoogleDriveUploader:
    """Manages file uploads to Google Drive."""

    def __init__(self, credentials_path: Optional[str] = None):
        # Determine repository root dynamically (go up 3 levels from python/utils/google_drive_uploader.py)
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.credentials_path = credentials_path or os.environ.get(
            "GOOGLE_DRIVE_CREDENTIALS_PATH", os.path.join(repo_root, "credentials.json")
        )

    def _get_service_account_creds(self):
        """Build service account credentials from file or environment."""
        from google.oauth2 import service_account
        from googleapiclient.discovery import build

        # Mode 1: SA JSON from env var (base64 or raw JSON)
        sa_json_b64 = os.environ.get("GOOGLE_DRIVE_SERVICE_ACCOUNT_B64")
        sa_json_raw = os.environ.get("GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON")
        sa_file = os.environ.get("GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE", "service_account.json")

        if sa_json_b64:
            sa_json = base64.b64decode(sa_json_b64).decode("utf-8")
            info = json.loads(sa_json)
        elif sa_json_raw:
            info = json.loads(sa_json_raw)
        elif os.path.exists(sa_file):
            with open(sa_file) as f:
                info = json.load(f)
        else:
            raise FileNotFoundError(
                "No service account credentials found. Set "
                "GOOGLE_DRIVE_SERVICE_ACCOUNT_B64 or provide service_account.json"
            )

        scopes = ["https://www.googleapis.com/auth/drive.file"]
        creds = service_account.Credentials.from_service_account_info(info, scopes=scopes)
        return build("drive", "v3", credentials=creds)

    def _get_oauth_creds(self):
        """Build OAuth credentials from credentials.json + token.json."""
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build

        SCOPES = ["https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]
        creds = None
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        token_path = os.path.join(repo_root, "token.json")
        if os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(token_path, SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception as re:
                    logger.error(f"Failed to refresh Google Drive token: {re}")
                    if os.environ.get("GITHUB_ACTIONS"):
                        raise RuntimeError("Headless environment: Cannot refresh expired Google Drive token. Please re-run token generator.")
            else:
                if os.environ.get("GITHUB_ACTIONS"):
                    raise RuntimeError("Headless environment: Missing Google Drive credentials token.json.")
                flow = InstalledAppFlow.from_client_secrets_file(self.credentials_path, SCOPES)
                creds = flow.run_local_server(port=0)
            # Only write back if it's local run or token refreshed successfully
            if creds and creds.valid:
                with open(token_path, "w") as token:
                    token.write(creds.to_json())
        return build("drive", "v3", credentials=creds)

    def _get_service(self):
        # Prefer service account (headless / CI)
        if (
            os.environ.get("GOOGLE_DRIVE_SERVICE_ACCOUNT_B64")
            or os.environ.get("GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON")
            or os.path.exists(os.environ.get("GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE", "service_account.json"))
        ):
            return self._get_service_account_creds()
        # Fall back to OAuth local creds
        return self._get_oauth_creds()

    def upload_file(self, file_path: str, folder_id: Optional[str] = None) -> Optional[str]:
        """
        Uploads a video to Google Drive.

        Args:
            file_path: Absolute path to the file to upload.
            folder_id: Optional destination Google Drive folder ID.

        Returns:
            The sharing/view link of the uploaded file if successful, or None.
        """
        if not os.path.exists(file_path):
            logger.error(f"Upload file not found: {file_path}")
            return None

        logger.info(f"Initiating Google Drive upload for {file_path}")

        # If no credentials at all, simulate and return a local path marker
        has_credentials = any([
            os.environ.get("GOOGLE_DRIVE_SERVICE_ACCOUNT_B64"),
            os.environ.get("GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON"),
            os.path.exists(os.environ.get("GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE", "service_account.json")),
            os.path.exists(self.credentials_path),
        ])

        if not has_credentials:
            logger.warning(
                "No Google Drive credentials found. Simulating upload and saving locally."
            )
            simulated_link = (
                f"https://drive.google.com/open?id=simulated_{os.path.basename(file_path)}"
            )
            logger.info(f"Simulated Upload Link: {simulated_link}")
            return simulated_link

        try:
            service = self._get_service()

            file_metadata: Dict[str, Any] = {"name": os.path.basename(file_path)}
            if folder_id:
                file_metadata["parents"] = [folder_id]

            from googleapiclient.http import MediaFileUpload

            media = MediaFileUpload(file_path, resumable=True, chunksize=50 * 1024 * 1024)
            uploaded = service.files().create(
                body=file_metadata,
                media_body=media,
                fields="id, webViewLink",
            ).execute()

            # Make file viewable by anyone with the link
            try:
                service.permissions().create(
                    fileId=uploaded["id"],
                    body={"type": "anyone", "role": "reader"},
                ).execute()
            except Exception as perm_err:
                logger.warning(f"Could not set public permission: {perm_err}")

            web_link = uploaded.get("webViewLink")
            logger.info(
                f"Upload successful. File ID: {uploaded.get('id')}, Link: {web_link}"
            )
            return web_link

        except Exception as e:
            logger.error(f"Failed to upload to Google Drive: {e}")
            return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Upload a file to Google Drive")
    parser.add_argument("file", help="Path to file to upload")
    parser.add_argument("--folder-id", default=None, help="Drive folder ID")
    args = parser.parse_args()

    uploader = GoogleDriveUploader()
    link = uploader.upload_file(args.file, args.folder_id)
    print(link or "Upload failed")
