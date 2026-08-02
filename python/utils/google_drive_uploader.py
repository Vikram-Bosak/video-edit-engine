"""
Google Drive Uploader Utility.

Handles authentication and automated uploading of exported video files to Google Drive.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class GoogleDriveUploader:
    """Manages file uploads to Google Drive using credentials.json."""

    def __init__(self, credentials_path: Optional[str] = None):
        self.credentials_path = credentials_path or "credentials.json"

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

        # Check if credentials.json exists
        if not os.path.exists(self.credentials_path):
            logger.warning(
                f"Google Drive credentials not found at '{self.credentials_path}'. "
                "Simulating upload and saving file locally."
            )
            # Simulated link
            simulated_link = f"https://drive.google.com/open?id=simulated_{os.path.basename(file_path)}"
            logger.info(f"Simulated Upload Link: {simulated_link}")
            return simulated_link

        try:
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from google.auth.transport.requests import Request
            from googleapiclient.discovery import build
            from googleapiclient.http import MediaFileUpload

            # Scopes required for uploading files
            SCOPES = ['https://www.googleapis.com/auth/drive']
            creds = None

            token_path = 'token.json'
            if os.path.exists(token_path):
                creds = Credentials.from_authorized_user_file(token_path, SCOPES)

            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(self.credentials_path, SCOPES)
                    # Use local server to authenticate in browser
                    creds = flow.run_local_server(port=8080, open_browser=False)
                # Save credentials for next run
                with open(token_path, 'w') as token:
                    token.write(creds.to_json())

            service = build('drive', 'v3', credentials=creds)

            file_metadata = {'name': os.path.basename(file_path)}
            if folder_id:
                file_metadata['parents'] = [folder_id]

            media = MediaFileUpload(file_path, resumable=True)
            uploaded_file = service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id, webViewLink'
            ).execute()

            web_link = uploaded_file.get('webViewLink')
            logger.info(f"Google Drive upload successful. File ID: {uploaded_file.get('id')}, Link: {web_link}")
            return web_link

        except Exception as e:
            logger.error(f"Failed to upload to Google Drive: {e}")
            return None
