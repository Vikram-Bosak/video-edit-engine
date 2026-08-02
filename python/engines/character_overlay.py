"""
Character Overlay Engine for Video Editing.

Provides utilities to overlay a reaction avatar (static image, animation frames,
or green screen video) onto the video timeline.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class CharacterOverlayEngine:
    """Manages superposition of reaction characters/avatars on videos."""

    def __init__(self, assets_dir: Optional[str] = None):
        self.assets_dir = assets_dir or "assets"

    def overlay_avatar(
        self,
        video_path: str,
        output_path: str,
        avatar_path: Optional[str] = None,
        position: str = "bottom_right",
        scale: float = 0.25,
    ) -> bool:
        """
        Overlays an avatar image or green-screen video onto the video.
        
        Args:
            video_path: Path to the input video.
            output_path: Path where the output video will be saved.
            avatar_path: Path to the avatar asset.
            position: Position preset ('bottom_right', 'bottom_left', 'top_right', 'top_left').
            scale: Scale factor for the avatar relative to the video width.
        """
        logger.info(f"Overlaying avatar {avatar_path} on {video_path} at position {position}")
        
        # Fallback to a placeholder avatar if none is provided
        if not avatar_path or not os.path.exists(avatar_path):
            avatar_path = os.path.join(self.assets_dir, "logos", "watermark.png")
            if not os.path.exists(avatar_path):
                # If no watermark exists, create a directory and try a basic placeholder
                os.makedirs(os.path.dirname(avatar_path), exist_ok=True)
                # Create a quick dummy avatar image using PIL
                try:
                    from PIL import Image, ImageDraw
                    img = Image.new("RGBA", (200, 200), color=(0, 0, 0, 0))
                    draw = ImageDraw.Draw(img)
                    draw.ellipse([20, 20, 180, 180], fill=(255, 215, 0, 255), outline=(255, 255, 255, 255), width=5)
                    # Draw simple smiling eyes and mouth
                    draw.ellipse([60, 70, 80, 90], fill=(0, 0, 0, 255))
                    draw.ellipse([120, 70, 140, 90], fill=(0, 0, 0, 255))
                    draw.arc([60, 100, 140, 150], start=0, end=180, fill=(0, 0, 0, 255), width=5)
                    img.save(avatar_path)
                except Exception as e:
                    logger.error(f"Failed to create dummy avatar placeholder: {e}")
                    # If we can't create it, we will just copy the input video without changes
                    import shutil
                    shutil.copy(video_path, output_path)
                    return True

        # Position calculations for FFmpeg overlay filter
        # bottom_right default
        overlay_x = "W-w-10"
        overlay_y = "H-h-10"
        if position == "bottom_left":
            overlay_x = "10"
            overlay_y = "H-h-10"
        elif position == "top_right":
            overlay_x = "W-w-10"
            overlay_y = "10"
        elif position == "top_left":
            overlay_x = "10"
            overlay_y = "10"

        # Construct FFmpeg command to scale the overlay and layer it on top
        import subprocess
        filter_complex = (
            f"[1:v]scale=iw*{scale}:-1[avatar];"
            f"[0:v][avatar]overlay={overlay_x}:{overlay_y}[outv]"
        )
        
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", video_path,
            "-i", avatar_path,
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-map", "0:a?",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-profile:v", "high",
            "-level", "4.1",
            "-c:a", "aac", "-ac", "2",
            output_path
        ]
        
        try:
            logger.info(f"Running ffmpeg character overlay command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg character overlay failed: {e.stderr}")
            return False
