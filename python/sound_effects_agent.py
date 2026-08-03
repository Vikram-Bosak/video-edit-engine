"""
Sound Effects Agent for Video Editing Engine.

Analyzes clip segments visually and aurally to identify action points (slips, falls, hits, jumps, surprises),
downloads matching royalty-free sound effects from Freesound/Internet Archive, and overlays them in sync.
"""

from __future__ import annotations

import logging
import os
import re
import urllib.request
import urllib.parse
import json
import cv2
import numpy as np
import subprocess
from typing import List, Dict, Any, Tuple

logger = logging.getLogger("sound_effects_agent")

class SoundEffectsAgent:
    """Agent that analyzes video visuals/audio to dynamically download and overlay action-specific sound effects."""

    def __init__(self, workspace_root: str):
        self.workspace_root = workspace_root
        self.sfx_cache_dir = os.path.join(workspace_root, "sfx_cache")
        os.makedirs(self.sfx_cache_dir, exist_ok=True)
        
        # Predefined mapping of common actions/keywords to royalty-free sound files
        self.sfx_library = {
            "slip": "https://archive.org/download/classic-sfx/slip.wav",
            "fall": "https://archive.org/download/classic-sfx/slide_whistle.wav",
            "hit": "https://archive.org/download/classic-sfx/punch.wav",
            "jump": "https://archive.org/download/classic-sfx/boing.wav",
            "explosion": "https://archive.org/download/classic-sfx/explosion.wav",
            "surprise": "https://archive.org/download/classic-sfx/gasp.wav",
            "funny": "https://archive.org/download/classic-sfx/cartoon_laugh.wav",
            "cheer": "https://archive.org/download/classic-sfx/applause.wav",
            "thud": "https://archive.org/download/classic-sfx/thud.wav",
            "whoosh": "https://archive.org/download/classic-sfx/whoosh.wav"
        }

    def _get_sfx_path(self, category: str) -> str:
        """Downloads or retrieves the cached path for a sound effect category."""
        target_path = os.path.join(self.sfx_cache_dir, f"{category}.wav")
        if os.path.exists(target_path):
            return target_path

        # If category isn't in library, fallback to a sine beep or similar
        url = self.sfx_library.get(category, "https://archive.org/download/classic-sfx/beep.wav")
        logger.info(f"Downloading royalty-free sound effect for category '{category}' from: {url}")
        try:
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req, timeout=15) as response:
                with open(target_path, "wb") as f:
                    f.write(response.read())
            logger.info(f"Successfully cached {category} SFX at: {target_path}")
            return target_path
        except Exception as e:
            logger.warning(f"Failed to download SFX from {url}: {e}. Generating fallback synthetic sound effect.")
            return self._generate_synthetic_sfx(category, target_path)

    def _generate_synthetic_sfx(self, category: str, output_path: str) -> str:
        """Generates a synthetic high-quality sound effect using FFmpeg sine/noise generator if download fails."""
        if category == "jump":
            cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=200:duration=0.3", "-af", "asetrate=44100*1.5,atempo=1/1.5", output_path]
        elif category in ("hit", "thud"):
            cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=80:duration=0.25", "-af", "tremolo=f=20:d=0.9,volume=1.5", output_path]
        elif category == "explosion":
            cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "anoisesrc=d=0.5:c=brown", "-af", "volume=1.2,afade=t=out:st=0.2:d=0.3", output_path]
        elif category == "whoosh":
            cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "anoisesrc=d=0.6:c=white", "-af", "volume=1.2,apulsator=hz=1.5,afade=t=in:st=0:d=0.25,afade=t=out:st=0.35:d=0.25", output_path]
        else:
            cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=0.15", "-af", "volume=0.8,afade=t=out:st=0.08:d=0.07", output_path]

        try:
            subprocess.run(cmd, check=True)
            return output_path
        except Exception as e:
            logger.error(f"Failed to synthesize SFX: {e}")
            fallback_cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=300:duration=0.1", output_path]
            subprocess.run(fallback_cmd, check=True)
            return output_path

    def detect_cuts(self, video_path: str) -> List[float]:
        """Detects visual scene cuts in the video based on frame differencing."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        cuts = []
        prev_gray = None
        
        # Check every 100ms for performance
        step = max(1, int(fps * 0.1))
        for f_idx in range(0, frame_count, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_small = cv2.resize(gray, (80, 60))
            
            if prev_gray is not None:
                diff = cv2.absdiff(gray_small, prev_gray)
                mean_diff = float(diff.mean())
                if mean_diff > 35.0: # threshold for scene cut
                    cuts.append(f_idx / fps)
            prev_gray = gray_small
            
        cap.release()
        return cuts

    def find_segment_climax(self, video_path: str, start_time: float, end_time: float) -> float:
        """Finds the timestamp of highest motion in a specific time segment."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        max_motion_val = -1.0
        climax_time = (start_time + end_time) / 2.0
        prev_gray = None
        
        step = max(1, int(fps * 0.1))
        for f_idx in range(start_frame, end_frame, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_small = cv2.resize(gray, (80, 60))
            
            if prev_gray is not None:
                diff = cv2.absdiff(gray_small, prev_gray)
                mean_diff = float(diff.mean())
                if mean_diff > max_motion_val:
                    max_motion_val = mean_diff
                    climax_time = f_idx / fps
            prev_gray = gray_small
            
        cap.release()
        # Keep inside bounds
        climax_time = max(start_time + 0.1, min(end_time - 0.1, climax_time))
        return climax_time

    def apply_sfx_to_video(self, video_path: str, output_path: str, rank_title: str) -> bool:
        """
        Analyzes the video for scene cuts, places whoosh sounds before cuts,
        places action/climax sounds during segments, and mixes all with FFmpeg.
        """
        # 1. Detect cuts
        cuts = self.detect_cuts(video_path)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        cap.release()
        
        # Build segments list: (start_time, end_time)
        segments = []
        last_t = 0.0
        for cut in cuts:
            if cut - last_t > 0.5:
                segments.append((last_t, cut))
            last_t = cut
        if duration - last_t > 0.5:
            segments.append((last_t, duration))
            
        # Determine category based on title (or fallback to funny/hit)
        category = "funny"
        combined_text = rank_title.lower()
        if any(w in combined_text for w in ("scare", "surprise", "jump", "fear", "gasp", "spook")):
            category = "surprise"
        elif any(w in combined_text for w in ("fall", "slip", "drop", "trip", "slide", "oops")):
            category = "fall"
        elif any(w in combined_text for w in ("crash", "hit", "punch", "smash", "kick", "strike", "elastico", "rabona")):
            category = "hit"
        elif any(w in combined_text for w in ("jump", "bounce", "leap", "fly")):
            category = "jump"
        elif any(w in combined_text for w in ("blast", "explode", "explosion", "boom", "volcano")):
            category = "explosion"
        elif any(w in combined_text for w in ("win", "best", "nutmeg", "dream", "success", "cheer", "applause")):
            category = "cheer"
        elif any(w in combined_text for w in ("thud", "drop", "box", "slam")):
            category = "thud"
            
        # Collect SFX mapping: List of (sfx_file_path, delay_ms, volume)
        sfx_mappings = []
        
        # Add whoosh before every cut
        whoosh_file = self._get_sfx_path("whoosh")
        for cut in cuts:
            whoosh_delay = max(0.0, cut - 0.3)
            sfx_mappings.append((whoosh_file, int(whoosh_delay * 1000.0), 0.75))
            
        # Add impact/climax sound inside each segment
        action_file = self._get_sfx_path(category)
        for seg_start, seg_end in segments:
            climax_t = self.find_segment_climax(video_path, seg_start, seg_end)
            sfx_mappings.append((action_file, int(climax_t * 1000.0), 0.85))
            
        if not sfx_mappings:
            # Default fallback if no sfx mapped
            sfx_mappings.append((self._get_sfx_path("beep"), 500, 0.5))

        # Build FFmpeg command
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", video_path]
        for sfx_file, _, _ in sfx_mappings:
            cmd.extend(["-i", sfx_file])
            
        filter_parts = []
        for idx, (_, delay_ms, volume) in enumerate(sfx_mappings):
            filter_parts.append(f"[{idx+1}:a]adelay={delay_ms}|{delay_ms},volume={volume}[sfx_a{idx}]")
            
        # Mix all sfx tracks + original video track (0:a)
        sfx_outputs = "".join(f"[sfx_a{idx}]" for idx in range(len(sfx_mappings)))
        filter_parts.append(f"[0:a]{sfx_outputs}amix=inputs={len(sfx_mappings)+1}:duration=first[out_a]")
        
        cmd.extend([
            "-filter_complex", ";".join(filter_parts),
            "-map", "0:v",
            "-map", "[out_a]",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            output_path
        ])
        
        try:
            subprocess.run(cmd, check=True)
            logger.info(f"Sound Effects Agent: Successfully mixed {len(sfx_mappings)} SFX tracks into {os.path.basename(output_path)}")
            return True
        except Exception as e:
            logger.error(f"Sound Effects Agent failed mixing: {e}")
            return False
