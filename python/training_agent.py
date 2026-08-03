"""
Training Agent for Video Editing Engine.

Downloads top-performing viral videos, decodes their pacing, cut frequency, 
motion climax points, and audio SFX timing profiles, and stores them in the Memory database.
"""

import os
import cv2
import json
import logging
import subprocess
import tempfile
import numpy as np
from typing import Dict, List, Any, Tuple

logger = logging.getLogger("training_agent")

class TrainingAgent:
    """Agent that analyzes viral videos to learn/decode transition pacing and SFX timings."""

    def __init__(self, workspace_root: str):
        self.workspace_root = workspace_root
        self.memory_path = os.path.join(workspace_root, "memory", "project_memory.json")

    def download_viral_videos(self, query: str, download_dir: str, limit: int = 5) -> List[str]:
        """Downloads the top `limit` most-viewed videos matching the query."""
        os.makedirs(download_dir, exist_ok=True)
        logger.info(f"Training Agent: Searching YouTube for top {limit} videos on '{query}'...")
        
        out_template = os.path.join(download_dir, "viral_%(autonumber)d.%(ext)s")
        cmd = [
            "yt-dlp",
            "--no-check-certificates",
            "-f", "best[height<=480]/mp4",  # Lower resolution for fast analysis
            "-o", out_template,
            "--max-downloads", str(limit),
            "--no-playlist",
            "--quiet",
            f"ytsearch{limit * 3}:{query}"
        ]
        try:
            # yt-dlp returns 101 when aborting due to max-downloads, which is normal
            subprocess.run(cmd, check=False, timeout=300)
            paths = [
                os.path.join(download_dir, f)
                for f in os.listdir(download_dir)
                if f.startswith("viral_") and f.endswith(".mp4")
            ]
            return sorted(paths)
        except Exception as e:
            logger.error(f"Failed to download training videos: {e}")
            return []

    def download_url_videos(self, urls: List[str], download_dir: str) -> List[str]:
        """Downloads videos from specific URLs."""
        os.makedirs(download_dir, exist_ok=True)
        paths = []
        for idx, url in enumerate(urls):
            logger.info(f"Training Agent: Downloading reference URL {idx+1}: {url}")
            out_template = os.path.join(download_dir, f"viral_ref_{idx+1}.%(ext)s")
            cmd = [
                "yt-dlp",
                "--no-check-certificates",
                "-f", "best[height<=480]/mp4",
                "-o", out_template,
                url
            ]
            try:
                subprocess.run(cmd, check=False, timeout=120)
            except Exception as e:
                logger.error(f"Failed to download reference video from {url}: {e}")
                
        paths = [
            os.path.join(download_dir, f)
            for f in os.listdir(download_dir)
            if f.startswith("viral_ref_") and f.endswith(".mp4")
        ]
        return sorted(paths)

    def analyze_single_video(self, video_path: str) -> Dict[str, Any]:
        """Analyzes scene cuts, motion peaks, and audio spikes of a single video."""
        logger.info(f"Analyzing style of: {os.path.basename(video_path)}")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps

        cuts = []
        motions = []
        prev_gray = None
        
        step = max(1, int(fps * 0.1)) # Sample every 100ms
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
                t = f_idx / fps
                motions.append((t, mean_diff))
                
                if mean_diff > 35.0:
                    cuts.append(t)
            prev_gray = gray_small
        cap.release()

        cut_spacings = []
        last_t = 0.0
        for cut in cuts:
            cut_spacings.append(cut - last_t)
            last_t = cut
        if duration - last_t > 0.5:
            cut_spacings.append(duration - last_t)
            
        avg_pacing = np.mean(cut_spacings) if cut_spacings else 8.0

        # Audio Peak analysis
        audio_temp = video_path.replace(".mp4", "_temp.wav")
        avg_sfx_delay_before_cut = 0.3 # Default baseline
        
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", video_path, "-ac", "1", "-ar", "8000", "-f", "wav", audio_temp
        ]
        try:
            subprocess.run(cmd, check=True, timeout=60)
            if os.path.exists(audio_temp):
                with open(audio_temp, "rb") as f:
                    f.seek(44)
                    audio_data = np.frombuffer(f.read(), dtype=np.int16)
                
                chunk = 800 # 100ms at 8kHz
                peaks = []
                for i in range(len(audio_data) // chunk):
                    val = np.max(np.abs(audio_data[i * chunk : (i+1) * chunk]))
                    if val > 12000:
                        peaks.append(i * 0.1)
                
                pre_cut_delays = []
                for cut in cuts:
                    matching_peaks = [p for p in peaks if cut - 1.2 <= p <= cut]
                    if matching_peaks:
                        closest_peak = max(matching_peaks)
                        pre_cut_delays.append(cut - closest_peak)
                if pre_cut_delays:
                    avg_sfx_delay_before_cut = float(np.mean(pre_cut_delays))
                
                os.remove(audio_temp)
        except Exception as ae:
            logger.warning(f"Audio training analysis failed: {ae}")

        return {
            "duration": duration,
            "cuts_count": len(cuts),
            "average_pacing": float(avg_pacing),
            "pre_cut_sfx_delay": float(avg_sfx_delay_before_cut)
        }

    def train_style_from_query(self, query: str, style_name: str = "Viral Trained"):
        """Downloads, decodes, and records style profile to the Memory database."""
        temp_dir = tempfile.mkdtemp(prefix="training_")
        downloaded = self.download_viral_videos(query, temp_dir, limit=3)
        
        if not downloaded:
            logger.warning("No videos downloaded for training.")
            return False
            
        pacings = []
        delays = []
        
        for vid in downloaded:
            try:
                stats = self.analyze_single_video(vid)
                pacings.append(stats["average_pacing"])
                delays.append(stats["pre_cut_sfx_delay"])
            except Exception as e:
                logger.error(f"Failed to analyze video {vid}: {e}")
                
        if not pacings:
            return False
            
        mean_pacing = float(np.mean(pacings))
        mean_delay = float(np.mean(delays))
        
        return self._save_style_profile(style_name, mean_pacing, mean_delay)

    def train_style_from_urls(self, urls: List[str], style_name: str = "Viral Custom Style") -> bool:
        """Downloads, decodes, and records style profile from a list of video URLs."""
        temp_dir = tempfile.mkdtemp(prefix="training_urls_")
        downloaded = self.download_url_videos(urls, temp_dir)
        
        if not downloaded:
            logger.warning("No reference videos downloaded for URL-based training.")
            return False
            
        pacings = []
        delays = []
        
        for vid in downloaded:
            try:
                stats = self.analyze_single_video(vid)
                pacings.append(stats["average_pacing"])
                delays.append(stats["pre_cut_sfx_delay"])
            except Exception as e:
                logger.error(f"Failed to analyze reference video {vid}: {e}")
                
        if not pacings:
            return False
            
        mean_pacing = float(np.mean(pacings))
        mean_delay = float(np.mean(delays))
        
        return self._save_style_profile(style_name, mean_pacing, mean_delay)

    def _save_style_profile(self, style_name: str, mean_pacing: float, mean_delay: float) -> bool:
        """Saves style pacing and delay parameter profile to project_memory.json."""
        try:
            if os.path.exists(self.memory_path):
                with open(self.memory_path, "r", encoding="utf-8") as f:
                    memory_data = json.load(f)
            else:
                memory_data = {}
                
            style_key = style_name.lower().replace(" ", "_")
            memory_data.setdefault("style_library", {})[style_key] = {
                "name": style_name,
                "average_pacing": round(mean_pacing, 2),
                "pre_cut_whoosh_delay": round(mean_delay, 2),
                "success_count": 0,
                "priority": "High"
            }
            
            with open(self.memory_path, "w", encoding="utf-8") as f:
                json.dump(memory_data, f, indent=2)
                
            logger.info(f"SUCCESSfully trained and updated style '{style_name}' in Memory!")
            logger.info(f"  - Average Pacing: {mean_pacing:.2f}s")
            logger.info(f"  - Pre-cut SFX Delay: {mean_delay:.2f}s")
            return True
        except Exception as me:
            logger.error(f"Failed to save trained style: {me}")
            return False
