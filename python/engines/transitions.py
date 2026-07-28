"""Transition engine for applying video transitions between clips."""

from __future__ import annotations

import os
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from python.core.config import TransitionType
except ImportError:
    class TransitionType(Enum):
        FADE = "fade"
        FLASH = "flash"
        WHIP = "whip"
        BLUR = "blur"
        SLIDE = "slide"
        PUSH = "push"
        ZOOM = "zoom"
        MASK_REVEAL = "mask_reveal"
        GLITCH = "glitch"
        LIGHT_LEAK = "light_leak"
        MOTION_BLUR = "motion_blur"
        FILM_BURN = "film_burn"
        CAMERA_MOVEMENT = "camera_movement"
        WIPE_LEFT = "wipe_left"
        WIPE_RIGHT = "wipe_right"
        WIPE_UP = "wipe_up"
        WIPE_DOWN = "wipe_down"


@dataclass
class TransitionConfig:
    transition_type: TransitionType
    duration: float = 0.5
    direction: str = "left"
    intensity: float = 1.0


@dataclass
class TransitionResult:
    output_path: str
    success: bool
    duration: float = 0.0
    error: str = ""


class TransitionEngine:
    """Engine for applying professional video transitions between clips."""

    def __init__(self, temp_dir: Optional[str] = None, ffmpeg_path: str = "ffmpeg"):
        self.ffmpeg_path = ffmpeg_path
        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix="transitions_")
        os.makedirs(self.temp_dir, exist_ok=True)

    def _run_ffmpeg(self, args: List[str], desc: str = "ffmpeg") -> subprocess.CompletedProcess:
        cmd = [self.ffmpeg_path, "-y", "-hide_banner", "-loglevel", "error"] + args
        return subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    def _get_duration(self, path: str) -> float:
        cmd = [
            "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        try:
            return float(result.stdout.strip())
        except ValueError:
            return 0.0

    def _get_resolution(self, path: str) -> Tuple[int, int]:
        cmd = [
            "ffprobe", "-v", "quiet", "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=s=x:p=0", path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        try:
            w, h = result.stdout.strip().split("x")
            return int(w), int(h)
        except Exception:
            return 1080, 1920

    def _xfade_filter(self, transition: str, offset: float, duration: float) -> str:
        return f"xfade=transition={transition}:duration={duration}:offset={offset}"

    def apply_transition(
        self,
        clip_a_path: str,
        clip_b_path: str,
        transition_type: TransitionType,
        duration: float = 0.5,
        output_path: Optional[str] = None,
        audio_crossfade: bool = True,
    ) -> TransitionResult:
        if not output_path:
            output_path = os.path.join(self.temp_dir, f"transition_{uuid.uuid4().hex[:8]}.mp4")

        dur_a = self._get_duration(clip_a_path)
        offset = max(0, dur_a - duration)
        w, h = self._get_resolution(clip_a_path)

        transition_map = {
            TransitionType.FADE: "fade",
            TransitionType.FLASH: "fadeblack",
            TransitionType.WHIP: "wipeleft",
            TransitionType.BLUR: "fadeblack",
            TransitionType.SLIDE: "slideleft",
            TransitionType.PUSH: "pushleft",
            TransitionType.ZOOM: "circlecrop",
            TransitionType.MASK_REveal: "radial",
            TransitionType.GLITCH: "fadeblack",
            TransitionType.LIGHT_LEAK: "fade",
            TransitionType.MOTION_BLUR: "fadeblack",
            TransitionType.FILM_BURN: "dissolve",
            TransitionType.CAMERA_MOVEMENT: "smoothleft",
            TransitionType.WIPE_LEFT: "wipeleft",
            TransitionType.WIPE_RIGHT: "wiperight",
            TransitionType.WIPE_UP: "wipeup",
            TransitionType.WIPE_DOWN: "wipedown",
        }

        xfade_name = transition_map.get(transition_type, "fade")

        if transition_type == TransitionType.FLASH:
            flash_filter = f"[0:v]fade=t=out:st={offset}:d={duration}[v0];[1:v]fade=t=in:st=0:d={duration}[v1];[v0][v1]xfade=transition=fade:duration={duration}:offset={offset}"
        elif transition_type == TransitionType.BLUR:
            flash_filter = (
                f"[0:v]boxblur=20:20:enable='between(t,{offset},{offset + duration})',fade=t=out:st={offset}:d={duration}[v0];"
                f"[1:v]fade=t=in:st=0:d={duration},boxblur=20:20:enable='between(t,0,{duration})'[v1];"
                f"[v0][v1]xfade=transition=fade:duration={duration}:offset={offset}"
            )
        elif transition_type == TransitionType.MOTION_BLUR:
            flash_filter = (
                f"[0:v]motionblur=7:7:enable='between(t,{offset},{offset + duration})',fade=t=out:st={offset}:d={duration}[v0];"
                f"[1:v]fade=t=in:st=0:d={duration}[v1];"
                f"[v0][v1]xfade=transition=fade:duration={duration}:offset={offset}"
            )
        elif transition_type == TransitionType.GLITCH:
            flash_filter = (
                f"[0:v]rgbashift=rh=5:bh=-5:enable='between(t,{offset},{offset + duration})'[v0];"
                f"[v0][1:v]xfade=transition=fade:duration={duration}:offset={offset}"
            )
        elif transition_type == TransitionType.LIGHT_LEAK:
            flash_filter = (
                f"[0:v]curves=brightness='0/0 0.5/0.7 1/1':enable='between(t,{offset},{offset + duration})'[v0];"
                f"[v0][1:v]xfade=transition=dissolve:duration={duration}:offset={offset}"
            )
        elif transition_type == TransitionType.FILM_BURN:
            flash_filter = (
                f"[0:v]colorbalance=rs=0.3:gs=-0.1:bs=-0.2:enable='between(t,{offset},{offset + duration})'[v0];"
                f"[v0][1:v]xfade=transition=dissolve:duration={duration}:offset={offset}"
            )
        else:
            flash_filter = f"[0:v][1:v]{self._xfade_filter(xfade_name, offset, duration)}"

        audio_filter = ""
        if audio_crossfade:
            audio_filter = f"[0:a][1:a]acrossfade=d={duration}:c1=tri:c2=tri[aout]"
        else:
            audio_filter = "[0:a][1:a]amix=inputs=2:duration=first[aout]"

        filter_complex = f"{flash_filter};{audio_filter}"

        args = [
            "-i", clip_a_path,
            "-i", clip_b_path,
            "-filter_complex", filter_complex,
            "-map", "[v]" if "[v]" in flash_filter else "0:v",
            "-map", "[aout]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "aac", "-b:a", "128k",
            "-pix_fmt", "yuv420p",
            output_path,
        ]

        # Fix mapping if xfade output is unnamed
        if "[v]" not in flash_filter:
            # xfade outputs to last unnamed link - it IS [v] by convention
            pass

        try:
            result = self._run_ffmpeg(args, "transition")
            if result.returncode != 0:
                return TransitionResult(
                    output_path=output_path, success=False,
                    error=result.stderr or "Unknown ffmpeg error"
                )
            return TransitionResult(
                output_path=output_path, success=True,
                duration=self._get_duration(output_path)
            )
        except subprocess.TimeoutExpired:
            return TransitionResult(output_path=output_path, success=False, error="FFmpeg timeout")
        except Exception as e:
            return TransitionResult(output_path=output_path, success=False, error=str(e))

    def chain_transitions(
        self,
        clip_paths: List[str],
        transition_types: List[TransitionType],
        transition_duration: float = 0.5,
        output_path: Optional[str] = None,
    ) -> TransitionResult:
        if not clip_paths:
            return TransitionResult(output_path="", success=False, error="No clips provided")
        if len(clip_paths) == 1:
            return TransitionResult(output_path=clip_paths[0], success=True)

        if not output_path:
            output_path = os.path.join(self.temp_dir, f"chained_{uuid.uuid4().hex[:8]}.mp4")

        current = clip_paths[0]
        for i in range(1, len(clip_paths)):
            t_type = transition_types[i - 1] if i - 1 < len(transition_types) else TransitionType.FADE
            next_out = os.path.join(self.temp_dir, f"chain_step_{i}_{uuid.uuid4().hex[:8]}.mp4") if i < len(clip_paths) - 1 else output_path

            result = self.apply_transition(
                current, clip_paths[i], t_type,
                transition_duration, next_out
            )
            if not result.success:
                return TransitionResult(output_path=output_path, success=False, error=f"Step {i} failed: {result.error}")
            current = next_out

        return TransitionResult(output_path=output_path, success=True, duration=self._get_duration(output_path))

    def get_available_transitions(self) -> List[str]:
        return [t.value for t in TransitionType]

    def estimate_output_duration(self, clip_paths: List[str], transition_duration: float) -> float:
        total = sum(self._get_duration(p) for p in clip_paths)
        if len(clip_paths) > 1:
            total -= transition_duration * (len(clip_paths) - 1)
        return total
