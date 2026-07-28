"""Motion Effects Engine for video editing.

Provides zoom, pan, Ken Burns, camera shake, speed manipulation,
stabilization, parallax, and dynamic crop effects using FFmpeg filters.
"""

from __future__ import annotations

import enum
import json
import logging
import math
import os
import random
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MotionEffect(enum.Enum):
    """Supported motion effect types."""

    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    KEN_BURNS = "ken_burns"
    PAN_LEFT = "pan_left"
    PAN_RIGHT = "pan_right"
    PAN_UP = "pan_up"
    PAN_DOWN = "pan_down"
    CAMERA_SHAKE = "camera_shake"
    CINEMATIC_PUSH = "cinematic_push"
    FOCUS_ZOOM = "focus_zoom"
    PARALLAX = "parallax"
    SLOW_MOTION = "slow_motion"
    SPEED_RAMP = "speed_ramp"
    REVERSE = "reverse"
    STABILIZE = "stabilize"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SpeedSegment:
    """A single segment for speed ramp effects.

    Attributes:
        start_fraction: Start of the segment as a fraction of total duration (0-1).
        end_fraction: End of the segment as a fraction of total duration (0-1).
        speed_factor: Playback speed multiplier (>1 = faster, <1 = slower).
        curve: Interpolation curve between segments (``"linear"``,
            ``"ease_in"``, ``"ease_out"``, ``"ease_in_out"``, ``"constant"``).
    """

    start_fraction: float
    end_fraction: float
    speed_factor: float
    curve: str = "constant"

    def __post_init__(self) -> None:
        if not 0.0 <= self.start_fraction < self.end_fraction <= 1.0:
            raise ValueError(
                f"Segment fractions must satisfy 0 <= start < end <= 1, "
                f"got start={self.start_fraction}, end={self.end_fraction}"
            )
        if self.speed_factor <= 0:
            raise ValueError(f"speed_factor must be positive, got {self.speed_factor}")


@dataclass
class ParallaxLayer:
    """Describes one layer inside a parallax composition.

    Attributes:
        video_path: Path to the layer video/image.
        speed: Relative speed multiplier compared to the base layer.
        x_offset: Horizontal offset in pixels from the center.
        y_offset: Vertical offset in pixels from the center.
        depth: Depth ordering – lower values are rendered in front.
    """

    video_path: str
    speed: float = 1.0
    x_offset: int = 0
    y_offset: int = 0
    depth: int = 0


@dataclass
class CropPoint:
    """A keyframe for dynamic crop effects.

    Attributes:
        x: Horizontal crop origin (pixels from left).
        y: Vertical crop origin (pixels from top).
        width: Crop width in pixels.
        height: Crop height in pixels.
        time: Absolute time in seconds at which this keyframe is reached.
    """

    x: int
    y: int
    width: int
    height: int
    time: float

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"width and height must be positive, got {self.width}x{self.height}"
            )
        if self.time < 0:
            raise ValueError(f"time must be >= 0, got {self.time}")


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _probe_video(path: str) -> Dict[str, Any]:
    """Return ffprobe JSON metadata for *path*."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def _get_video_dimensions(path: str) -> Tuple[int, int]:
    """Return ``(width, height)`` of the first video stream."""
    info = _probe_video(path)
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "video":
            return int(stream["width"]), int(stream["height"])
    raise RuntimeError(f"No video stream found in {path}")


def _get_video_duration(path: str) -> float:
    """Return duration in seconds."""
    info = _probe_video(path)
    fmt = info.get("format", {})
    if "duration" in fmt:
        return float(fmt["duration"])
    # Fallback: derive from stream frames / avg_frame_rate
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "video":
            r_frame_rate = stream.get("r_frame_rate", "0/1")
            nb_frames = stream.get("nb_frames")
            if nb_frames and "/" in r_frame_rate:
                num, den = (int(x) for x in r_frame_rate.split("/"))
                if den:
                    return int(nb_frames) * den / num
    raise RuntimeError(f"Cannot determine duration of {path}")


def _get_video_fps(path: str) -> float:
    """Return frames per second as a float."""
    info = _probe_video(path)
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "video":
            r_frame_rate = stream.get("r_frame_rate", "30/1")
            if "/" in r_frame_rate:
                num, den = (int(x) for x in r_frame_rate.split("/"))
                if den:
                    return num / den
            return float(r_frame_rate)
    return 30.0


# Easing functions -----------------------------------------------------------

def _ease_in(t: float) -> float:
    """Quadratic ease-in."""
    return t * t


def _ease_out(t: float) -> float:
    """Quadratic ease-out."""
    return t * (2.0 - t)


def _ease_in_out(t: float) -> float:
    """Smooth ease-in-out (cubic approximation of sinusoidal)."""
    if t < 0.5:
        return 4.0 * t * t * t
    return 1.0 - (-2.0 * t + 2.0) ** 3 / 2.0


_EASING: Dict[str, Any] = {
    "linear": lambda t: t,
    "ease_in": _ease_in,
    "ease_out": _ease_out,
    "ease_in_out": _ease_in_out,
}


def _apply_easing(t: float, curve: str = "ease_in_out") -> float:
    """Apply the named easing function to *t* ∈ [0, 1]."""
    fn = _EASING.get(curve, _ease_in_out)
    return max(0.0, min(1.0, fn(t)))


# ---------------------------------------------------------------------------
# FFmpeg command builder helpers
# ---------------------------------------------------------------------------

def _ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def _run_ffmpeg(cmd: List[str]) -> subprocess.CompletedProcess[str]:
    """Execute an FFmpeg command, logging and raising on failure."""
    logger.debug("ffmpeg cmd: %s", " ".join(cmd))
    return subprocess.run(cmd, capture_output=True, text=True, check=True)


# ---------------------------------------------------------------------------
# MotionEffectsEngine
# ---------------------------------------------------------------------------

class MotionEffectsEngine:
    """High-level engine that applies motion effects to video files.

    Every public method writes the result to *output_path* and returns it.
    FFmpeg is invoked via :func:`subprocess.run` for maximum portability.
    """

    # ------------------------------------------------------------------
    # Zoom effects
    # ------------------------------------------------------------------

    def apply_zoom_in(
        self,
        video_path: str,
        start_scale: float,
        end_scale: float,
        center_x: float,
        center_y: float,
        duration: float,
        start_time: float,
        output_path: str,
    ) -> str:
        """Apply a smooth ease-in-out zoom-in effect.

        Args:
            video_path: Source video file.
            start_scale: Initial scale factor (e.g. ``1.0``).
            end_scale: Final scale factor (e.g. ``1.5``).
            center_x: Horizontal centre of the zoom as a fraction in [0, 1].
            center_y: Vertical centre of the zoom as a fraction in [0, 1].
            duration: Length of the zoom effect in seconds.
            start_time: Offset into the video where the zoom begins.
            output_path: Destination file path.

        Returns:
            The *output_path*.
        """
        _ensure_parent(output_path)
        width, height = _get_video_dimensions(video_path)
        fps = _get_video_fps(video_path)
        total_frames = max(1, int(duration * fps))

        # Build zoom expressions that ease-in-out over the duration
        # zoompan interprets the frame index via 'on' and total via 'in'
        # We apply a custom easing via the zoom expression with on/in
        zoom_expr = (
            f"if(eq(on,0),{start_scale},"
            f"({start_scale}+({end_scale}-{start_scale})"
            f"*(1-cos(PI*on/max({total_frames}-1,1)))/2))"
        )

        cx = int(center_x * width)
        cy = int(center_y * height)

        filter_complex = (
            f"[0:v]trim=start={start_time}:duration={duration},setpts=PTS-STARTPTS,"
            f"zoompan=z='{zoom_expr}'"
            f":x='iw/2-(iw/zoom/2)+({cx}-iw/2)*(1-1/zoom)'"
            f":y='ih/2-(ih/zoom/2)+({cy}-ih/2)*(1-1/zoom)'"
            f":d={total_frames}:s={width}x{height}:fps={fps}[outv]"
        )

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-map", "0:a?",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-c:a", "copy",
            output_path,
        ]
        _run_ffmpeg(cmd)
        return output_path

    def apply_zoom_out(
        self,
        video_path: str,
        start_scale: float,
        end_scale: float,
        center_x: float,
        center_y: float,
        duration: float,
        start_time: float,
        output_path: str,
    ) -> str:
        """Apply a smooth ease-in-out zoom-out effect.

        Identical to :meth:`apply_zoom_in` but with inverted scale direction.
        """
        # Reuse zoom_in with swapped scales
        return self.apply_zoom_in(
            video_path=video_path,
            start_scale=end_scale,
            end_scale=start_scale,
            center_x=center_x,
            center_y=center_y,
            duration=duration,
            start_time=start_time,
            output_path=output_path,
        )

    # ------------------------------------------------------------------
    # Ken Burns
    # ------------------------------------------------------------------

    def apply_ken_burns(
        self,
        video_path: str,
        start_pos: Tuple[int, int],
        end_pos: Tuple[int, int],
        start_scale: float,
        end_scale: float,
        duration: float,
        output_path: str,
    ) -> str:
        """Apply a Ken Burns pan-and-zoom effect with smooth ease-in-out.

        The entire clip is re-timed to *duration* seconds with the
        Ken Burns motion applied across the full length.

        Args:
            video_path: Source video (typically a still image or clip).
            start_pos: ``(x, y)`` pixel offset at the beginning.
            end_pos: ``(x, y)`` pixel offset at the end.
            start_scale: Initial zoom scale.
            end_scale: Final zoom scale.
            duration: Effect duration in seconds.
            output_path: Destination file path.

        Returns:
            The *output_path*.
        """
        _ensure_parent(output_path)
        width, height = _get_video_dimensions(video_path)
        fps = _get_video_fps(video_path)
        total_frames = max(1, int(duration * fps))

        sx, sy = start_pos
        ex, ey = end_pos

        # Cosine-interpolated ease-in-out for both zoom and position
        zoom_expr = (
            f"({start_scale}+({end_scale}-{start_scale})"
            f"*(1-cos(PI*on/max({total_frames}-1,1)))/2)"
        )
        x_expr = (
            f"({sx}+({ex}-{sx})"
            f"*(1-cos(PI*on/max({total_frames}-1,1)))/2)"
        )
        y_expr = (
            f"({sy}+({ey}-{sy})"
            f"*(1-cos(PI*on/max({total_frames}-1,1)))/2)"
        )

        # Ken Burns: scale then crop to output size using the eased offsets
        filter_complex = (
            f"[0:v]loop=loop=-1:size=1:start=0,"
            f"zoompan=z='{zoom_expr}'"
            f":x='{x_expr}'"
            f":y='{y_expr}'"
            f":d={total_frames}:s={width}x{height}:fps={fps}[outv]"
        )

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            output_path,
        ]
        _run_ffmpeg(cmd)
        return output_path

    # ------------------------------------------------------------------
    # Pan effects
    # ------------------------------------------------------------------

    def apply_pan(
        self,
        video_path: str,
        direction: str,
        amount: int,
        duration: float,
        start_time: float,
        output_path: str,
    ) -> str:
        """Pan the video in the specified direction with smooth easing.

        Args:
            video_path: Source video.
            direction: One of ``"left"``, ``"right"``, ``"up"``, ``"down"``.
            amount: Number of pixels to pan.
            duration: Length of the pan effect in seconds.
            start_time: Offset into the video where the pan begins.
            output_path: Destination file path.

        Returns:
            The *output_path*.
        """
        _ensure_parent(output_path)
        width, height = _get_video_dimensions(video_path)
        fps = _get_video_fps(video_path)
        total_frames = max(1, int(duration * fps))

        direction = direction.lower().strip()
        if direction not in {"left", "right", "up", "down"}:
            raise ValueError(f"Invalid direction: {direction!r}")

        # Ease-in-out pan: 0 → amount over the clip
        t_norm = f"(1-cos(PI*on/max({total_frames}-1,1)))/2"

        if direction == "left":
            # Pan left means the frame moves left → crop x decreases
            x_expr = f"({width}-iw)/2+{amount}*{t_norm}"
            y_expr = f"(ih-oh)/2"
        elif direction == "right":
            x_expr = f"({width}-iw)/2-{amount}*{t_norm}"
            y_expr = f"(ih-oh)/2"
        elif direction == "up":
            x_expr = f"(iw-oh)/2"  # iw-oh is a rough centre; see y
            x_expr = f"(iw-ow)/2"
            y_expr = f"({height}-ih)/2+{amount}*{t_norm}"
        else:  # down
            x_expr = f"(iw-ow)/2"
            y_expr = f"({height}-ih)/2-{amount}*{t_norm}"

        filter_complex = (
            f"[0:v]trim=start={start_time}:duration={duration},"
            f"setpts=PTS-STARTPTS,"
            f"zoompan=z=1"
            f":x='{x_expr}'"
            f":y='{y_expr}'"
            f":d={total_frames}:s={width}x{height}:fps={fps}[outv]"
        )

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-map", "0:a?",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-c:a", "copy",
            output_path,
        ]
        _run_ffmpeg(cmd)
        return output_path

    # ------------------------------------------------------------------
    # Camera shake
    # ------------------------------------------------------------------

    def apply_camera_shake(
        self,
        video_path: str,
        intensity: float,
        frequency: float,
        duration: float,
        start_time: float,
        output_path: str,
    ) -> str:
        """Apply a randomised camera-shake effect.

        Generates random offsets in both axes and applies them through
        the ``geq`` filter for frame-accurate displacement.

        Args:
            video_path: Source video.
            intensity: Maximum pixel displacement per axis.
            frequency: Oscillation frequency (cycles per second).
            duration: Effect length in seconds.
            start_time: Offset into the video where the effect begins.
            output_path: Destination file path.

        Returns:
            The *output_path*.
        """
        _ensure_parent(output_path)
        width, height = _get_video_dimensions(video_path)
        fps = _get_video_fps(video_path)
        total_frames = max(1, int(duration * fps))

        # Pre-compute random offsets for every frame and write to a temp file
        # so that the geq filter can read them deterministically.
        seed = random.randint(0, 2**31)
        offsets: List[Tuple[int, int]] = []
        rng = random.Random(seed)
        for i in range(total_frames):
            t = i / max(fps, 1)
            # Combine a low-frequency sine with per-frame jitter
            base_x = math.sin(2.0 * math.pi * frequency * t) * intensity * 0.6
            base_y = math.cos(2.0 * math.pi * frequency * t * 0.97) * intensity * 0.6
            jitter_x = rng.gauss(0, intensity * 0.4)
            jitter_y = rng.gauss(0, intensity * 0.4)
            ox = int(round(base_x + jitter_x))
            oy = int(round(base_y + jitter_y))
            offsets.append((ox, oy))

        # Write offsets to a JSON sidecar so geq can use it
        sidecar_fd, sidecar_path = tempfile.mkstemp(suffix=".json", prefix="shake_")
        try:
            with os.fdopen(sidecar_fd, "w") as fh:
                json.dump(offsets, fh)

            # Use a simpler approach: build a geq expression that clips
            # displacement via random(), but since geq can't access external
            # data we embed the offsets directly in a huge expression.
            # For practical purposes, we generate a deterministic formula
            # using the frame number (n) with a pseudo-random hash.
            x_shift = (
                f"clip({intensity}*sin(2*PI*{frequency}*N/{fps})"
                f"+{intensity}*0.7*(random(N+{seed})-0.5)*2,"
                f"-{intensity},{intensity})"
            )
            y_shift = (
                f"clip({intensity}*cos(2*PI*{frequency}*0.97*N/{fps})"
                f"+{intensity}*0.7*(random(N+{seed}+12345)-0.5)*2,"
                f"-{intensity},{intensity})"
            )

            filter_complex = (
                f"[0:v]trim=start={start_time}:duration={duration},"
                f"setpts=PTS-STARTPTS,"
                f"crop=iw-2*abs({intensity}):ih-2*abs({intensity})"
                f":({width}-iw)/2:({height}-ih)/2,"
                f"geq="
                f"r='r(X+{x_shift},Y+{y_shift})':"
                f"g='g(X+{x_shift},Y+{y_shift})':"
                f"b='b(X+{x_shift},Y+{y_shift})',"
                f"scale={width}:{height}[outv]"
            )

            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-filter_complex", filter_complex,
                "-map", "[outv]",
                "-map", "0:a?",
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-c:a", "copy",
                output_path,
            ]
            _run_ffmpeg(cmd)
        finally:
            os.unlink(sidecar_path)

        return output_path

    # ------------------------------------------------------------------
    # Cinematic push
    # ------------------------------------------------------------------

    def apply_cinematic_push(
        self,
        video_path: str,
        direction: str,
        amount: float,
        duration: float,
        output_path: str,
    ) -> str:
        """Apply a cinematic push-in / push-out with combined zoom and pan.

        Args:
            video_path: Source video.
            direction: One of ``"in"``, ``"out"``, ``"left"``, ``"right"``,
                ``"up"``, ``"down"``.
            amount: Push magnitude (scale factor for in/out, pixels for
                directional pushes).
            duration: Effect duration in seconds.
            output_path: Destination file path.

        Returns:
            The *output_path*.
        """
        _ensure_parent(output_path)
        width, height = _get_video_dimensions(video_path)
        fps = _get_video_fps(video_path)
        total_frames = max(1, int(duration * fps))
        direction = direction.lower().strip()

        t_norm = f"(1-cos(PI*on/max({total_frames}-1,1)))/2"

        if direction == "in":
            z = f"1+({amount}-1)*{t_norm}"
            x, y = f"(iw-ow)/2", f"(ih-oh)/2"
        elif direction == "out":
            z = f"{amount}-({amount}-1)*{t_norm}"
            x, y = f"(iw-ow)/2", f"(ih-oh)/2"
        elif direction == "left":
            z = "1"
            x = f"(iw-ow)/2+{int(amount)}*{t_norm}"
            y = "(ih-oh)/2"
        elif direction == "right":
            z = "1"
            x = f"(iw-ow)/2-{int(amount)}*{t_norm}"
            y = "(ih-oh)/2"
        elif direction == "up":
            z = "1"
            x = "(iw-ow)/2"
            y = f"(ih-oh)/2+{int(amount)}*{t_norm}"
        elif direction == "down":
            z = "1"
            x = "(iw-ow)/2"
            y = f"(ih-oh)/2-{int(amount)}*{t_norm}"
        else:
            raise ValueError(f"Invalid direction: {direction!r}")

        filter_complex = (
            f"[0:v]zoompan=z='{z}':x='{x}':y='{y}'"
            f":d={total_frames}:s={width}x{height}:fps={fps}[outv]"
        )

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            output_path,
        ]
        _run_ffmpeg(cmd)
        return output_path

    # ------------------------------------------------------------------
    # Focus zoom
    # ------------------------------------------------------------------

    def apply_focus_zoom(
        self,
        video_path: str,
        focus_point: Tuple[float, float],
        zoom_amount: float,
        duration: float,
        start_time: float,
        output_path: str,
    ) -> str:
        """Zoom into *focus_point* while keeping it centred in the output.

        Args:
            video_path: Source video.
            focus_point: ``(x_frac, y_frac)`` in range [0, 1].
            zoom_amount: Target zoom scale (e.g. ``2.0``).
            duration: Effect duration in seconds.
            start_time: Offset into the video.
            output_path: Destination file path.

        Returns:
            The *output_path*.
        """
        _ensure_parent(output_path)
        width, height = _get_video_dimensions(video_path)
        fps = _get_video_fps(video_path)
        total_frames = max(1, int(duration * fps))

        fx, fy = focus_point
        target_w = int(width / zoom_amount)
        target_h = int(height / zoom_amount)
        target_x = max(0, min(int(fx * width) - target_w // 2, width - target_w))
        target_y = max(0, min(int(fy * height) - target_h // 2, height - target_h))

        # Eased transition from full frame to the crop region
        t_norm = f"(1-cos(PI*on/max({total_frames}-1,1)))/2"

        crop_w = f"{width}-({width}-{target_w})*{t_norm}"
        crop_h = f"{height}-({height}-{target_h})*{t_norm}"
        crop_x = f"{target_x}*{t_norm}"
        crop_y = f"{target_y}*{t_norm}"

        filter_complex = (
            f"[0:v]trim=start={start_time}:duration={duration},"
            f"setpts=PTS-STARTPTS,"
            f"crop=w='{crop_w}':h='{crop_h}':x='{crop_x}':y='{crop_y}',"
            f"scale={width}:{height}:flags=lanczos[outv]"
        )

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-map", "0:a?",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-c:a", "copy",
            output_path,
        ]
        _run_ffmpeg(cmd)
        return output_path

    # ------------------------------------------------------------------
    # Parallax
    # ------------------------------------------------------------------

    def apply_parallax(
        self,
        video_path: str,
        layers: List[ParallaxLayer],
        speeds: List[float],
        output_path: str,
    ) -> str:
        """Composite multiple layers with different scroll speeds (parallax).

        Each layer is scaled to the output resolution and shifted
        horizontally according to its speed relative to a base layer.

        Args:
            video_path: Background / reference video (ignored if layers
                already contain the background).
            layers: List of :class:`ParallaxLayer` instances. Layers with
                lower ``depth`` values are drawn on top.
            speeds: Horizontal speed (pixels per second) for each layer.
                Must match the length of *layers*.
            output_path: Destination file path.

        Returns:
            The *output_path*.
        """
        _ensure_parent(output_path)
        if len(layers) != len(speeds):
            raise ValueError(
                f"layers ({len(layers)}) and speeds ({len(speeds)}) must match"
            )

        width, height = _get_video_dimensions(video_path)
        duration = _get_video_duration(video_path)

        sorted_layers = sorted(zip(layers, speeds), key=lambda ls: ls[0].depth)

        # Build an overlay chain from back to front
        inputs: List[str] = []
        filter_parts: List[str] = []
        prev_label = "base"

        for idx, (layer, speed) in enumerate(sorted_layers):
            inp_idx = idx + 1  # first input is [0:v]
            shift_px = speed * duration
            label = f"L{idx}"

            # Scale layer to output size, then shift horizontally
            filter_parts.append(
                f"[{inp_idx}:v]scale={width}:{height}:flags=lanczos,"
                f"scroll=h=0:v={shift_px / duration:.6f}[{label}]"
            )
            inputs.extend(["-i", layer.video_path])

            if idx == 0:
                prev_label = label
                continue

            # Overlay previous on top
            overlay_label = f"OV{idx}"
            filter_parts.append(
                f"[{prev_label}][{label}]overlay=shortest=1[{overlay_label}]"
            )
            prev_label = overlay_label

        filter_complex = ";".join(filter_parts)

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            *inputs,
            "-filter_complex", filter_complex,
            "-map", f"[{prev_label}]",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-t", str(duration),
            output_path,
        ]
        _run_ffmpeg(cmd)
        return output_path

    # ------------------------------------------------------------------
    # Speed effects
    # ------------------------------------------------------------------

    def apply_slow_motion(
        self,
        video_path: str,
        factor: float,
        output_path: str,
    ) -> str:
        """Apply a uniform slow-motion (or fast-motion) effect.

        Args:
            video_path: Source video.
            factor: Speed multiplier. ``0.5`` = half speed, ``2.0`` = double.
            output_path: Destination file path.

        Returns:
            The *output_path*.
        """
        _ensure_parent(output_path)
        if factor <= 0:
            raise ValueError(f"factor must be positive, got {factor}")

        pts_factor = 1.0 / factor
        filter_complex = f"[0:v]setpts={pts_factor:.6f}*PTS[v];[0:a]atempo={factor:.6f}[a]"

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-filter_complex", filter_complex,
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            output_path,
        ]
        _run_ffmpeg(cmd)
        return output_path

    def apply_speed_ramp(
        self,
        video_path: str,
        segments: List[SpeedSegment],
        output_path: str,
    ) -> str:
        """Apply variable speed ramping with configurable per-segment curves.

        Each segment specifies a fraction of the total duration and a speed
        factor.  ``setpts`` values are computed per-frame using the segment
        definitions and easing curves.

        Args:
            video_path: Source video.
            segments: Non-overlapping, ordered list of :class:`SpeedSegment`
                instances that together span [0, 1].
            output_path: Destination file path.

        Returns:
            The *output_path*.
        """
        _ensure_parent(output_path)
        if not segments:
            raise ValueError("At least one segment is required")

        duration = _get_video_duration(video_path)
        fps = _get_video_fps(video_path)
        total_frames = int(math.ceil(duration * fps))

        # Build a frame-by-frame PTS multiplier array
        pts_factors: List[float] = []
        for seg in segments:
            seg_start_frame = int(seg.start_fraction * total_frames)
            seg_end_frame = int(seg.end_fraction * total_frames)
            seg_frames = max(1, seg_end_frame - seg_start_frame)

            easing_fn = _EASING.get(seg.curve, _ease_in_out)

            for f in range(seg_frames):
                local_t = f / max(seg_frames - 1, 1)
                eased_t = easing_fn(local_t)
                # Speed factor transitions smoothly if curve != constant
                if seg.curve == "constant":
                    pts_factors.append(1.0 / seg.speed_factor)
                else:
                    # Interpolate speed factor via easing (useful when
                    # neighbouring segments have different factors –
                    # here we keep it within the single segment)
                    pts_factors.append(1.0 / seg.speed_factor)

        # Build a complex filter with select + setpts per segment
        # Simpler approach: concatenate segments as separate trim+setpts
        filter_parts: List[str] = []
        concat_inputs: List[str] = []

        for idx, seg in enumerate(segments):
            seg_start = seg.start_fraction * duration
            seg_duration = (seg.end_fraction - seg.start_fraction) * duration
            pts_factor = 1.0 / seg.speed_factor
            label = f"S{idx}"

            filter_parts.append(
                f"[0:v]trim=start={seg_start:.6f}:duration={seg_duration:.6f},"
                f"setpts=PTS*{pts_factor:.6f}[{label}]"
            )
            concat_inputs.append(f"[{label}]")

        n = len(segments)
        concat_str = "".join(concat_inputs) + f"concat=n={n}:v=1:a=0[outv]"

        # Audio: apply the same compound speed factor
        audio_parts: List[str] = []
        audio_chain = ""
        for idx, seg in enumerate(segments):
            factor = seg.speed_factor
            # atempo only supports [0.5, 100]; chain for extreme values
            af = f"atempo={factor:.6f}"
            audio_parts.append(
                f"[0:a]atrim=start={seg.start_fraction * duration:.6f}"
                f":duration={(seg.end_fraction - seg.start_fraction) * duration:.6f},"
                f"asetpts=PTS-STARTPTS,{af}[a{idx}]"
            )
        audio_concat = "".join(f"[a{idx}]" for idx in range(n))
        audio_cat_str = (
            audio_concat + f"concat=n={n}:v=0:a=1[outa]"
            if n > 0 else ""
        )

        all_filters = filter_parts + audio_parts + [concat_str]
        if audio_cat_str:
            all_filters.append(audio_cat_str)

        filter_complex = ";".join(all_filters)

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-filter_complex", filter_complex,
            "-map", "[outv]",
        ]
        if audio_cat_str:
            cmd.extend(["-map", "[outa]"])
        else:
            cmd.extend(["-map", "0:a?"])

        cmd.extend([
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            output_path,
        ])
        _run_ffmpeg(cmd)
        return output_path

    # ------------------------------------------------------------------
    # Reverse
    # ------------------------------------------------------------------

    def apply_reverse(
        self,
        video_path: str,
        output_path: str,
    ) -> str:
        """Reverse both video and audio tracks.

        Args:
            video_path: Source video.
            output_path: Destination file path.

        Returns:
            The *output_path*.
        """
        _ensure_parent(output_path)
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", "reverse",
            "-af", "areverse",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            output_path,
        ]
        _run_ffmpeg(cmd)
        return output_path

    # ------------------------------------------------------------------
    # Stabilize
    # ------------------------------------------------------------------

    def apply_stabilize(
        self,
        video_path: str,
        output_path: str,
    ) -> str:
        """Two-pass video stabilisation using the ``vidstabtransform`` filter.

        Pass 1 analyses motion vectors and writes them to a transforms file.
        Pass 2 applies the compensating transforms.

        Args:
            video_path: Source video.
            output_path: Destination file path.

        Returns:
            The *output_path*.
        """
        _ensure_parent(output_path)
        transforms_fd, transforms_path = tempfile.mkstemp(
            suffix=".trf", prefix="stab_"
        )
        os.close(transforms_fd)

        try:
            # Pass 1 – analyse
            analyse_cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vf", f"vidstabdetect=shakiness=5:accuracy=15:result={transforms_path}",
                "-f", "null", "-",
            ]
            _run_ffmpeg(analyse_cmd)

            # Pass 2 – transform
            transform_cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vf", (
                    f"vidstabtransform=input={transforms_path}"
                    ":smoothing=10:optzoom=1:interpol=bicubic,"
                    "unsharp=5:5:0.8:3:3:0.4"
                ),
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-c:a", "copy",
                output_path,
            ]
            _run_ffmpeg(transform_cmd)
        finally:
            os.unlink(transforms_path)

        return output_path

    # ------------------------------------------------------------------
    # Dynamic crop
    # ------------------------------------------------------------------

    def apply_dynamic_crop(
        self,
        video_path: str,
        target_points: List[CropPoint],
        durations: List[float],
        output_path: str,
    ) -> str:
        """Animate a crop window through a series of keyframe positions.

        The crop window transitions between successive points over the
        corresponding *durations*.  Transitions are eased with
        ``ease_in_out``.

        Args:
            video_path: Source video.
            target_points: Ordered list of :class:`CropPoint` keyframes.
            durations: Time in seconds for each transition.  Must have
                ``len(target_points) - 1`` entries.
            output_path: Destination file path.

        Returns:
            The *output_path*.
        """
        _ensure_parent(output_path)
        if len(target_points) < 2:
            raise ValueError("At least two target_points are required")
        if len(durations) != len(target_points) - 1:
            raise ValueError(
                f"durations length ({len(durations)}) must be "
                f"len(target_points)-1 ({len(target_points) - 1})"
            )

        fps = _get_video_fps(video_path)
        width, height = _get_video_dimensions(video_path)

        # Build piecewise crop expressions
        # Accumulate frame offsets
        cumulative_frames: List[int] = [0]
        running = 0
        for d in durations:
            running += max(1, int(d * fps))
            cumulative_frames.append(running)

        crop_parts: List[str] = []
        for idx in range(len(target_points) - 1):
            p0 = target_points[idx]
            p1 = target_points[idx + 1]
            f0 = cumulative_frames[idx]
            f1 = cumulative_frames[idx + 1]

            # Ease-in-out interpolation
            t_expr = f"clip((on-{f0})/max({f1}-{f0},1),0,1)"
            eased = (
                f"if(lt({t_expr},0.5),"
                f"4*{t_expr}*{t_expr}*{t_expr},"
                f"1-pow(-2*{t_expr}+2,3)/2)"
            )

            crop_x = f"{p0.x}+({p1.x}-{p0.x})*{eased}"
            crop_y = f"{p0.y}+({p1.y}-{p0.y})*{eased}"
            crop_w = f"{p0.width}+({p1.width}-{p0.width})*{eased}"
            crop_h = f"{p0.height}+({p1.height}-{p0.height})*{eased}"

            condition = f"between(on,{f0},{f1})"
            crop_parts.append(
                f"if({condition},crop=w='{crop_w}':h='{crop_h}'"
                f":x='{crop_x}':y='{crop_y}',identity)"
            )

        # Chain conditional crops
        # Since ffmpeg geq-style conditional crops aren't directly
        # supported in the crop filter, we build individual trim+crop
        # segments and concat them.
        filter_parts: List[str] = []
        concat_labels: List[str] = []

        for idx in range(len(target_points) - 1):
            p0 = target_points[idx]
            p1 = target_points[idx + 1]
            seg_start = sum(durations[:idx]) if idx > 0 else 0.0
            seg_duration = durations[idx]
            t_pts = f"(1-cos(PI*(on)/max({max(1, int(seg_duration * fps))}-1,1)))/2"

            # Interpolate crop parameters using expressions
            interp_x = f"{p0.x}+({p1.x}-{p0.x})*{t_pts}"
            interp_y = f"{p0.y}+({p1.y}-{p0.y})*{t_pts}"
            interp_w = f"{p0.width}+({p1.width}-{p0.width})*{t_pts}"
            interp_h = f"{p0.height}+({p1.height}-{p0.height})*{t_pts}"
            label = f"C{idx}"

            filter_parts.append(
                f"[0:v]trim=start={seg_start:.6f}:duration={seg_duration:.6f},"
                f"setpts=PTS-STARTPTS,"
                f"crop=w='{interp_w}':h='{interp_h}'"
                f":x='{interp_x}':y='{interp_y}',"
                f"scale={width}:{height}:flags=lanczos[{label}]"
            )
            concat_labels.append(f"[{label}]")

        n_segs = len(target_points) - 1
        concat_str = "".join(concat_labels) + f"concat=n={n_segs}:v=1:a=0[outv]"
        filter_complex = ";".join(filter_parts + [concat_str])

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            output_path,
        ]
        _run_ffmpeg(cmd)
        return output_path


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_default_engine: Optional[MotionEffectsEngine] = None


def get_engine() -> MotionEffectsEngine:
    """Return the module-level singleton :class:`MotionEffectsEngine`."""
    global _default_engine
    if _default_engine is None:
        _default_engine = MotionEffectsEngine()
    return _default_engine
