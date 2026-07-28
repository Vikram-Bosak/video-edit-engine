"""
Graphics and Overlay Engine for Video Editing.

Provides image/logo overlay, progress bars, gradients, vignettes,
watermarks, lower thirds, intro/outro animations, and more.
Uses ffmpeg for compositing and Pillow for image generation.
"""

from __future__ import annotations

import enum
import math
import os
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

from PIL import Image, ImageDraw, ImageFilter, ImageFont


# ---------------------------------------------------------------------------
# Enums & Data Classes
# ---------------------------------------------------------------------------

class OverlayPosition(enum.Enum):
    """Named screen positions for overlays."""

    TOP_LEFT = "top_left"
    TOP_CENTER = "top_center"
    TOP_RIGHT = "top_right"
    CENTER_LEFT = "center_left"
    CENTER = "center"
    CENTER_RIGHT = "center_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_CENTER = "bottom_center"
    BOTTOM_RIGHT = "bottom_right"


@dataclass
class Overlay:
    """Describes a single overlay element to composite onto video."""

    path: str
    position: OverlayPosition = OverlayPosition.CENTER
    scale: float = 1.0
    opacity: float = 1.0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    z_index: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_position(
    position: OverlayPosition,
    video_width: int,
    video_height: int,
    overlay_width: int,
    overlay_height: int,
    margin: int = 10,
) -> Tuple[int, int]:
    """Return (x, y) pixel coordinates for *position* relative to the video frame."""

    mapping = {
        OverlayPosition.TOP_LEFT: (margin, margin),
        OverlayPosition.TOP_CENTER: ((video_width - overlay_width) // 2, margin),
        OverlayPosition.TOP_RIGHT: (video_width - overlay_width - margin, margin),
        OverlayPosition.CENTER_LEFT: (margin, (video_height - overlay_height) // 2),
        OverlayPosition.CENTER: (
            (video_width - overlay_width) // 2,
            (video_height - overlay_height) // 2,
        ),
        OverlayPosition.CENTER_RIGHT: (
            video_width - overlay_width - margin,
            (video_height - overlay_height) // 2,
        ),
        OverlayPosition.BOTTOM_LEFT: (margin, video_height - overlay_height - margin),
        OverlayPosition.BOTTOM_CENTER: (
            (video_width - overlay_width) // 2,
            video_height - overlay_height - margin,
        ),
        OverlayPosition.BOTTOM_RIGHT: (
            video_width - overlay_width - margin,
            video_height - overlay_height - margin,
        ),
    }
    return mapping[position]


def _get_video_dimensions(video_path: str) -> Tuple[int, int]:
    """Return (width, height) of *video_path* via ffprobe."""

    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0:s=x",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    parts = result.stdout.strip().split("x")
    return int(parts[0]), int(parts[1])


def _get_video_duration(video_path: str) -> float:
    """Return duration in seconds of *video_path* via ffprobe."""

    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def _temp_path(suffix: str = ".png") -> str:
    """Return a temporary file path inside a dedicated working directory."""

    work_dir = os.path.join(tempfile.gettempdir(), "graphics_overlay_engine")
    os.makedirs(work_dir, exist_ok=True)
    return os.path.join(work_dir, f"{uuid.uuid4().hex}{suffix}")


def _build_overlay_filter(
    overlays: List[Tuple[str, int, int, float]],
    enable_ranges: Optional[List[Tuple[float, float]]] = None,
) -> str:
    """Build an ffmpeg -filter_complex overlay chain.

    *overlays* items: (overlay_path, x, y, opacity)
    *enable_ranges* items: (start, end) in seconds – one per overlay, or None.
    """

    parts: List[str] = []
    current_label = "[0:v]"

    for idx, (img_path, x, y, opacity) in enumerate(overlays):
        next_label = f"[ov{idx}]"
        input_label = f"[img{idx}]"

        enable_expr = ""
        if enable_ranges and idx < len(enable_ranges):
            s, e = enable_ranges[idx]
            enable_expr = f":enable='between(t,{s},{e})'"

        if opacity < 1.0:
            alpha_label = f"[a{idx}]"
            parts.append(
                f"{input_label}format=rgba,colorchannelmixer=aa={opacity}{alpha_label}"
            )
            parts.append(
                f"{current_label}{alpha_label}overlay={x}:{y}:format=auto{enable_expr}{next_label}"
            )
        else:
            parts.append(
                f"{current_label}{input_label}overlay={x}:{y}:format=auto{enable_expr}{next_label}"
            )

        current_label = next_label

    return ";".join(parts)


# ---------------------------------------------------------------------------
# GraphicsEngine
# ---------------------------------------------------------------------------

class GraphicsEngine:
    """Full-featured graphics and overlay engine backed by ffmpeg + Pillow."""

    def __init__(self, ffmpeg_path: str = "ffmpeg", ffprobe_path: str = "ffprobe") -> None:
        self._ffmpeg = ffmpeg_path
        self._ffprobe = ffprobe_path
        self._verify_ffmpeg()

    # -- internal helpers ---------------------------------------------------

    def _verify_ffmpeg(self) -> None:
        """Raise if ffmpeg binary is not reachable."""
        if shutil.which(self._ffmpeg) is None:
            raise FileNotFoundError(
                f"ffmpeg not found at '{self._ffmpeg}'. "
                "Please install ffmpeg and ensure it is on PATH."
            )

    def _run_ffmpeg(self, cmd: List[str], description: str = "ffmpeg") -> None:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"{description} failed: {result.stderr}")

    def _get_dimensions(self, path: str) -> Tuple[int, int]:
        return _get_video_dimensions(path)

    def _get_duration(self, path: str) -> float:
        return _get_video_duration(path)

    def _ensure_dir(self, path: str) -> None:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)

    # -----------------------------------------------------------------------
    # overlay_image
    # -----------------------------------------------------------------------

    def overlay_image(
        self,
        video_path: str,
        image_path: str,
        position: OverlayPosition = OverlayPosition.CENTER,
        scale: float = 1.0,
        opacity: float = 1.0,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        output_path: str = "output_overlay.mp4",
    ) -> str:
        """Overlay an image onto a video with position, scale, opacity, and timing.

        Args:
            video_path: Path to the source video.
            image_path: Path to the overlay image (PNG with transparency supported).
            position: Where to place the overlay on the video frame.
            scale: Scale factor for the overlay image (1.0 = original size).
            opacity: Opacity for the overlay (0.0 – 1.0).
            start_time: Time in seconds when the overlay appears. None = entire video.
            end_time: Time in seconds when the overlay disappears. None = end of video.
            output_path: Path for the rendered output file.

        Returns:
            The *output_path* string.
        """
        self._ensure_dir(output_path)
        vw, vh = self._get_dimensions(video_path)

        img = Image.open(image_path)
        iw, ih = img.size
        nw, nh = int(iw * scale), int(ih * scale)
        x, y = _resolve_position(position, vw, vh, nw, nh)

        resized = _temp_path(".png")
        img.resize((nw, nh), Image.LANCZOS).save(resized)

        enable = ""
        if start_time is not None or end_time is not None:
            s = start_time if start_time is not None else 0
            e = end_time if end_time is not None else 99999
            enable = f":enable='between(t,{s},{e})'"

        filter_parts: List[str] = []
        input_maps: List[str] = ["-i", video_path, "-i", resized]

        if opacity < 1.0:
            filter_complex = (
                f"[1:v]format=rgba,colorchannelmixer=aa={opacity}[ov];"
                f"[0:v][ov]overlay={x}:{y}:format=auto{enable}[out]"
            )
        else:
            filter_complex = (
                f"[0:v][1:v]overlay={x}:{y}:format=auto{enable}[out]"
            )

        cmd = [
            self._ffmpeg, "-y",
            *input_maps,
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-map", "0:a?",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "copy",
            output_path,
        ]
        self._run_ffmpeg(cmd, "overlay_image")

        try:
            os.remove(resized)
        except OSError:
            pass

        return output_path

    # -----------------------------------------------------------------------
    # overlay_logo
    # -----------------------------------------------------------------------

    def overlay_logo(
        self,
        video_path: str,
        logo_path: str,
        position: OverlayPosition = OverlayPosition.TOP_RIGHT,
        scale: float = 0.15,
        opacity: float = 1.0,
        output_path: str = "output_logo.mp4",
    ) -> str:
        """Overlay a logo on every frame of the video.

        Args:
            video_path: Path to the source video.
            logo_path: Path to the logo image (PNG preferred).
            position: Named position on the video frame.
            scale: Scale factor relative to the video width (0.0 – 1.0).
            opacity: Logo opacity (0.0 – 1.0).
            output_path: Path for the rendered output file.

        Returns:
            The *output_path* string.
        """
        self._ensure_dir(output_path)
        vw, vh = self._get_dimensions(video_path)

        logo = Image.open(logo_path)
        lw = int(vw * scale)
        lh = int(logo.size[1] * (lw / logo.size[0]))
        x, y = _resolve_position(position, vw, vh, lw, lh)

        resized = _temp_path(".png")
        logo.resize((lw, lh), Image.LANCZOS).save(resized)

        if opacity < 1.0:
            fc = (
                f"[1:v]format=rgba,colorchannelmixer=aa={opacity}[ov];"
                f"[0:v][ov]overlay={x}:{y}:format=auto[out]"
            )
        else:
            fc = f"[0:v][1:v]overlay={x}:{y}:format=auto[out]"

        cmd = [
            self._ffmpeg, "-y",
            "-i", video_path, "-i", resized,
            "-filter_complex", fc,
            "-map", "[out]", "-map", "0:a?",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "copy",
            output_path,
        ]
        self._run_ffmpeg(cmd, "overlay_logo")

        try:
            os.remove(resized)
        except OSError:
            pass
        return output_path

    # -----------------------------------------------------------------------
    # create_progress_bar
    # -----------------------------------------------------------------------

    def create_progress_bar(
        self,
        video_path: str,
        color: str = "#00FF00",
        height: int = 8,
        position: OverlayPosition = OverlayPosition.BOTTOM_CENTER,
        output_path: str = "output_progress.mp4",
    ) -> str:
        """Add an animated progress bar that fills from left to right over the video.

        Args:
            video_path: Path to the source video.
            color: Hex colour string for the bar (e.g. ``"#00FF00"``).
            height: Height in pixels of the progress bar.
            position: Named position for the bar on the frame.
            output_path: Path for the rendered output file.

        Returns:
            The *output_path* string.
        """
        self._ensure_dir(output_path)
        vw, vh = self._get_dimensions(video_path)
        duration = self._get_duration(video_path)

        bar_frames_dir = _temp_path(suffix="")
        os.makedirs(bar_frames_dir, exist_ok=True)

        fps = 30
        total_frames = int(math.ceil(duration * fps))

        for i in range(total_frames):
            frac = (i + 1) / max(total_frames, 1)
            frame_w = max(1, int(vw * frac))
            frame_h = height

            bar = Image.new("RGBA", (frame_w, frame_h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(bar)
            r, g, b = _hex_to_rgb(color)
            draw.rectangle([0, 0, frame_w - 1, frame_h - 1], fill=(r, g, b, 255))
            bar.save(os.path.join(bar_frames_dir, f"bar_{i:06d}.png"))

        x, y = _resolve_position(position, vw, vh, vw, height)

        padding = _temp_path(suffix="")
        os.makedirs(padding, exist_ok=True)
        padded_frames_dir = _temp_path(suffix="")
        os.makedirs(padded_frames_dir, exist_ok=True)

        for i in range(total_frames):
            frac = (i + 1) / max(total_frames, 1)
            frame_w = max(1, int(vw * frac))

            padded = Image.new("RGBA", (vw, height), (0, 0, 0, 0))
            r, g, b = _hex_to_rgb(color)
            draw = ImageDraw.Draw(padded)
            draw.rectangle([0, 0, frame_w - 1, height - 1], fill=(r, g, b, 255))
            padded.save(os.path.join(padded_frames_dir, f"pad_{i:06d}.png"))

        bar_video = _temp_path(".mp4")
        cmd = [
            self._ffmpeg, "-y",
            "-framerate", str(fps),
            "-i", os.path.join(padded_frames_dir, "pad_%06d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuva420p",
            "-t", str(duration),
            bar_video,
        ]
        self._run_ffmpeg(cmd, "create_progress_bar_video")

        fc = f"[0:v][1:v]overlay={x}:{y}:format=auto:shortest=1[out]"
        cmd = [
            self._ffmpeg, "-y",
            "-i", video_path, "-i", bar_video,
            "-filter_complex", fc,
            "-map", "[out]", "-map", "0:a?",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "copy",
            output_path,
        ]
        self._run_ffmpeg(cmd, "overlay_progress_bar")

        shutil.rmtree(bar_frames_dir, ignore_errors=True)
        shutil.rmtree(padded_frames_dir, ignore_errors=True)
        shutil.rmtree(padding, ignore_errors=True)
        try:
            os.remove(bar_video)
        except OSError:
            pass

        return output_path

    # -----------------------------------------------------------------------
    # create_gradient_overlay
    # -----------------------------------------------------------------------

    def create_gradient_overlay(
        self,
        video_path: str,
        colors: List[str],
        direction: str = "vertical",
        opacity: float = 0.5,
        output_path: str = "output_gradient.mp4",
    ) -> str:
        """Apply a gradient overlay across the entire video.

        Args:
            video_path: Path to the source video.
            colors: List of hex colour strings to blend in the gradient.
            direction: ``"vertical"`` or ``"horizontal"``.
            opacity: Overlay opacity (0.0 – 1.0).
            output_path: Path for the rendered output file.

        Returns:
            The *output_path* string.
        """
        self._ensure_dir(output_path)
        vw, vh = self._get_dimensions(video_path)

        grad_path = _temp_path(".png")
        self.generate_gradient_image(vw, vh, colors, direction, grad_path)

        if opacity < 1.0:
            img = Image.open(grad_path)
            alpha = img.split()[3]
            alpha = alpha.point(lambda p: int(p * opacity))
            img.putalpha(alpha)
            grad_path_opaque = _temp_path(".png")
            img.save(grad_path_opaque)
            os.remove(grad_path)
            grad_path = grad_path_opaque

        fc = (
            f"[0:v]split[s0][s1];"
            f"[s1]colorchannelmixer=aa=0[bg];"
            f"[bg][1:v]overlay=0:0:format=auto[fg];"
            f"[s0][fg]overlay=0:0:format=auto:shortest=1[out]"
        )
        cmd = [
            self._ffmpeg, "-y",
            "-i", video_path, "-i", grad_path,
            "-filter_complex", fc,
            "-map", "[out]", "-map", "0:a?",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "copy",
            output_path,
        ]
        self._run_ffmpeg(cmd, "create_gradient_overlay")

        try:
            os.remove(grad_path)
        except OSError:
            pass
        return output_path

    # -----------------------------------------------------------------------
    # create_vignette
    # -----------------------------------------------------------------------

    def create_vignette(
        self,
        video_path: str,
        intensity: float = 0.6,
        output_path: str = "output_vignette.mp4",
    ) -> str:
        """Apply a vignette (darkened edges) effect to the video.

        Args:
            video_path: Path to the source video.
            intensity: Strength of the vignette effect (0.0 – 1.0).
            output_path: Path for the rendered output file.

        Returns:
            The *output_path* string.
        """
        self._ensure_dir(output_path)
        vw, vh = self._get_dimensions(video_path)

        vig = _generate_vignette_image(vw, vh, intensity)

        if intensity < 1.0:
            alpha = vig.split()[3]
            alpha = alpha.point(lambda p: int(p * intensity))
            vig.putalpha(alpha)

        vig_path = _temp_path(".png")
        vig.save(vig_path)

        fc = (
            f"[0:v]split[s0][s1];"
            f"[s1]colorchannelmixer=aa=0[bg];"
            f"[bg][1:v]overlay=0:0:format=auto[fg];"
            f"[s0][fg]overlay=0:0:format=auto:shortest=1[out]"
        )
        cmd = [
            self._ffmpeg, "-y",
            "-i", video_path, "-i", vig_path,
            "-filter_complex", fc,
            "-map", "[out]", "-map", "0:a?",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "copy",
            output_path,
        ]
        self._run_ffmpeg(cmd, "create_vignette")

        try:
            os.remove(vig_path)
        except OSError:
            pass
        return output_path

    # -----------------------------------------------------------------------
    # add_frame_border
    # -----------------------------------------------------------------------

    def add_frame_border(
        self,
        video_path: str,
        border_image: str,
        output_path: str = "output_bordered.mp4",
    ) -> str:
        """Add a decorative frame border around the video.

        The *border_image* should be larger than the video.  The video is
        scaled down (if necessary) and centred inside the border.

        Args:
            video_path: Path to the source video.
            border_image: Path to the border/frame PNG.
            output_path: Path for the rendered output file.

        Returns:
            The *output_path* string.
        """
        self._ensure_dir(output_path)

        border = Image.open(border_image)
        bw, bh = border.size
        vw, vh = self._get_dimensions(video_path)

        border_path = _temp_path(".png")
        border.save(border_path)

        fc = f"[1:v][0:v]overlay=(W-w)/2:(H-h)/2:format=auto:shortest=1[out]"
        cmd = [
            self._ffmpeg, "-y",
            "-i", video_path, "-i", border_path,
            "-filter_complex", fc,
            "-map", "[out]", "-map", "0:a?",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "copy",
            output_path,
        ]
        self._run_ffmpeg(cmd, "add_frame_border")

        try:
            os.remove(border_path)
        except OSError:
            pass
        return output_path

    # -----------------------------------------------------------------------
    # add_watermark
    # -----------------------------------------------------------------------

    def add_watermark(
        self,
        video_path: str,
        watermark_text_or_image: Union[str, bytes],
        position: OverlayPosition = OverlayPosition.BOTTOM_RIGHT,
        opacity: float = 0.5,
        output_path: str = "output_watermark.mp4",
    ) -> str:
        """Add a text or image watermark to the video.

        Args:
            video_path: Path to the source video.
            watermark_text_or_image: A text string **or** a path to an image file.
            position: Where to place the watermark.
            opacity: Watermark opacity (0.0 – 1.0).
            output_path: Path for the rendered output file.

        Returns:
            The *output_path* string.
        """
        self._ensure_dir(output_path)
        vw, vh = self._get_dimensions(video_path)

        if os.path.isfile(watermark_text_or_image):
            return self.overlay_image(
                video_path,
                watermark_text_or_image,
                position=position,
                opacity=opacity,
                output_path=output_path,
            )

        wm_img = _create_text_image(
            watermark_text_or_image,
            font_size=24,
            text_color="#FFFFFF",
            bg_color=None,
        )
        wm_path = _temp_path(".png")
        wm_img.save(wm_path)

        return self.overlay_image(
            video_path,
            wm_path,
            position=position,
            opacity=opacity,
            output_path=output_path,
        )

    # -----------------------------------------------------------------------
    # create_intro_animation
    # -----------------------------------------------------------------------

    def create_intro_animation(
        self,
        video_path: str,
        logo_path: str,
        duration: float = 3.0,
        style: str = "fade_in",
        output_path: str = "output_intro.mp4",
    ) -> str:
        """Create an intro animation with a logo overlay.

        Styles:
            * ``"fade_in"``  – Logo fades in over *duration* seconds, then remains.
            * ``"slide_in"`` – Logo slides in from the left.
            * ``"zoom_in"``  – Logo scales up from 0 % to full size.
            * ``"bounce"``   – Logo bounces in from the top.

        Args:
            video_path: Path to the source video (or ``""`` for a generated background).
            logo_path: Path to the logo image.
            duration: Intro duration in seconds.
            style: One of the supported style names.
            output_path: Path for the rendered output file.

        Returns:
            The *output_path* string.
        """
        self._ensure_dir(output_path)

        vw, vh = 1920, 1080
        if video_path and os.path.isfile(video_path):
            vw, vh = self._get_dimensions(video_path)

        logo = Image.open(logo_path)
        lw, lh = logo.size
        max_w, max_h = int(vw * 0.4), int(vh * 0.4)
        ratio = min(max_w / lw, max_h / lh, 1.0)
        new_w, new_h = int(lw * ratio), int(lh * ratio)
        logo_resized = logo.resize((new_w, new_h), Image.LANCZOS)

        fps = 30
        total_frames = int(math.ceil(duration * fps))
        frames_dir = _temp_path(suffix="")
        os.makedirs(frames_dir, exist_ok=True)

        bg_color = (18, 18, 24)

        for i in range(total_frames):
            t = (i + 1) / fps
            progress = min(t / duration, 1.0)

            frame = Image.new("RGBA", (vw, vh), (*bg_color, 255))

            if style == "fade_in":
                alpha = int(255 * _ease_out_cubic(progress))
                logo_frame = logo_resized.copy()
                r, g, b, _ = logo_frame.split()
                a = r.point(lambda _: alpha)
                logo_frame = Image.merge("RGBA", (r, g, b, a))
                px = (vw - new_w) // 2
                py = (vh - new_h) // 2
                frame.paste(logo_frame, (px, py), logo_frame)

            elif style == "slide_in":
                start_x = -new_w
                end_x = (vw - new_w) // 2
                cx = int(start_x + (end_x - start_x) * _ease_out_back(progress))
                cy = (vh - new_h) // 2
                frame.paste(logo_resized, (cx, cy), logo_resized)

            elif style == "zoom_in":
                scale = _ease_out_cubic(progress)
                zw = max(1, int(new_w * scale))
                zh = max(1, int(new_h * scale))
                zoomed = logo_resized.resize((zw, zh), Image.LANCZOS)
                zx = (vw - zw) // 2
                zy = (vh - zh) // 2
                frame.paste(zoomed, (zx, zy), zoomed)

            elif style == "bounce":
                start_y = -new_h
                end_y = (vh - new_h) // 2
                cy = int(start_y + (end_y - start_y) * _bounce_ease(progress))
                cx = (vw - new_w) // 2
                frame.paste(logo_resized, (cx, cy), logo_resized)

            else:
                alpha = int(255 * _ease_out_cubic(progress))
                logo_frame = logo_resized.copy()
                r, g, b, _ = logo_frame.split()
                a = r.point(lambda _: alpha)
                logo_frame = Image.merge("RGBA", (r, g, b, a))
                px = (vw - new_w) // 2
                py = (vh - new_h) // 2
                frame.paste(logo_frame, (px, py), logo_frame)

            frame.convert("RGB").save(os.path.join(frames_dir, f"f_{i:06d}.png"))

        intro_video = _temp_path(".mp4")
        cmd = [
            self._ffmpeg, "-y",
            "-framerate", str(fps),
            "-i", os.path.join(frames_dir, "f_%06d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-t", str(duration),
            intro_video,
        ]
        self._run_ffmpeg(cmd, "create_intro_frames")

        if video_path and os.path.isfile(video_path):
            cmd = [
                self._ffmpeg, "-y",
                "-i", intro_video, "-i", video_path,
                "-filter_complex",
                "[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
                "pad=1920:1080:(ow-iw)/2:(oh-ih)/2[v0];"
                "[1:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
                "pad=1920:1080:(ow-iw)/2:(oh-ih)/2[v1];"
                "[v0][v1]concat=n=2:v=1:a=0[out]",
                "-map", "[out]", "-map", "0:a?", "-map", "1:a?",
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-c:a", "copy",
                output_path,
            ]
            self._run_ffmpeg(cmd, "concat_intro")
            try:
                os.remove(intro_video)
            except OSError:
                pass
        else:
            shutil.move(intro_video, output_path)

        shutil.rmtree(frames_dir, ignore_errors=True)
        return output_path

    # -----------------------------------------------------------------------
    # create_outro_card
    # -----------------------------------------------------------------------

    def create_outro_card(
        self,
        channel_name: str,
        subscribe_text: str = "Subscribe for more!",
        duration: float = 5.0,
        width: int = 1920,
        height: int = 1080,
        output_path: str = "output_outro.mp4",
    ) -> str:
        """Generate an outro / end-screen card with channel name and subscribe CTA.

        Args:
            channel_name: Name of the channel to display.
            subscribe_text: Call-to-action text below the name.
            duration: How long the outro card plays (seconds).
            width: Output frame width in pixels.
            height: Output frame height in pixels.
            output_path: Path for the rendered output file.

        Returns:
            The *output_path* string.
        """
        self._ensure_dir(output_path)
        fps = 30
        total_frames = int(math.ceil(duration * fps))
        frames_dir = _temp_path(suffix="")
        os.makedirs(frames_dir, exist_ok=True)

        bg_top = (15, 15, 30)
        bg_bottom = (30, 10, 50)
        accent = (220, 50, 50)

        for i in range(total_frames):
            t = (i + 1) / fps
            progress = min(t / duration, 1.0)

            frame = _create_gradient_bg(width, height, bg_top, bg_bottom)
            draw = ImageDraw.Draw(frame)

            # --- channel name ---
            try:
                title_font = ImageFont.truetype("arial.ttf", 64)
            except (OSError, IOError):
                title_font = ImageFont.load_default()

            title_alpha = int(255 * _ease_out_cubic(min(progress * 2, 1.0)))
            bbox = draw.textbbox((0, 0), channel_name, font=title_font)
            tw = bbox[2] - bbox[0]
            tx = (width - tw) // 2
            ty = int(height * 0.3)
            draw.text(
                (tx, ty), channel_name,
                fill=(255, 255, 255, title_alpha), font=title_font,
            )

            # --- subscribe text ---
            try:
                sub_font = ImageFont.truetype("arial.ttf", 32)
            except (OSError, IOError):
                sub_font = ImageFont.load_default()

            sub_delay = 0.5
            sub_progress = max(0.0, (t - sub_delay) / (duration - sub_delay))
            sub_alpha = int(255 * _ease_out_cubic(min(sub_progress * 2, 1.0)))
            bbox_s = draw.textbbox((0, 0), subscribe_text, font=sub_font)
            sw = bbox_s[2] - bbox_s[0]
            sx = (width - sw) // 2
            sy = int(height * 0.45)
            draw.text(
                (sx, sy), subscribe_text,
                fill=(*accent, sub_alpha), font=sub_font,
            )

            # --- decorative line ---
            line_progress = _ease_out_cubic(min(progress * 1.5, 1.0))
            line_w = int(width * 0.3 * line_progress)
            line_x = (width - line_w) // 2
            line_y = int(height * 0.42)
            draw.rectangle(
                [line_x, line_y, line_x + line_w, line_y + 3],
                fill=(*accent, int(200 * line_progress)),
            )

            # --- circular subscribe button hint ---
            btn_progress = max(0.0, (t - 1.0) / (duration - 1.0))
            btn_alpha = int(255 * _ease_out_cubic(min(btn_progress * 2, 1.0)))
            btn_r = 30
            btn_cx = width // 2
            btn_cy = int(height * 0.65)
            draw.ellipse(
                [btn_cx - btn_r, btn_cy - btn_r, btn_cx + btn_r, btn_cy + btn_r],
                fill=(220, 50, 50, btn_alpha),
            )
            try:
                btn_font = ImageFont.truetype("arial.ttf", 24)
            except (OSError, IOError):
                btn_font = ImageFont.load_default()
            play_bbox = draw.textbbox((0, 0), "\u25b6", font=btn_font)
            pw = play_bbox[2] - play_bbox[0]
            ph = play_bbox[3] - play_bbox[1]
            draw.text(
                (btn_cx - pw // 2, btn_cy - ph // 2 - 2),
                "\u25b6",
                fill=(255, 255, 255, btn_alpha), font=btn_font,
            )

            frame.convert("RGB").save(os.path.join(frames_dir, f"o_{i:06d}.png"))

        cmd = [
            self._ffmpeg, "-y",
            "-framerate", str(fps),
            "-i", os.path.join(frames_dir, "o_%06d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-t", str(duration),
            output_path,
        ]
        self._run_ffmpeg(cmd, "create_outro_card")
        shutil.rmtree(frames_dir, ignore_errors=True)
        return output_path

    # -----------------------------------------------------------------------
    # apply_lower_third
    # -----------------------------------------------------------------------

    def apply_lower_third(
        self,
        video_path: str,
        name: str,
        title: str = "",
        style: str = "modern",
        start_time: float = 0.0,
        duration: float = 4.0,
        output_path: str = "output_lowerthird.mp4",
    ) -> str:
        """Apply a lower-third name/title graphic over the video.

        Styles:
            * ``"modern"``  – Dark translucent bar with accent stripe.
            * ``"minimal"`` – Simple white text on transparent background.
            * ``"bold"``    – Coloured bar with white text.

        Args:
            video_path: Path to the source video.
            name: Primary name to display.
            title: Secondary title / role text.
            style: One of ``"modern"``, ``"minimal"``, ``"bold"``.
            start_time: Time (seconds) when the lower third appears.
            duration: How long the lower third stays visible (seconds).
            output_path: Path for the rendered output file.

        Returns:
            The *output_path* string.
        """
        self._ensure_dir(output_path)
        vw, vh = self._get_dimensions(video_path)

        bar_w = int(vw * 0.45)
        bar_h = 100 if title else 60
        lt_img = _create_lower_third_image(name, title, bar_w, bar_h, style)
        lt_path = _temp_path(".png")
        lt_img.save(lt_path)

        end_time = start_time + duration
        x, y = _resolve_position(
            OverlayPosition.BOTTOM_LEFT, vw, vh, bar_w, bar_h, margin=40
        )

        fc = (
            f"[1:v]format=rgba[ov];"
            f"[0:v][ov]overlay={x}:{y}:format=auto"
            f":enable='between(t,{start_time},{end_time})'[out]"
        )
        cmd = [
            self._ffmpeg, "-y",
            "-i", video_path, "-i", lt_path,
            "-filter_complex", fc,
            "-map", "[out]", "-map", "0:a?",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "copy",
            output_path,
        ]
        self._run_ffmpeg(cmd, "apply_lower_third")

        try:
            os.remove(lt_path)
        except OSError:
            pass
        return output_path

    # -----------------------------------------------------------------------
    # create_countdown
    # -----------------------------------------------------------------------

    def create_countdown(
        self,
        video_path: str,
        start_time: float = 0.0,
        duration: float = 5.0,
        style: str = "circular",
        output_path: str = "output_countdown.mp4",
    ) -> str:
        """Overlay an animated countdown timer on the video.

        Styles:
            * ``"circular"``  – Number inside a depleting circle.
            * ``"digital"``   – Large digital-style number.
            * ``"minimal"``   – Simple number in the corner.

        Args:
            video_path: Path to the source video.
            start_time: Time (seconds) when countdown begins.
            duration: Countdown duration in seconds (counts down from *duration*).
            style: One of the supported style names.
            output_path: Path for the rendered output file.

        Returns:
            The *output_path* string.
        """
        self._ensure_dir(output_path)
        vw, vh = self._get_dimensions(video_path)

        fps = 30
        total_frames = int(math.ceil(duration * fps))
        frames_dir = _temp_path(suffix="")
        os.makedirs(frames_dir, exist_ok=True)

        overlay_size = min(vw, vh) // 4

        for i in range(total_frames):
            t = (i + 1) / fps
            remaining = max(0, duration - t)
            seconds_left = int(math.ceil(remaining))
            frac = remaining / duration

            overlay = Image.new("RGBA", (overlay_size, overlay_size), (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            if style == "circular":
                cx, cy = overlay_size // 2, overlay_size // 2
                r = overlay_size // 2 - 4
                draw.ellipse(
                    [cx - r, cy - r, cx + r, cy + r],
                    outline=(255, 255, 255, 200), width=3,
                )
                angle = 360 * frac
                draw.arc(
                    [cx - r, cy - r, cx + r, cy + r],
                    start=-90, end=-90 + angle,
                    fill=(220, 50, 50, 255), width=5,
                )
                try:
                    num_font = ImageFont.truetype("arial.ttf", r)
                except (OSError, IOError):
                    num_font = ImageFont.load_default()
                text = str(seconds_left)
                bbox = draw.textbbox((0, 0), text, font=num_font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                draw.text(
                    (cx - tw // 2, cy - th // 2 - 4),
                    text, fill=(255, 255, 255, 255), font=num_font,
                )

            elif style == "digital":
                try:
                    d_font = ImageFont.truetype("arial.ttf", overlay_size - 10)
                except (OSError, IOError):
                    d_font = ImageFont.load_default()
                text = str(seconds_left)
                bbox = draw.textbbox((0, 0), text, font=d_font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                draw.text(
                    ((overlay_size - tw) // 2, (overlay_size - th) // 2 - 5),
                    text, fill=(255, 255, 255, 240), font=d_font,
                )

            else:
                try:
                    m_font = ImageFont.truetype("arial.ttf", 36)
                except (OSError, IOError):
                    m_font = ImageFont.load_default()
                text = str(seconds_left)
                draw.text((5, 5), text, fill=(255, 255, 255, 220), font=m_font)

            overlay.save(os.path.join(frames_dir, f"cd_{i:06d}.png"))

        cd_video = _temp_path(".mp4")
        cmd = [
            self._ffmpeg, "-y",
            "-framerate", str(fps),
            "-i", os.path.join(frames_dir, "cd_%06d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuva420p",
            "-t", str(duration),
            cd_video,
        ]
        self._run_ffmpeg(cmd, "create_countdown_video")

        margin = 40
        x = vw - overlay_size - margin
        y = margin

        fc = f"[0:v][1:v]overlay={x}:{y}:format=auto:enable='between(t,{start_time},{start_time + duration})'[out]"
        cmd = [
            self._ffmpeg, "-y",
            "-i", video_path, "-i", cd_video,
            "-filter_complex", fc,
            "-map", "[out]", "-map", "0:a?",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "copy",
            output_path,
        ]
        self._run_ffmpeg(cmd, "overlay_countdown")

        shutil.rmtree(frames_dir, ignore_errors=True)
        try:
            os.remove(cd_video)
        except OSError:
            pass
        return output_path

    # -----------------------------------------------------------------------
    # create_subscribe_button
    # -----------------------------------------------------------------------

    def create_subscribe_button(
        self,
        video_path: str,
        position: OverlayPosition = OverlayPosition.BOTTOM_RIGHT,
        start_time: float = 5.0,
        output_path: str = "output_subscribe.mp4",
    ) -> str:
        """Overlay an animated subscribe button that pulses on screen.

        Args:
            video_path: Path to the source video.
            position: Named position for the button.
            start_time: Time (seconds) when the button appears.
            output_path: Path for the rendered output file.

        Returns:
            The *output_path* string.
        """
        self._ensure_dir(output_path)
        vw, vh = self._get_dimensions(video_path)
        video_duration = self._get_duration(video_path)
        remaining = max(0.1, video_duration - start_time)

        fps = 30
        total_frames = int(math.ceil(remaining * fps))
        frames_dir = _temp_path(suffix="")
        os.makedirs(frames_dir, exist_ok=True)

        btn_w, btn_h = 220, 56
        rx, ry = 20, 12
        corner_radius = 28

        for i in range(total_frames):
            t = (i + 1) / fps
            entrance = min(t / 0.4, 1.0)  # 0.4s entrance
            pulse = 1.0 + 0.04 * math.sin(t * 3.0)

            frame = Image.new("RGBA", (btn_w + 20, btn_h + 20), (0, 0, 0, 0))

            eased = _ease_out_back(entrance)
            actual_w = int(btn_w * eased * pulse)
            actual_h = int(btn_h * eased)
            if actual_w < 1 or actual_h < 1:
                frame.save(os.path.join(frames_dir, f"sub_{i:06d}.png"))
                continue

            draw = ImageDraw.Draw(frame)
            ox = (btn_w + 20 - actual_w) // 2
            oy = (btn_h + 20 - actual_h) // 2

            draw.rounded_rectangle(
                [ox, oy, ox + actual_w, oy + actual_h],
                radius=min(corner_radius, actual_h // 2),
                fill=(220, 30, 30, int(255 * eased)),
            )

            if eased > 0.5:
                try:
                    btn_font = ImageFont.truetype("arial.ttf", 20)
                except (OSError, IOError):
                    btn_font = ImageFont.load_default()
                text = "SUBSCRIBE"
                bbox = draw.textbbox((0, 0), text, font=btn_font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                txt_alpha = int(255 * min((eased - 0.5) * 2, 1.0))
                draw.text(
                    (ox + (actual_w - tw) // 2, oy + (actual_h - th) // 2 - 2),
                    text,
                    fill=(255, 255, 255, txt_alpha),
                    font=btn_font,
                )

            frame.save(os.path.join(frames_dir, f"sub_{i:06d}.png"))

        sub_video = _temp_path(".mp4")
        cmd = [
            self._ffmpeg, "-y",
            "-framerate", str(fps),
            "-i", os.path.join(frames_dir, "sub_%06d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuva420p",
            "-t", str(remaining),
            sub_video,
        ]
        self._run_ffmpeg(cmd, "create_subscribe_video")

        x, y = _resolve_position(position, vw, vh, btn_w + 20, btn_h + 20, margin=30)

        fc = (
            f"[0:v][1:v]overlay={x}:{y}:format=auto"
            f":enable='gte(t,{start_time})'[out]"
        )
        cmd = [
            self._ffmpeg, "-y",
            "-i", video_path, "-i", sub_video,
            "-filter_complex", fc,
            "-map", "[out]", "-map", "0:a?",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "copy",
            output_path,
        ]
        self._run_ffmpeg(cmd, "overlay_subscribe_button")

        shutil.rmtree(frames_dir, ignore_errors=True)
        try:
            os.remove(sub_video)
        except OSError:
            pass
        return output_path

    # -----------------------------------------------------------------------
    # generate_gradient_image
    # -----------------------------------------------------------------------

    def generate_gradient_image(
        self,
        width: int,
        height: int,
        colors: List[str],
        direction: str = "vertical",
        output_path: str = "gradient.png",
    ) -> str:
        """Generate and save a gradient image.

        Args:
            width: Image width in pixels.
            height: Image height in pixels.
            colors: List of hex colour strings (2+).
            direction: ``"vertical"`` or ``"horizontal"``.
            output_path: Path for the output PNG.

        Returns:
            The *output_path* string.
        """
        self._ensure_dir(output_path)

        if len(colors) < 2:
            raise ValueError("At least two colours are required for a gradient.")

        rgb_cols = [_hex_to_rgb(c) for c in colors]

        if direction == "vertical":
            img = Image.new("RGB", (width, height))
            pixels = img.load()
            for y in range(height):
                frac = y / max(height - 1, 1)
                r, g, b = _interpolate_colors(rgb_cols, frac)
                for x in range(width):
                    pixels[x, y] = (int(r), int(g), int(b))
        else:
            img = Image.new("RGB", (width, height))
            pixels = img.load()
            for x in range(width):
                frac = x / max(width - 1, 1)
                r, g, b = _interpolate_colors(rgb_cols, frac)
                for y in range(height):
                    pixels[x, y] = (int(r), int(g), int(b))

        img.save(output_path)
        return output_path


# ---------------------------------------------------------------------------
# Private module-level helpers
# ---------------------------------------------------------------------------

def _hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    """Convert ``"#RRGGBB"`` or ``"RRGGBB"`` to ``(r, g, b)``."""
    h = hex_str.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _interpolate_colors(
    colours: List[Tuple[int, int, int]], t: float
) -> Tuple[float, float, float]:
    """Linearly interpolate across *colours* at position *t* ∈ [0, 1]."""
    if len(colours) < 2:
        return colours[0]
    t = max(0.0, min(1.0, t))
    seg = t * (len(colours) - 1)
    idx = min(int(seg), len(colours) - 2)
    local_t = seg - idx
    r1, g1, b1 = colours[idx]
    r2, g2, b2 = colours[idx + 1]
    return (
        r1 + (r2 - r1) * local_t,
        g1 + (g2 - g1) * local_t,
        b1 + (b2 - b1) * local_t,
    )


def _ease_out_cubic(t: float) -> float:
    return 1.0 - (1.0 - t) ** 3


def _ease_out_back(t: float) -> float:
    c1 = 1.70158
    c3 = c1 + 1.0
    return 1.0 + c3 * (t - 1.0) ** 3 + c1 * (t - 1.0) ** 2


def _bounce_ease(t: float) -> float:
    if t < 1 / 2.75:
        return 7.5625 * t * t
    elif t < 2 / 2.75:
        t -= 1.5 / 2.75
        return 7.5625 * t * t + 0.75
    elif t < 2.5 / 2.75:
        t -= 2.25 / 2.75
        return 7.5625 * t * t + 0.9375
    else:
        t -= 2.625 / 2.75
        return 7.5625 * t * t + 0.984375


def _generate_vignette_image(
    width: int, height: int, intensity: float
) -> Image.Image:
    """Create a vignette overlay image (dark edges, transparent centre)."""

    vig = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    pixels = vig.load()
    cx, cy = width / 2.0, height / 2.0
    max_dist = math.sqrt(cx * cx + cy * cy)

    for y in range(height):
        for x in range(width):
            dx = (x - cx) / cx
            dy = (y - cy) / cy
            dist = math.sqrt(dx * dx + dy * dy)
            factor = min(dist, 1.0)
            alpha = int(255 * factor * factor * intensity)
            pixels[x, y] = (0, 0, 0, min(alpha, 255))

    return vig


def _create_text_image(
    text: str,
    font_size: int = 24,
    text_color: str = "#FFFFFF",
    bg_color: Optional[str] = None,
    padding: int = 10,
) -> Image.Image:
    """Render *text* to a PIL Image with optional background."""

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()

    tmp = Image.new("RGBA", (1, 1))
    tmp_draw = ImageDraw.Draw(tmp)
    bbox = tmp_draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    img_w = tw + padding * 2
    img_h = th + padding * 2

    if bg_color:
        r, g, b = _hex_to_rgb(bg_color)
        img = Image.new("RGBA", (img_w, img_h), (r, g, b, 200))
    else:
        img = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))

    draw = ImageDraw.Draw(img)
    tr, tg, tb = _hex_to_rgb(text_color)
    draw.text((padding, padding), text, fill=(tr, tg, tb, 255), font=font)
    return img


def _create_lower_third_image(
    name: str,
    title: str,
    width: int,
    height: int,
    style: str,
) -> Image.Image:
    """Build a lower-third graphic as a PIL Image."""

    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    try:
        name_font = ImageFont.truetype("arial.ttf", 28)
        title_font = ImageFont.truetype("arial.ttf", 18)
    except (OSError, IOError):
        name_font = ImageFont.load_default()
        title_font = ImageFont.load_default()

    if style == "modern":
        bar_alpha = 210
        draw.rounded_rectangle(
            [0, 0, width, height], radius=8, fill=(20, 20, 30, bar_alpha),
        )
        draw.rectangle([0, 0, 6, height], fill=(220, 50, 50, 255))
        draw.text((20, 12), name, fill=(255, 255, 255, 255), font=name_font)
        if title:
            draw.text(
                (20, height - 36), title,
                fill=(180, 180, 190, 255), font=title_font,
            )

    elif style == "minimal":
        draw.text((10, 10), name, fill=(255, 255, 255, 240), font=name_font)
        if title:
            draw.text(
                (10, height - 30), title,
                fill=(255, 255, 255, 180), font=title_font,
            )

    elif style == "bold":
        draw.rectangle([0, 0, width, height], fill=(220, 50, 50, 230))
        draw.text((20, 12), name, fill=(255, 255, 255, 255), font=name_font)
        if title:
            draw.text(
                (20, height - 36), title,
                fill=(255, 255, 255, 220), font=title_font,
            )

    else:
        draw.rounded_rectangle(
            [0, 0, width, height], radius=8, fill=(20, 20, 30, 200),
        )
        draw.text((20, 12), name, fill=(255, 255, 255, 255), font=name_font)
        if title:
            draw.text(
                (20, height - 36), title,
                fill=(180, 180, 190, 255), font=title_font,
            )

    return img


def _create_gradient_bg(
    width: int,
    height: int,
    top: Tuple[int, int, int],
    bottom: Tuple[int, int, int],
) -> Image.Image:
    """Vertical linear gradient background."""

    img = Image.new("RGBA", (width, height))
    pixels = img.load()
    for y in range(height):
        t = y / max(height - 1, 1)
        r = int(top[0] + (bottom[0] - top[0]) * t)
        g = int(top[1] + (bottom[1] - top[1]) * t)
        b = int(top[2] + (bottom[2] - top[2]) * t)
        for x in range(width):
            pixels[x, y] = (r, g, b, 255)
    return img
