"""Video analysis and downloading module.

Provides VideoAnalyzer for extracting metadata, detecting orientation,
calculating quality scores, and analyzing audio/visual properties using
ffprobe and OpenCV. VideoDownloader handles single, audio-only, and batch
downloads via yt-dlp/ffmpeg.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class VideoMetadata:
    """Core metadata extracted from a video file."""

    width: int = 0
    height: int = 0
    fps: float = 0.0
    duration: float = 0.0
    codec: str = ""
    audio_codec: str = ""
    bitrate: float = 0.0
    file_size: int = 0
    frame_count: int = 0
    pixel_format: str = ""
    has_audio: bool = False
    has_video: bool = False


@dataclass
class AudioAnalysis:
    """Detailed audio properties."""

    has_audio: bool = False
    sample_rate: int = 0
    channels: int = 0
    bitrate: float = 0.0
    loudness: float = 0.0
    peak_level: float = 0.0


@dataclass
class VisualQuality:
    """Visual quality metrics computed from sampled frames."""

    sharpness_score: float = 0.0
    brightness_score: float = 0.0
    contrast_score: float = 0.0
    noise_level: float = 0.0
    color_richness: float = 0.0


@dataclass
class RecommendedSettings:
    """Suggested post-processing settings derived from analysis."""

    target_crop: str = "none"
    target_duration: float = 0.0
    color_grading: str = "none"
    motion_effects: List[str] = field(default_factory=list)
    transitions: List[str] = field(default_factory=list)
    text_overlay_needed: bool = False


@dataclass
class VideoAnalysis:
    """Aggregate analysis result for a video file."""

    metadata: VideoMetadata = field(default_factory=VideoMetadata)
    quality_score: float = 0.0
    orientation: str = "landscape"
    aspect_ratio: str = "16:9"
    recommended_settings: RecommendedSettings = field(
        default_factory=RecommendedSettings
    )
    visual_quality: VisualQuality = field(default_factory=VisualQuality)
    audio_analysis: AudioAnalysis = field(default_factory=AudioAnalysis)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _probe(video_path: str) -> Dict[str, Any]:
    """Run ffprobe and return parsed JSON output."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=30,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed for {video_path}: {result.stderr.strip()}"
        )
    return json.loads(result.stdout)


def _run_ffmpeg(args: List[str], timeout: int = 120) -> subprocess.CompletedProcess:
    """Run an ffmpeg command with standard options."""
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + args
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )


def _compute_aspect_ratio(width: int, height: int) -> str:
    """Return a simplified aspect-ratio string."""
    if width == 0 or height == 0:
        return "unknown"
    from math import gcd
    d = gcd(width, height)
    w, h = width // d, height // d
    ratio_map = {
        (16, 9): "16:9",
        (4, 3): "4:3",
        (21, 9): "21:9",
        (1, 1): "1:1",
        (9, 16): "9:16",
        (3, 4): "3:4",
    }
    return ratio_map.get((w, h), f"{w}:{h}")


# ---------------------------------------------------------------------------
# VideoAnalyzer
# ---------------------------------------------------------------------------

class VideoAnalyzer:
    """Analyze video files for metadata, quality, orientation, and more.

    Uses ffprobe for metadata extraction and OpenCV + NumPy for visual
    quality analysis. All public methods accept a file path and return
    dedicated dataclass results.
    """

    # Number of frames sampled for visual-quality analysis.
    _SAMPLE_FRAME_COUNT: int = 10

    # Weights used in the composite quality score (must sum to 1.0).
    _W_RESOLUTION: float = 0.20
    _W_BITRATE: float = 0.25
    _W_SHARPNESS: float = 0.25
    _W_AUDIO: float = 0.15
    _W_BRIGHTNESS: float = 0.10
    _W_CONTRAST: float = 0.05

    # Maximum values used for normalisation.
    _MAX_RESOLUTION_PIXELS: float = 3840 * 2160  # 4K UHD
    _MAX_BITRATE_BPS: float = 50_000_000  # 50 Mbps
    _MAX_SHARPNESS: float = 800.0
    _MAX_LOUDNESS_LUFS: float = -10.0
    _MAX_AUDIO_BITRATE_BPS: float = 320_000

    def __init__(self) -> None:
        if cv2 is None:
            raise ImportError(
                "opencv-python is required for VideoAnalyzer. "
                "Install it with: pip install opencv-python"
            )
        if np is None:
            raise ImportError(
                "numpy is required for VideoAnalyzer. "
                "Install it with: pip install numpy"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_video(self, video_path: str) -> VideoAnalysis:
        """Perform a full analysis of *video_path*.

        Returns a :class:`VideoAnalysis` containing metadata, quality
        score, orientation, aspect ratio, recommended settings, visual
        quality metrics, and audio analysis.
        """
        metadata = self.get_metadata(video_path)
        orientation = self.detect_orientation(video_path)
        aspect_ratio = self.detect_aspect_ratio(video_path)
        quality_score = self.calculate_quality_score(video_path)
        visual_quality = self.analyze_visual_quality(video_path)
        audio_analysis = self.analyze_audio(video_path)
        recommended = self.get_recommended_settings(video_path)

        return VideoAnalysis(
            metadata=metadata,
            quality_score=quality_score,
            orientation=orientation,
            aspect_ratio=aspect_ratio,
            recommended_settings=recommended,
            visual_quality=visual_quality,
            audio_analysis=audio_analysis,
        )

    def get_metadata(self, video_path: str) -> VideoMetadata:
        """Extract core metadata from *video_path* via ffprobe."""
        info = _probe(video_path)
        fmt: Dict[str, Any] = info.get("format", {})
        streams: List[Dict[str, Any]] = info.get("streams", [])

        video_stream = next(
            (s for s in streams if s.get("codec_type") == "video"), {}
        )
        audio_stream = next(
            (s for s in streams if s.get("codec_type") == "audio"), {}
        )

        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))

        # Frame rate – evaluate fraction expressions like "30000/1001".
        fps_raw = video_stream.get("r_frame_rate", "0/1")
        fps = self._parse_fps(fps_raw)

        duration = float(
            video_stream.get("duration", fmt.get("duration", 0))
        )
        frame_count = int(
            video_stream.get("nb_frames", self.get_frame_count(video_path))
        )

        bitrate = float(
            video_stream.get("bit_rate", fmt.get("bit_rate", 0))
        )
        file_size = int(fmt.get("size", os.path.getsize(video_path)))

        has_video = any(
            s.get("codec_type") == "video" for s in streams
        )
        has_audio = any(
            s.get("codec_type") == "audio" for s in streams
        )

        return VideoMetadata(
            width=width,
            height=height,
            fps=fps,
            duration=duration,
            codec=video_stream.get("codec_name", ""),
            audio_codec=audio_stream.get("codec_name", ""),
            bitrate=bitrate,
            file_size=file_size,
            frame_count=frame_count,
            pixel_format=video_stream.get("pix_fmt", ""),
            has_audio=has_audio,
            has_video=has_video,
        )

    def detect_orientation(self, video_path: str) -> str:
        """Return ``'landscape'``, ``'portrait'``, or ``'square'``."""
        meta = self.get_metadata(video_path)
        if meta.width > meta.height:
            return "landscape"
        if meta.height > meta.width:
            return "portrait"
        return "square"

    def calculate_quality_score(self, video_path: str) -> float:
        """Compute a 0-100 composite quality score.

        The score is a weighted combination of resolution, bitrate,
        sharpness, brightness, contrast, and audio quality, each
        normalised to 0-1.
        """
        meta = self.get_metadata(video_path)
        visual = self.analyze_visual_quality(video_path)
        audio = self.analyze_audio(video_path)

        resolution_score = min(
            (meta.width * meta.height) / self._MAX_RESOLUTION_PIXELS, 1.0
        )
        bitrate_score = min(meta.bitrate / self._MAX_BITRATE_BPS, 1.0)
        sharpness_score = min(
            visual.sharpness_score / self._MAX_SHARPNESS, 1.0
        )
        brightness_score = visual.brightness_score  # already 0-1
        contrast_score = visual.contrast_score  # already 0-1

        if audio.has_audio and audio.bitrate > 0:
            audio_score = min(
                audio.bitrate / self._MAX_AUDIO_BITRATE_BPS, 1.0
            )
        else:
            audio_score = 0.0

        raw = (
            self._W_RESOLUTION * resolution_score
            + self._W_BITRATE * bitrate_score
            + self._W_SHARPNESS * sharpness_score
            + self._W_AUDIO * audio_score
            + self._W_BRIGHTNESS * brightness_score
            + self._W_CONTRAST * contrast_score
        )
        return round(raw * 100, 2)

    def detect_aspect_ratio(self, video_path: str) -> str:
        """Return a simplified aspect-ratio string (e.g. ``'16:9'``)."""
        meta = self.get_metadata(video_path)
        return _compute_aspect_ratio(meta.width, meta.height)

    def get_frame_count(self, video_path: str) -> int:
        """Return the total number of frames."""
        info = _probe(video_path)
        for stream in info.get("streams", []):
            if stream.get("codec_type") == "video":
                nb = stream.get("nb_frames")
                if nb and nb != "N/A":
                    return int(nb)
        # Fallback: duration * fps.
        meta = self.get_metadata(video_path)
        if meta.fps > 0 and meta.duration > 0:
            return int(meta.duration * meta.fps)
        return 0

    def get_bitrate(self, video_path: str) -> float:
        """Return the overall bitrate in bits per second."""
        meta = self.get_metadata(video_path)
        return meta.bitrate

    def detect_interlacing(self, video_path: str) -> bool:
        """Detect whether the video is interlaced.

        Uses ffprobe field-order metadata and a heuristic Laplacian
        variance check on sampled frames.
        """
        info = _probe(video_path)
        for stream in info.get("streams", []):
            if stream.get("codec_type") != "video":
                continue
            field_order = stream.get("field_order", "")
            if field_order in ("tt", "bb", "tb", "bt"):
                return True
        # Heuristic fallback – interlaced frames exhibit horizontal
        # edge patterns that increase high-frequency energy.
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                return False
            sample_indices = [
                int(total * i / (self._SAMPLE_FRAME_COUNT + 1))
                for i in range(1, self._SAMPLE_FRAME_COUNT + 1)
            ]
            interlaced_votes = 0
            for idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                even = gray[0::2, :]
                odd = gray[1::2, :]
                diff_var = float(np.var(even.astype(np.float32) - odd.astype(np.float32)))
                if diff_var > 50.0:
                    interlaced_votes += 1
            return interlaced_votes > len(sample_indices) // 2
        finally:
            cap.release()

    def analyze_audio(self, video_path: str) -> AudioAnalysis:
        """Analyze the audio stream of *video_path*."""
        info = _probe(video_path)
        streams = info.get("streams", [])
        audio_stream = next(
            (s for s in streams if s.get("codec_type") == "audio"), None
        )
        if audio_stream is None:
            return AudioAnalysis(has_audio=False)

        sample_rate = int(audio_stream.get("sample_rate", 0))
        channels = int(audio_stream.get("channels", 0))
        bitrate = float(audio_stream.get("bit_rate", 0))

        # Loudness / peak via ffmpeg's loudnorm filter (two-pass first pass).
        loudness = 0.0
        peak_level = 0.0
        try:
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-i", str(video_path),
                "-af", "loudnorm=print_format=json",
                "-f", "null", "-",
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                creationflags=(
                    subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                ),
            )
            stderr = result.stderr
            json_match = re.search(r"\{.*\}", stderr, re.DOTALL)
            if json_match:
                loudnorm_data = json.loads(json_match.group())
                loudness = float(loudnorm_data.get("input_i", 0))
                peak_level = float(loudnorm_data.get("input_tp", 0))
        except Exception:
            pass  # Non-critical – leave defaults.

        return AudioAnalysis(
            has_audio=True,
            sample_rate=sample_rate,
            channels=channels,
            bitrate=bitrate,
            loudness=loudness,
            peak_level=peak_level,
        )

    def analyze_visual_quality(self, video_path: str) -> VisualQuality:
        """Sample frames and compute visual-quality metrics.

        Metrics returned (all 0-1 unless noted):
        - ``sharpness_score``: Laplacian variance (raw, higher = sharper).
        - ``brightness_score``: Mean luminance normalised to 0-1.
        - ``contrast_score``: Standard deviation of luminance normalised.
        - ``noise_level``: High-frequency residual energy.
        - ``color_richness``: Mean saturation from HSV.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return VisualQuality()

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return VisualQuality()

        sample_indices = [
            int(total * i / (self._SAMPLE_FRAME_COUNT + 1))
            for i in range(1, self._SAMPLE_FRAME_COUNT + 1)
        ]

        sharpness_vals: List[float] = []
        brightness_vals: List[float] = []
        contrast_vals: List[float] = []
        noise_vals: List[float] = []
        color_vals: List[float] = []

        try:
            for idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(
                    np.float32
                )

                # Sharpness via Laplacian variance.
                laplacian = cv2.Laplacian(gray, cv2.CV_32F)
                sharpness_vals.append(float(np.var(laplacian)))

                # Brightness.
                brightness_vals.append(float(np.mean(gray)) / 255.0)

                # Contrast.
                contrast_vals.append(
                    min(float(np.std(gray)) / 128.0, 1.0)
                )

                # Noise estimate: difference from blurred version.
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                noise = gray - blurred
                noise_vals.append(
                    min(float(np.std(noise)) / 30.0, 1.0)
                )

                # Colour richness via HSV saturation.
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                saturation = hsv[:, :, 1].astype(np.float32) / 255.0
                color_vals.append(float(np.mean(saturation)))
        finally:
            cap.release()

        def _mean(vals: List[float], default: float = 0.0) -> float:
            return float(np.mean(vals)) if vals else default

        return VisualQuality(
            sharpness_score=round(_mean(sharpness_vals), 4),
            brightness_score=round(_mean(brightness_vals), 4),
            contrast_score=round(_mean(contrast_vals), 4),
            noise_level=round(_mean(noise_vals), 4),
            color_richness=round(_mean(color_vals), 4),
        )

    def get_recommended_settings(self, video_path: str) -> RecommendedSettings:
        """Derive recommended post-processing settings from analysis."""
        meta = self.get_metadata(video_path)
        visual = self.analyze_visual_quality(video_path)
        audio = self.analyze_audio(video_path)
        orientation = self.detect_orientation(video_path)
        aspect = self.detect_aspect_ratio(video_path)

        # Crop recommendation.
        target_crop = "none"
        if aspect in ("4:3", "3:4"):
            target_crop = "pillarbox" if orientation == "landscape" else "letterbox"

        # Target duration – keep original unless very long.
        target_duration = meta.duration
        if meta.duration > 600:
            target_duration = 600.0  # Recommend trimming to 10 min.

        # Color grading.
        color_grading = "none"
        if visual.brightness_score < 0.3:
            color_grading = "brighten"
        elif visual.brightness_score > 0.75:
            color_grading = "darken"
        if visual.color_richness < 0.25:
            color_grading = "boost_saturation"

        # Motion effects.
        motion_effects: List[str] = []
        if visual.sharpness_score < 100:
            motion_effects.append("stabilize")
        if meta.fps < 24:
            motion_effects.append("smooth_interpolation")

        # Transitions.
        transitions: List[str] = []
        if meta.duration > 120:
            transitions.append("fade_between_segments")

        # Text overlay.
        text_overlay_needed = not audio.has_audio

        return RecommendedSettings(
            target_crop=target_crop,
            target_duration=target_duration,
            color_grading=color_grading,
            motion_effects=motion_effects,
            transitions=transitions,
            text_overlay_needed=text_overlay_needed,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_fps(raw: str) -> float:
        """Parse an fps value that may be a fraction like ``'30000/1001'``."""
        try:
            if "/" in raw:
                num, den = raw.split("/", 1)
                den_f = float(den)
                return float(num) / den_f if den_f else 0.0
            return float(raw)
        except (ValueError, ZeroDivisionError):
            return 0.0


# ---------------------------------------------------------------------------
# VideoDownloader
# ---------------------------------------------------------------------------

class VideoDownloader:
    """Download videos and audio from URLs.

    Supports direct file URLs as well as platforms supported by
    *yt-dlp* (YouTube, Twitter, etc.) when available. Falls back to
    :mod:`urllib` for plain HTTP downloads.
    """

    _YTDLP_AVAILABLE: Optional[bool] = None

    @classmethod
    def _has_ytdlp(cls) -> bool:
        """Check whether yt-dlp is installed and callable."""
        if cls._YTDLP_AVAILABLE is not None:
            return cls._YTDLP_AVAILABLE
        try:
            result = subprocess.run(
                ["yt-dlp", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                creationflags=(
                    subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                ),
            )
            cls._YTDLP_AVAILABLE = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            cls._YTDLP_AVAILABLE = False
        return cls._YTDLP_AVAILABLE

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """Remove or replace characters unsafe for file names."""
        name = re.sub(r'[<>:"/\\|?*]', "_", name)
        name = re.sub(r"\s+", "_", name)
        return name.strip("_")[:200] or "download"

    @staticmethod
    def _url_to_filename(url: str) -> str:
        """Derive a reasonable file name from *url*."""
        parsed = urllib.parse.urlparse(url)
        path_part = os.path.basename(parsed.path)
        if path_part:
            return urllib.parse.unquote(path_part)
        netloc = parsed.netloc.replace(".", "_")
        return netloc or "download"

    def download_video(self, url: str, output_dir: str) -> str:
        """Download the best-quality video to *output_dir*.

        Returns the path to the downloaded file.
        """
        os.makedirs(output_dir, exist_ok=True)

        if self._has_ytdlp():
            return self._download_with_ytdlp(
                url, output_dir, extract_audio=False
            )

        return self._download_with_urllib(url, output_dir)

    def download_audio(self, url: str, output_dir: str) -> str:
        """Download and extract audio from *url* to *output_dir*.

        Returns the path to the extracted audio file (mp3).
        """
        os.makedirs(output_dir, exist_ok=True)

        if self._has_ytdlp():
            return self._download_with_ytdlp(
                url, output_dir, extract_audio=True
            )

        # Fallback: download then extract audio via ffmpeg.
        temp_path = self._download_with_urllib(url, output_dir)
        audio_path = os.path.splitext(temp_path)[0] + ".mp3"
        result = _run_ffmpeg([
            "-i", temp_path,
            "-vn",
            "-acodec", "libmp3lame",
            "-q:a", "2",
            audio_path,
        ])
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg audio extraction failed: {result.stderr.strip()}"
            )
        # Remove the original video file.
        try:
            os.remove(temp_path)
        except OSError:
            pass
        return audio_path

    def download_batch(
        self, urls: List[str], output_dir: str
    ) -> List[str]:
        """Download multiple URLs sequentially.

        Returns a list of file paths in the same order as *urls*.
        """
        results: List[str] = []
        for url in urls:
            try:
                path = self.download_video(url, output_dir)
                results.append(path)
            except Exception as exc:
                # Record failures as empty strings so the caller can
                # correlate results by index.
                results.append("")
        return results

    # ------------------------------------------------------------------
    # Internal downloaders
    # ------------------------------------------------------------------

    def _download_with_ytdlp(
        self,
        url: str,
        output_dir: str,
        extract_audio: bool = False,
    ) -> str:
        """Use yt-dlp to download *url*."""
        filename_tpl = os.path.join(output_dir, "%(title)s.%(ext)s")
        cmd = [
            "yt-dlp",
            "--no-check-certificates",
            "-o", filename_tpl,
        ]
        if extract_audio:
            cmd += [
                "-x",
                "--audio-format", "mp3",
                "--audio-quality", "2",
            ]
        cmd.append(url)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            creationflags=(
                subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            ),
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"yt-dlp failed for {url}: {result.stderr.strip()}"
            )

        # Parse the output to find the downloaded file path.
        # yt-dlp prints the destination line: [download] Destination: ...
        for line in result.stdout.splitlines():
            match = re.search(
                r"\[download\]\s+(?:Destination|already downloaded):\s+(.+)",
                line,
            )
            if match:
                return match.group(1).strip()

        # Fallback: find the most recently modified file in output_dir.
        return self._find_latest_file(output_dir)

    def _download_with_urllib(self, url: str, output_dir: str) -> str:
        """Download a direct file URL via :mod:`urllib`."""
        filename = self._url_to_filename(url)
        filename = self._sanitize_filename(filename)
        dest = os.path.join(output_dir, filename)

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            with open(dest, "wb") as fp:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    fp.write(chunk)
        return dest

    @staticmethod
    def _find_latest_file(directory: str) -> str:
        """Return the path of the most recently modified file in *directory*."""
        latest_time = -1.0
        latest_path = ""
        for entry in os.scandir(directory):
            if entry.is_file():
                stat = entry.stat()
                if stat.st_mtime > latest_time:
                    latest_time = stat.st_mtime
                    latest_path = entry.path
        if not latest_path:
            raise FileNotFoundError(
                f"No files found in download directory: {directory}"
            )
        return latest_path
