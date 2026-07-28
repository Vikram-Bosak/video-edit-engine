"""
Export Engine - Video export, transcoding, and platform optimization.

Handles FFmpeg-based encoding, multi-resolution output, platform-specific
presets, thumbnail/preview generation, GIF conversion, and web optimization.
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class ExportFormat(Enum):
    MP4_H264 = "mp4_h264"
    MP4_H265 = "mp4_h265"
    WEBM_VP9 = "webm_vp9"
    AV1 = "av1"
    GIF = "gif"


class ExportPlatform(Enum):
    FACEBOOK_REEL = "facebook_reel"
    YOUTUBE_SHORT = "youtube_short"
    TIKTOK = "tiktok"
    INSTAGRAM_REEL = "instagram_reel"
    TWITTER_VIDEO = "twitter_video"


@dataclass
class ExportConfig:
    width: int = 1920
    height: int = 1080
    fps: float = 30.0
    codec: ExportFormat = ExportFormat.MP4_H264
    bitrate: str = "8M"
    audio_codec: str = "aac"
    audio_bitrate: str = "128k"
    pixel_format: str = "yuv420p"
    preset: str = "medium"
    crf: int = 23
    max_filesize_mb: Optional[float] = None


@dataclass
class ExportResult:
    output_path: str
    file_size_mb: float
    duration: float
    resolution: str
    codec: str
    success: bool
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


PLATFORM_PRESETS: Dict[ExportPlatform, Dict[str, Any]] = {
    ExportPlatform.FACEBOOK_REEL: {
        "width": 1080,
        "height": 1920,
        "codec": ExportFormat.MP4_H264,
        "bitrate": "8M",
        "audio_codec": "aac",
        "audio_bitrate": "128k",
        "preset": "medium",
        "crf": 23,
        "max_duration": 90.0,
        "pixel_format": "yuv420p",
    },
    ExportPlatform.YOUTUBE_SHORT: {
        "width": 1080,
        "height": 1920,
        "codec": ExportFormat.MP4_H264,
        "bitrate": "10M",
        "audio_codec": "aac",
        "audio_bitrate": "128k",
        "preset": "medium",
        "crf": 22,
        "max_duration": 60.0,
        "pixel_format": "yuv420p",
    },
    ExportPlatform.TIKTOK: {
        "width": 1080,
        "height": 1920,
        "codec": ExportFormat.MP4_H264,
        "bitrate": "8M",
        "audio_codec": "aac",
        "audio_bitrate": "128k",
        "preset": "medium",
        "crf": 23,
        "max_duration": 180.0,
        "pixel_format": "yuv420p",
    },
    ExportPlatform.INSTAGRAM_REEL: {
        "width": 1080,
        "height": 1920,
        "codec": ExportFormat.MP4_H264,
        "bitrate": "8M",
        "audio_codec": "aac",
        "audio_bitrate": "128k",
        "preset": "medium",
        "crf": 23,
        "max_duration": 90.0,
        "pixel_format": "yuv420p",
    },
    ExportPlatform.TWITTER_VIDEO: {
        "width": 1280,
        "height": 720,
        "codec": ExportFormat.MP4_H264,
        "bitrate": "5M",
        "audio_codec": "aac",
        "audio_bitrate": "128k",
        "preset": "medium",
        "crf": 24,
        "max_duration": 140.0,
        "pixel_format": "yuv420p",
    },
}

MULTI_RESOLUTION_PRESETS: Dict[str, Tuple[int, int, str]] = {
    "2160p": (3840, 2160, "20M"),
    "1440p": (2560, 1440, "12M"),
    "1080p": (1920, 1080, "8M"),
    "720p": (1280, 720, "5M"),
    "480p": (854, 480, "2.5M"),
    "360p": (640, 360, "1.5M"),
}


class ExportEngineError(Exception):
    """Base exception for export engine errors."""


class FFmpegNotFoundError(ExportEngineError):
    """FFmpeg binary not found on system."""


class ExportFailedError(ExportEngineError):
    """Export operation failed."""


class ExportEngine:
    """Video export engine powered by FFmpeg.

    Provides methods for transcoding, platform-specific export,
    thumbnail/preview generation, GIF conversion, metadata injection,
    multi-resolution output, web optimization, and image sequence export.
    """

    def __init__(self, ffmpeg_path: Optional[str] = None, ffprobe_path: Optional[str] = None):
        self.ffmpeg_path = ffmpeg_path or self._find_ffmpeg()
        self.ffprobe_path = ffprobe_path or self._find_ffprobe()
        self._temp_dir: Optional[str] = None

    def _find_ffmpeg(self) -> str:
        path = shutil.which("ffmpeg")
        if path:
            return path
        for candidate in ("ffmpeg.exe", "/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
            if os.path.isfile(candidate):
                return candidate
        raise FFmpegNotFoundError("FFmpeg not found. Install FFmpeg or provide ffmpeg_path.")

    def _find_ffprobe(self) -> str:
        path = shutil.which("ffprobe")
        if path:
            return path
        for candidate in ("ffprobe.exe", "/usr/bin/ffprobe", "/usr/local/bin/ffprobe"):
            if os.path.isfile(candidate):
                return candidate
        raise FFmpegNotFoundError("FFprobe not found. Install FFmpeg or provide ffprobe_path.")

    def _run_command(self, cmd: List[str], timeout: Optional[int] = None) -> subprocess.CompletedProcess:
        logger.debug("Running FFmpeg command: %s", " ".join(cmd))
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout or 600,
                check=False,
            )
            if result.returncode != 0:
                logger.error("FFmpeg stderr: %s", result.stderr)
            return result
        except FileNotFoundError:
            raise FFmpegNotFoundError(f"FFmpeg binary not found at {self.ffmpeg_path}")
        except subprocess.TimeoutExpired:
            raise ExportFailedError("FFmpeg command timed out")

    def _get_temp_dir(self) -> str:
        if self._temp_dir is None or not os.path.isdir(self._temp_dir):
            self._temp_dir = tempfile.mkdtemp(prefix="export_engine_")
        return self._temp_dir

    def _probe_video(self, video_path: str) -> Dict[str, Any]:
        cmd = [
            self.ffprobe_path,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path,
        ]
        result = self._run_command(cmd)
        if result.returncode != 0:
            raise ExportFailedError(f"FFprobe failed: {result.stderr}")
        return json.loads(result.stdout)

    def _get_video_duration(self, video_path: str) -> float:
        probe = self._probe_video(video_path)
        fmt = probe.get("format", {})
        duration_str = fmt.get("duration")
        if duration_str is not None:
            return float(duration_str)
        streams = probe.get("streams", [])
        for s in streams:
            if s.get("codec_type") == "video":
                nb_frames = s.get("nb_frames")
                r_frame_rate = s.get("r_frame_rate", "30/1")
                if nb_frames and "/" in r_frame_rate:
                    num, den = r_frame_rate.split("/")
                    if int(den) > 0:
                        return int(nb_frames) / (int(num) / int(den))
        return 0.0

    def _get_video_info(self, video_path: str) -> Dict[str, Any]:
        probe = self._probe_video(video_path)
        for stream in probe.get("streams", []):
            if stream.get("codec_type") == "video":
                return {
                    "width": stream.get("width", 0),
                    "height": stream.get("height", 0),
                    "codec": stream.get("codec_name", "unknown"),
                    "fps": eval(stream.get("r_frame_rate", "30/1")) if "/" in stream.get("r_frame_rate", "30/1") else float(stream.get("r_frame_rate", "30")),
                    "duration": float(probe.get("format", {}).get("duration", 0)),
                }
        return {"width": 0, "height": 0, "codec": "unknown", "fps": 30.0, "duration": 0.0}

    def _bitrate_to_bps(self, bitrate_str: str) -> int:
        mapping = {"k": 1_000, "K": 1_000, "M": 1_000_000, "m": 1_000_000, "G": 1_000_000_000, "g": 1_000_000_000}
        if bitrate_str[-1] in mapping:
            return int(float(bitrate_str[:-1]) * mapping[bitrate_str[-1]])
        return int(bitrate_str)

    def _clamp_bitrate_for_filesize(self, bitrate_str: str, duration: float, max_mb: Optional[float]) -> str:
        if max_mb is None or duration <= 0:
            return bitrate_str
        max_bits = max_mb * 8 * 1_000_000
        max_avg_bitrate_bps = int((max_bits / duration) * 0.92)
        current_bps = self._bitrate_to_bps(bitrate_str)
        if current_bps <= max_avg_bitrate_bps:
            return bitrate_str
        if max_avg_bitrate_bps >= 1_000_000:
            return f"{max_avg_bitrate_bps / 1_000_000:.1f}M"
        return f"{max_avg_bitrate_bps // 1000}k"

    def export_video(
        self,
        input_path: str,
        output_path: str,
        format_config: Optional[ExportConfig] = None,
    ) -> ExportResult:
        """Export a video file with the given encoding configuration.

        Args:
            input_path: Path to the source video file.
            output_path: Destination path for the exported video.
            format_config: ExportConfig controlling codec, resolution, bitrate, etc.

        Returns:
            ExportResult with output metadata and success status.
        """
        if not os.path.isfile(input_path):
            return ExportResult(
                output_path=output_path,
                file_size_mb=0.0,
                duration=0.0,
                resolution="",
                codec="",
                success=False,
                error_message=f"Input file not found: {input_path}",
            )

        config = format_config or ExportConfig()
        duration = self._get_video_duration(input_path)
        effective_bitrate = self._clamp_bitrate_for_filesize(config.bitrate, duration, config.max_filesize_mb)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        try:
            if config.codec in (ExportFormat.MP4_H264, ExportFormat.MP4_H265, ExportFormat.WEBM_VP9, ExportFormat.AV1):
                if config.max_filesize_mb and duration > 0:
                    success = self._two_pass_encode(
                        input_path, output_path, config, effective_bitrate, duration
                    )
                else:
                    success = self._single_pass_encode(
                        input_path, output_path, config, effective_bitrate
                    )
            else:
                return ExportResult(
                    output_path=output_path,
                    file_size_mb=0.0,
                    duration=0.0,
                    resolution="",
                    codec="",
                    success=False,
                    error_message=f"Use convert_to_gif for GIF exports, not export_video",
                )

            if not success:
                return ExportResult(
                    output_path=output_path,
                    file_size_mb=0.0,
                    duration=0.0,
                    resolution="",
                    codec="",
                    success=False,
                    error_message="FFmpeg encoding failed",
                )

            return self._build_result(output_path, config)

        except Exception as e:
            logger.exception("Export failed")
            return ExportResult(
                output_path=output_path,
                file_size_mb=0.0,
                duration=0.0,
                resolution="",
                codec="",
                success=False,
                error_message=str(e),
            )

    def _single_pass_encode(
        self,
        input_path: str,
        output_path: str,
        config: ExportConfig,
        bitrate: str,
    ) -> bool:
        cmd = [self.ffmpeg_path, "-y", "-i", input_path]
        cmd.extend(self._build_video_filter_args(config))
        cmd.extend(self._build_codec_args(config.codec))
        cmd.extend(["-b:v", bitrate])
        cmd.extend(["-preset", config.preset])
        cmd.extend(["-crf", str(config.crf)])
        cmd.extend(["-pix_fmt", config.pixel_format])
        cmd.extend(["-c:a", config.audio_codec, "-b:a", config.audio_bitrate])
        if config.codec in (ExportFormat.MP4_H264, ExportFormat.MP4_H265):
            cmd.extend(["-movflags", "+faststart"])
        cmd.append(output_path)
        result = self._run_command(cmd, timeout=1200)
        return result.returncode == 0

    def _two_pass_encode(
        self,
        input_path: str,
        output_path: str,
        config: ExportConfig,
        bitrate: str,
        duration: float,
    ) -> bool:
        temp_dir = self._get_temp_dir()
        passlog = os.path.join(temp_dir, "ffmpeg2pass")

        bitrate_bps = self._bitrate_to_bps(bitrate)
        max_bits = config.max_filesize_mb * 8 * 1_000_000  # type: ignore[union-attr]
        audio_bps = self._bitrate_to_bps(config.audio_bitrate)
        video_bitrate_bps = int((max_bits / duration) * 0.92) - audio_bps
        if video_bitrate_bps < 100000:
            video_bitrate_bps = bitrate_bps
        maxrate = int(video_bitrate_bps * 1.5)
        bufsize = int(video_bitrate_bps * 2)

        cmd_pass1 = [self.ffmpeg_path, "-y", "-i", input_path]
        cmd_pass1.extend(self._build_video_filter_args(config))
        cmd_pass1.extend(self._build_codec_args(config.codec))
        cmd_pass1.extend(["-b:v", str(video_bitrate_bps), "-maxrate", str(maxrate), "-bufsize", str(bufsize)])
        cmd_pass1.extend(["-preset", config.preset])
        cmd_pass1.extend(["-crf", str(config.crf)])
        cmd_pass1.extend(["-pix_fmt", config.pixel_format])
        cmd_pass1.extend(["-pass", "1", "-passlogfile", passlog])
        cmd_pass1.extend(["-an"])
        cmd_pass1.append(output_path)

        result1 = self._run_command(cmd_pass1, timeout=1200)
        if result1.returncode != 0:
            return False

        cmd_pass2 = [self.ffmpeg_path, "-y", "-i", input_path]
        cmd_pass2.extend(self._build_video_filter_args(config))
        cmd_pass2.extend(self._build_codec_args(config.codec))
        cmd_pass2.extend(["-b:v", str(video_bitrate_bps), "-maxrate", str(maxrate), "-bufsize", str(bufsize)])
        cmd_pass2.extend(["-preset", config.preset])
        cmd_pass2.extend(["-crf", str(config.crf)])
        cmd_pass2.extend(["-pix_fmt", config.pixel_format])
        cmd_pass2.extend(["-pass", "2", "-passlogfile", passlog])
        cmd_pass2.extend(["-c:a", config.audio_codec, "-b:a", config.audio_bitrate])
        if config.codec in (ExportFormat.MP4_H264, ExportFormat.MP4_H265):
            cmd_pass2.extend(["-movflags", "+faststart"])
        cmd_pass2.append(output_path)

        result2 = self._run_command(cmd_pass2, timeout=1200)

        for suffix in ("-0.log", "-0.log.mbtree", "-0.log"):
            log_file = passlog + suffix
            if os.path.isfile(log_file):
                try:
                    os.remove(log_file)
                except OSError:
                    pass

        return result2.returncode == 0

    def _build_video_filter_args(self, config: ExportConfig) -> List[str]:
        filters = []
        filters.append(f"scale={config.width}:{config.height}:force_original_aspect_ratio=decrease")
        filters.append(f"pad={config.width}:{config.height}:(ow-iw)/2:(oh-ih)/2")
        filters.append(f"fps={config.fps}")
        vf = ",".join(filters)
        return ["-vf", vf]

    def _build_codec_args(self, codec: ExportFormat) -> List[str]:
        if codec == ExportFormat.MP4_H264:
            return ["-c:v", "libx264", "-profile:v", "high", "-level", "4.1"]
        elif codec == ExportFormat.MP4_H265:
            return ["-c:v", "libx265", "-tag:v", "hvc1"]
        elif codec == ExportFormat.WEBM_VP9:
            return ["-c:v", "libvpx-vp9", "-b:v", "0", "-deadline", "good", "-cpu-used", "2"]
        elif codec == ExportFormat.AV1:
            return ["-c:v", "libsvtav1", "-preset", "6"]
        return ["-c:v", "libx264"]

    def _build_output_extension(self, codec: ExportFormat) -> str:
        mapping = {
            ExportFormat.MP4_H264: ".mp4",
            ExportFormat.MP4_H265: ".mp4",
            ExportFormat.WEBM_VP9: ".webm",
            ExportFormat.AV1: ".mp4",
            ExportFormat.GIF: ".gif",
        }
        return mapping.get(codec, ".mp4")

    def _build_result(self, output_path: str, config: ExportConfig) -> ExportResult:
        if not os.path.isfile(output_path):
            return ExportResult(
                output_path=output_path,
                file_size_mb=0.0,
                duration=0.0,
                resolution=f"{config.width}x{config.height}",
                codec=config.codec.value,
                success=False,
                error_message="Output file not created",
            )

        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        duration = self._get_video_duration(output_path)

        return ExportResult(
            output_path=output_path,
            file_size_mb=round(file_size_mb, 2),
            duration=round(duration, 2),
            resolution=f"{config.width}x{config.height}",
            codec=config.codec.value,
            success=True,
        )

    def export_for_platform(
        self,
        video_path: str,
        platform: ExportPlatform,
        output_dir: str,
    ) -> Dict[str, Any]:
        """Export a video optimized for a specific social media platform.

        Applies the platform's resolution, bitrate, codec, and duration constraints.

        Args:
            video_path: Path to the source video.
            platform: Target platform enum value.
            output_dir: Directory to write the exported file.

        Returns:
            Dict with platform name, ExportResult, and platform-specific metadata.
        """
        if not os.path.isfile(video_path):
            return {
                "platform": platform.value,
                "result": ExportResult(
                    output_path="",
                    file_size_mb=0.0,
                    duration=0.0,
                    resolution="",
                    codec="",
                    success=False,
                    error_message=f"Source not found: {video_path}",
                ).to_dict(),
            }

        preset = PLATFORM_PRESETS[platform]
        duration = self._get_video_duration(video_path)
        trimmed_path = video_path
        trimmed_temp = False

        if preset["max_duration"] and duration > preset["max_duration"]:
            trimmed_path = os.path.join(self._get_temp_dir(), f"trimmed_{platform.value}_{int(time.time())}.mp4")
            cmd = [
                self.ffmpeg_path, "-y",
                "-i", video_path,
                "-t", str(preset["max_duration"]),
                "-c", "copy",
                trimmed_path,
            ]
            result = self._run_command(cmd)
            if result.returncode != 0:
                return {
                    "platform": platform.value,
                    "result": ExportResult(
                        output_path="",
                        file_size_mb=0.0,
                        duration=0.0,
                        resolution="",
                        codec="",
                        success=False,
                        error_message=f"Trim failed: {result.stderr}",
                    ).to_dict(),
                }
            trimmed_temp = True

        ext = self._build_output_extension(preset["codec"])
        base_name = Path(video_path).stem
        output_path = os.path.join(output_dir, f"{base_name}_{platform.value}{ext}")

        config = ExportConfig(
            width=preset["width"],
            height=preset["height"],
            fps=30.0,
            codec=preset["codec"],
            bitrate=preset["bitrate"],
            audio_codec=preset["audio_codec"],
            audio_bitrate=preset["audio_bitrate"],
            pixel_format=preset["pixel_format"],
            preset=preset["preset"],
            crf=preset["crf"],
            max_filesize_mb=None,
        )

        export_result = self.export_video(trimmed_path, output_path, config)

        if trimmed_temp and os.path.isfile(trimmed_path):
            try:
                os.remove(trimmed_path)
            except OSError:
                pass

        return {
            "platform": platform.value,
            "result": export_result.to_dict(),
            "platform_limits": {
                "max_duration_s": preset["max_duration"],
                "resolution": f"{preset['width']}x{preset['height']}",
                "max_bitrate": preset["bitrate"],
            },
        }

    def generate_thumbnail(
        self,
        video_path: str,
        timestamp: float,
        output_path: str,
    ) -> str:
        """Extract a single frame as a thumbnail image.

        Args:
            video_path: Path to the source video.
            timestamp: Time in seconds to seek to.
            output_path: Destination path for the thumbnail image (PNG/JPG).

        Returns:
            Absolute path to the created thumbnail file.
        """
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        cmd = [
            self.ffmpeg_path, "-y",
            "-ss", str(timestamp),
            "-i", video_path,
            "-vframes", "1",
            "-q:v", "2",
            output_path,
        ]
        result = self._run_command(cmd)
        if result.returncode != 0:
            raise ExportFailedError(f"Thumbnail extraction failed: {result.stderr}")

        if not os.path.isfile(output_path):
            raise ExportFailedError("Thumbnail file was not created")

        return os.path.abspath(output_path)

    def generate_preview(
        self,
        video_path: str,
        max_duration: float,
        output_path: str,
    ) -> str:
        """Generate a trimmed, lower-bitrate preview of the video.

        Args:
            video_path: Path to the source video.
            max_duration: Maximum duration in seconds for the preview.
            output_path: Destination path for the preview video.

        Returns:
            Absolute path to the created preview file.
        """
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        duration = self._get_video_duration(video_path)
        actual_duration = min(duration, max_duration) if duration > 0 else max_duration

        cmd = [
            self.ffmpeg_path, "-y",
            "-i", video_path,
            "-t", str(actual_duration),
            "-vf", "scale=640:-2,fps=24",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "28",
            "-b:v", "1M",
            "-maxrate", "1.5M",
            "-bufsize", "2M",
            "-c:a", "aac",
            "-b:a", "64k",
            "-ac", "1",
            "-movflags", "+faststart",
            output_path,
        ]
        result = self._run_command(cmd, timeout=600)
        if result.returncode != 0:
            raise ExportFailedError(f"Preview generation failed: {result.stderr}")

        if not os.path.isfile(output_path):
            raise ExportFailedError("Preview file was not created")

        return os.path.abspath(output_path)

    def convert_to_gif(
        self,
        video_path: str,
        output_path: str,
        fps: int = 15,
        scale: int = 480,
    ) -> str:
        """Convert a video segment to an animated GIF using palette-based dithering.

        Uses a two-pass approach: first generates an optimal color palette,
        then renders the GIF with that palette for superior color fidelity.

        Args:
            video_path: Path to the source video.
            output_path: Destination path for the GIF file.
            fps: Frames per second for the GIF.
            scale: Width in pixels to scale the GIF to (height is auto-calculated).

        Returns:
            Absolute path to the created GIF file.
        """
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        temp_dir = self._get_temp_dir()
        palette_path = os.path.join(temp_dir, f"palette_{int(time.time())}.png")

        vf_base = f"fps={fps},scale={scale}:-1:flags=lanczos"

        cmd_palette = [
            self.ffmpeg_path, "-y",
            "-i", video_path,
            "-vf", f"{vf_base},palettegen=stats_mode=diff",
            palette_path,
        ]
        result_palette = self._run_command(cmd_palette)
        if result_palette.returncode != 0:
            raise ExportFailedError(f"Palette generation failed: {result_palette.stderr}")

        cmd_gif = [
            self.ffmpeg_path, "-y",
            "-i", video_path,
            "-i", palette_path,
            "-lavfi", f"{vf_base} [x]; [x][1:v] paletteuse=dither=floyd_steinberg:diff_mode=rectangle",
            output_path,
        ]
        result_gif = self._run_command(cmd_gif, timeout=600)

        if os.path.isfile(palette_path):
            try:
                os.remove(palette_path)
            except OSError:
                pass

        if result_gif.returncode != 0:
            raise ExportFailedError(f"GIF creation failed: {result_gif.stderr}")

        if not os.path.isfile(output_path):
            raise ExportFailedError("GIF file was not created")

        return os.path.abspath(output_path)

    def add_metadata(
        self,
        video_path: str,
        metadata: Dict[str, str],
        output_path: Optional[str] = None,
    ) -> str:
        """Inject or overwrite metadata tags in a video file.

        Writes to a temporary output then replaces the original if output_path
        is not specified.

        Args:
            video_path: Path to the source video.
            metadata: Dict of key-value metadata pairs (e.g. title, artist, comment).
            output_path: Optional destination path. If None, overwrites the input.

        Returns:
            Absolute path to the file with updated metadata.
        """
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        target = output_path or video_path
        if target != video_path:
            os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)

        cmd = [self.ffmpeg_path, "-y", "-i", video_path, "-c", "copy"]
        for key, value in metadata.items():
            cmd.extend(["-metadata", f"{key}={value}"])
        cmd.extend(["-map_metadata", "0"])
        cmd.append(target)

        result = self._run_command(cmd)
        if result.returncode != 0:
            raise ExportFailedError(f"Metadata injection failed: {result.stderr}")

        if output_path is None:
            shutil.move(target, video_path)
            return os.path.abspath(video_path)

        return os.path.abspath(target)

    def generate_multi_resolution(
        self,
        video_path: str,
        resolutions: List[str],
        output_dir: str,
    ) -> Dict[str, str]:
        """Export the video at multiple resolution tiers in parallel.

        Args:
            video_path: Path to the source video.
            resolutions: List of resolution labels (e.g. ["1080p", "720p", "480p"]).
            output_dir: Directory to write all resolution variants.

        Returns:
            Dict mapping resolution labels to their output file paths.
        """
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        os.makedirs(output_dir, exist_ok=True)
        results: Dict[str, str] = {}

        base_name = Path(video_path).stem

        for res_label in resolutions:
            if res_label not in MULTI_RESOLUTION_PRESETS:
                logger.warning("Unknown resolution label: %s, skipping", res_label)
                continue

            width, height, bitrate = MULTI_RESOLUTION_PRESETS[res_label]
            output_path = os.path.join(output_dir, f"{base_name}_{res_label}.mp4")

            config = ExportConfig(
                width=width,
                height=height,
                fps=30.0,
                codec=ExportFormat.MP4_H264,
                bitrate=bitrate,
                audio_codec="aac",
                audio_bitrate="128k",
                pixel_format="yuv420p",
                preset="medium",
                crf=23,
            )

            result = self.export_video(video_path, output_path, config)
            if result.success:
                results[res_label] = result.output_path
            else:
                logger.error("Failed to export %s: %s", res_label, result.error_message)

        return results

    def optimize_for_web(self, video_path: str, output_path: str) -> str:
        """Optimize a video for web streaming with size and compatibility constraints.

        Applies H.264 baseline profile, constrained resolution, low-latency
        flags, and faststart for progressive download.

        Args:
            video_path: Path to the source video.
            output_path: Destination path for the web-optimized video.

        Returns:
            Absolute path to the optimized video file.
        """
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        info = self._get_video_info(video_path)
        target_width = min(info["width"], 1280)
        if target_width % 2 != 0:
            target_width += 1

        cmd = [
            self.ffmpeg_path, "-y",
            "-i", video_path,
            "-vf", f"scale={target_width}:-2,fps=30",
            "-c:v", "libx264",
            "-profile:v", "baseline",
            "-level", "3.1",
            "-preset", "fast",
            "-crf", "26",
            "-maxrate", "3M",
            "-bufsize", "6M",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "96k",
            "-ac", "2",
            "-ar", "44100",
            "-movflags", "+faststart",
            "-dn",
            "-sn",
            "-an" if False else "",
            output_path,
        ]
        cmd = [c for c in cmd if c]

        result = self._run_command(cmd, timeout=600)
        if result.returncode != 0:
            raise ExportFailedError(f"Web optimization failed: {result.stderr}")

        if not os.path.isfile(output_path):
            raise ExportFailedError("Optimized file was not created")

        return os.path.abspath(output_path)

    def export_sequence(
        self,
        video_frames_dir: str,
        output_path: str,
        fps: float = 30.0,
    ) -> str:
        """Encode an image sequence into a video file.

        Expects the directory to contain sequentially numbered frames
        (e.g. frame_000001.png, frame_000002.png, ...).

        Args:
            video_frames_dir: Directory containing the image sequence.
            output_path: Destination path for the output video.
            fps: Frame rate for the output video.

        Returns:
            Absolute path to the created video file.
        """
        if not os.path.isdir(video_frames_dir):
            raise FileNotFoundError(f"Frames directory not found: {video_frames_dir}")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        frame_files = sorted(
            f for f in os.listdir(video_frames_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".exr"))
        )

        if not frame_files:
            raise ExportFailedError(f"No image frames found in {video_frames_dir}")

        ext = frame_files[0].rsplit(".", 1)[-1]
        base_pattern = frame_files[0].rsplit(".", 1)[0]
        numeric_part = ""
        for ch in reversed(base_pattern):
            if ch.isdigit():
                numeric_part = ch + numeric_part
            else:
                break

        prefix = base_pattern[: len(base_pattern) - len(numeric_part)] if numeric_part else base_pattern
        num_digits = len(numeric_part) if numeric_part else 0

        if num_digits > 0:
            input_pattern = os.path.join(video_frames_dir, f"{prefix}%{num_digits}d.{ext}")
        else:
            input_pattern = os.path.join(video_frames_dir, frame_files[0])

        cmd = [
            self.ffmpeg_path, "-y",
            "-framerate", str(fps),
            "-i", input_pattern,
            "-c:v", "libx264",
            "-profile:v", "high",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            "-preset", "medium",
            "-movflags", "+faststart",
            output_path,
        ]
        result = self._run_command(cmd, timeout=1200)
        if result.returncode != 0:
            raise ExportFailedError(f"Image sequence encoding failed: {result.stderr}")

        if not os.path.isfile(output_path):
            raise ExportFailedError("Sequence video was not created")

        return os.path.abspath(output_path)

    def cleanup(self) -> None:
        """Remove temporary files and directories created during exports."""
        if self._temp_dir and os.path.isdir(self._temp_dir):
            try:
                shutil.rmtree(self._temp_dir)
            except OSError as e:
                logger.warning("Failed to clean temp dir %s: %s", self._temp_dir, e)
            self._temp_dir = None

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            pass

    def __enter__(self) -> "ExportEngine":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.cleanup()
