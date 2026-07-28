"""Main configuration module for the video editing engine.

Provides dataclasses for all settings, platform presets, resolution presets,
enum definitions, and load/save functionality for YAML/JSON configs with
environment variable overrides for CI/CD pipelines.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields, asdict
from enum import Enum, auto
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TransitionType(Enum):
    """Available transition types between video clips."""
    CUT = "cut"
    CROSSFADE = "crossfade"
    FADE_BLACK = "fade_black"
    FADE_WHITE = "fade_white"
    WIPE_LEFT = "wipe_left"
    WIPE_RIGHT = "wipe_right"
    WIPE_UP = "wipe_up"
    WIPE_DOWN = "wipe_down"
    DISSOLVE = "dissolve"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    SLIDE_LEFT = "slide_left"
    SLIDE_RIGHT = "slide_right"
    SLIDE_UP = "slide_up"
    SLIDE_DOWN = "slide_down"
    PUSH_LEFT = "push_left"
    PUSH_RIGHT = "push_right"
    SPIN_CW = "spin_cw"
    SPIN_CCW = "spin_ccw"
    PIXELIZE = "pixelize"
    GLITCH = "glitch"
    GLASS = "glass"
    MORPH = "morph"
    BLUR = "blur"
    FILM_BURN = "film_burn"
    LIGHT_LEAK = "light_leak"


class TextAnimation(Enum):
    """Available text animation styles."""
    NONE = "none"
    FADE_IN = "fade_in"
    FADE_OUT = "fade_out"
    FADE_IN_OUT = "fade_in_out"
    TYPEWRITER = "typewriter"
    SLIDE_LEFT = "slide_left"
    SLIDE_RIGHT = "slide_right"
    SLIDE_UP = "slide_up"
    SLIDE_DOWN = "slide_down"
    BOUNCE = "bounce"
    ELASTIC = "elastic"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    SPIN_IN = "spin_in"
    WAVE = "wave"
    GLITCH = "glitch"
    POP = "pop"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    BLUR_IN = "blur_in"
    BLUR_OUT = "blur_out"
    FLIP_H = "flip_h"
    FLIP_V = "flip_v"
    JIGGLE = "jiggle"


class MotionEffect(Enum):
    """Available motion effects for clips."""
    NONE = "none"
    PAN_LEFT = "pan_left"
    PAN_RIGHT = "pan_right"
    PAN_UP = "pan_up"
    PAN_DOWN = "pan_down"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    KEN_BURNS = "ken_burns"
    SHAKE = "shake"
    FLOAT = "float"
    PULSE = "pulse"
    ROTATE_CW = "rotate_cw"
    ROTATE_CCW = "rotate_ccw"
    PARALLAX = "parallax"
    DOLLY_IN = "dolly_in"
    DOLLY_OUT = "dolly_out"
    WHIP_PAN = "whip_pan"


class ExportPlatform(Enum):
    """Target export platforms with platform-specific defaults."""
    FACEBOOK_REELS = "facebook_reels"
    YOUTUBE_SHORTS = "youtube_shorts"
    TIKTOK = "tiktok"
    INSTAGRAM_REELS = "instagram_reels"
    INSTAGRAM_STORY = "instagram_story"
    CUSTOM = "custom"


class ColorSpace(Enum):
    """Supported color spaces."""
    SRGB = "srgb"
    REC709 = "rec709"
    REC2020 = "rec2020"
    DISPLAY_P3 = "display_p3"


class AspectRatio(Enum):
    """Common aspect ratios."""
    PORTRAIT_9_16 = "9:16"
    SQUARE_1_1 = "1:1"
    LANDSCAPE_16_9 = "16:9"
    PORTRAIT_4_5 = "4:5"
    CINEMA_21_9 = "21:9"


# ---------------------------------------------------------------------------
# Dataclass settings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Resolution:
    """Video resolution settings."""
    width: int = 1080
    height: int = 1920
    aspect_ratio: AspectRatio = AspectRatio.PORTRAIT_9_16

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"Resolution dimensions must be positive, got {self.width}x{self.height}"
            )
        if self.width % 2 != 0 or self.height % 2 != 0:
            raise ValueError(
                f"Resolution dimensions must be even, got {self.width}x{self.height}"
            )

    @property
    def megapixels(self) -> float:
        """Return the resolution in megapixels."""
        return (self.width * self.height) / 1_000_000

    def rotated(self) -> Resolution:
        """Return the resolution with width and height swapped."""
        return Resolution(width=self.height, height=self.width, aspect_ratio=self.aspect_ratio)


@dataclass(frozen=True)
class CropAspect:
    """Crop aspect ratio configuration."""
    ratio: str = "9:16"
    anchor_x: float = 0.5
    anchor_y: float = 0.5

    def __post_init__(self) -> None:
        parts = self.ratio.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid aspect ratio format '{self.ratio}', expected 'W:H'")
        try:
            w, h = float(parts[0]), float(parts[1])
        except ValueError as exc:
            raise ValueError(f"Invalid aspect ratio values in '{self.ratio}'") from exc
        if w <= 0 or h <= 0:
            raise ValueError(f"Aspect ratio values must be positive, got {self.ratio}")
        if not (0.0 <= self.anchor_x <= 1.0):
            raise ValueError(f"anchor_x must be in [0, 1], got {self.anchor_x}")
        if not (0.0 <= self.anchor_y <= 1.0):
            raise ValueError(f"anchor_y must be in [0, 1], got {self.anchor_y}")


@dataclass(frozen=True)
class AudioSettings:
    """Audio processing settings."""
    sample_rate: int = 44100
    channels: int = 2
    bitrate: str = "192k"
    codec: str = "aac"
    volume: float = 1.0
    normalize: bool = True
    fade_in_ms: int = 0
    fade_out_ms: int = 0

    def __post_init__(self) -> None:
        if self.sample_rate not in (8000, 11025, 16000, 22050, 44100, 48000, 96000):
            raise ValueError(f"Unsupported sample rate: {self.sample_rate}")
        if self.channels not in (1, 2, 6):
            raise ValueError(f"Unsupported channel count: {self.channels}")
        if not (0.0 <= self.volume <= 2.0):
            raise ValueError(f"Volume must be in [0.0, 2.0], got {self.volume}")
        if self.fade_in_ms < 0 or self.fade_out_ms < 0:
            raise ValueError("Fade durations must be non-negative")


@dataclass(frozen=True)
class SubtitleSettings:
    """Subtitle/caption settings."""
    enabled: bool = False
    font_family: str = "Arial"
    font_size: int = 48
    font_color: str = "#FFFFFF"
    background_color: str = "#00000080"
    position: str = "bottom"
    margin_bottom: int = 60
    max_chars_per_line: int = 40
    animation: TextAnimation = TextAnimation.FADE_IN
    word_highlight_color: Optional[str] = None

    def __post_init__(self) -> None:
        if self.font_size <= 0:
            raise ValueError(f"Font size must be positive, got {self.font_size}")
        if self.position not in ("top", "center", "bottom"):
            raise ValueError(f"Invalid position '{self.position}', expected top/center/bottom")


@dataclass(frozen=True)
class ColorGrading:
    """Color grading / correction settings."""
    brightness: float = 0.0
    contrast: float = 1.0
    saturation: float = 1.0
    gamma: float = 1.0
    temperature: float = 0.0
    tint: float = 0.0
    highlights: float = 0.0
    shadows: float = 0.0
    vibrance: float = 0.0
    sharpness: float = 0.0
    vignette: float = 0.0
    lut_path: Optional[str] = None
    color_space: ColorSpace = ColorSpace.SRGB

    def __post_init__(self) -> None:
        ranges = {
            "brightness": (-1.0, 1.0),
            "contrast": (0.0, 3.0),
            "saturation": (0.0, 3.0),
            "gamma": (0.1, 5.0),
            "temperature": (-1.0, 1.0),
            "tint": (-1.0, 1.0),
            "highlights": (-1.0, 1.0),
            "shadows": (-1.0, 1.0),
            "vibrance": (-1.0, 1.0),
            "sharpness": (0.0, 3.0),
            "vignette": (0.0, 1.0),
        }
        for name, (lo, hi) in ranges.items():
            val = getattr(self, name)
            if not (lo <= val <= hi):
                raise ValueError(f"{name} must be in [{lo}, {hi}], got {val}")


@dataclass(frozen=True)
class MotionEffectSettings:
    """Motion effect configuration for a clip."""
    effect: MotionEffect = MotionEffect.NONE
    intensity: float = 1.0
    duration_sec: Optional[float] = None
    ease_in: bool = True
    ease_out: bool = True

    def __post_init__(self) -> None:
        if not (0.0 <= self.intensity <= 3.0):
            raise ValueError(f"intensity must be in [0.0, 3.0], got {self.intensity}")
        if self.duration_sec is not None and self.duration_sec <= 0:
            raise ValueError(f"duration_sec must be positive, got {self.duration_sec}")


@dataclass(frozen=True)
class ExportFormat:
    """Export output format settings."""
    container: str = "mp4"
    video_codec: str = "h264"
    audio_codec: str = "aac"
    video_bitrate: str = "8M"
    audio_bitrate: str = "192k"
    crf: int = 18
    pixel_format: str = "yuv420p"
    faststart: bool = True

    def __post_init__(self) -> None:
        valid_containers = {"mp4", "mov", "avi", "mkv", "webm"}
        if self.container not in valid_containers:
            raise ValueError(f"container must be one of {valid_containers}, got '{self.container}'")
        if not (0 <= self.crf <= 51):
            raise ValueError(f"CRF must be in [0, 51], got {self.crf}")
        valid_pixel_fmts = {"yuv420p", "yuv422p", "yuv444p", "rgb24"}
        if self.pixel_format not in valid_pixel_fmts:
            raise ValueError(f"pixel_format must be one of {valid_pixel_fmts}, got '{self.pixel_format}'")


@dataclass(frozen=True)
class TransitionSettings:
    """Transition settings between clips."""
    type: TransitionType = TransitionType.CUT
    duration_sec: float = 0.5

    def __post_init__(self) -> None:
        if self.duration_sec < 0:
            raise ValueError(f"Transition duration must be non-negative, got {self.duration_sec}")


@dataclass
class ExportPlatformConfig:
    """Full export configuration for a platform."""
    resolution: Resolution = field(default_factory=Resolution)
    export_format: ExportFormat = field(default_factory=ExportFormat)
    audio: AudioSettings = field(default_factory=AudioSettings)
    subtitles: SubtitleSettings = field(default_factory=SubtitleSettings)
    color_grading: ColorGrading = field(default_factory=ColorGrading)
    default_transition: TransitionSettings = field(default_factory=TransitionSettings)
    max_duration_sec: float = 60.0
    max_file_size_mb: float = 250.0
    target_fps: int = 30

    def __post_init__(self) -> None:
        if self.target_fps not in (24, 25, 30, 60):
            raise ValueError(f"target_fps must be one of 24, 25, 30, 60, got {self.target_fps}")
        if self.max_duration_sec <= 0:
            raise ValueError(f"max_duration_sec must be positive, got {self.max_duration_sec}")
        if self.max_file_size_mb <= 0:
            raise ValueError(f"max_file_size_mb must be positive, got {self.max_file_size_mb}")


# ---------------------------------------------------------------------------
# Resolution presets
# ---------------------------------------------------------------------------

RESOLUTION_PRESETS: dict[str, Resolution] = {
    "vertical_1080p": Resolution(width=1080, height=1920, aspect_ratio=AspectRatio.PORTRAIT_9_16),
    "square_1080p": Resolution(width=1080, height=1080, aspect_ratio=AspectRatio.SQUARE_1_1),
    "vertical_720p": Resolution(width=720, height=1280, aspect_ratio=AspectRatio.PORTRAIT_9_16),
    "landscape_1080p": Resolution(width=1920, height=1080, aspect_ratio=AspectRatio.LANDSCAPE_16_9),
    "vertical_4k": Resolution(width=2160, height=3840, aspect_ratio=AspectRatio.PORTRAIT_9_16),
    "landscape_4k": Resolution(width=3840, height=2160, aspect_ratio=AspectRatio.LANDSCAPE_16_9),
    "portrait_4_5": Resolution(width=1080, height=1350, aspect_ratio=AspectRatio.PORTRAIT_4_5),
}


# ---------------------------------------------------------------------------
# Platform presets
# ---------------------------------------------------------------------------

PLATFORM_PRESETS: dict[ExportPlatform, ExportPlatformConfig] = {
    ExportPlatform.FACEBOOK_REELS: ExportPlatformConfig(
        resolution=RESOLUTION_PRESETS["vertical_1080p"],
        export_format=ExportFormat(container="mp4", video_codec="h264", crf=20),
        audio=AudioSettings(sample_rate=44100, channels=2, bitrate="128k"),
        max_duration_sec=60.0,
        max_file_size_mb=250.0,
        target_fps=30,
    ),
    ExportPlatform.YOUTUBE_SHORTS: ExportPlatformConfig(
        resolution=RESOLUTION_PRESETS["vertical_1080p"],
        export_format=ExportFormat(container="mp4", video_codec="h264", crf=18, video_bitrate="12M"),
        audio=AudioSettings(sample_rate=48000, channels=2, bitrate="256k"),
        max_duration_sec=60.0,
        max_file_size_mb=256.0,
        target_fps=30,
    ),
    ExportPlatform.TIKTOK: ExportPlatformConfig(
        resolution=RESOLUTION_PRESETS["vertical_1080p"],
        export_format=ExportFormat(container="mp4", video_codec="h264", crf=20, video_bitrate="10M"),
        audio=AudioSettings(sample_rate=44100, channels=2, bitrate="128k"),
        max_duration_sec=180.0,
        max_file_size_mb=287.0,
        target_fps=30,
    ),
    ExportPlatform.INSTAGRAM_REELS: ExportPlatformConfig(
        resolution=RESOLUTION_PRESETS["vertical_1080p"],
        export_format=ExportFormat(container="mp4", video_codec="h264", crf=20),
        audio=AudioSettings(sample_rate=44100, channels=2, bitrate="128k"),
        max_duration_sec=90.0,
        max_file_size_mb=250.0,
        target_fps=30,
    ),
    ExportPlatform.INSTAGRAM_STORY: ExportPlatformConfig(
        resolution=RESOLUTION_PRESETS["vertical_1080p"],
        export_format=ExportFormat(container="mp4", video_codec="h264", crf=20),
        audio=AudioSettings(sample_rate=44100, channels=2, bitrate="128k"),
        max_duration_sec=15.0,
        max_file_size_mb=250.0,
        target_fps=30,
    ),
    ExportPlatform.CUSTOM: ExportPlatformConfig(),
}


# ---------------------------------------------------------------------------
# Main project config
# ---------------------------------------------------------------------------

@dataclass
class ProjectConfig:
    """Top-level project configuration."""
    project_name: str = "untitled_project"
    output_dir: str = "./output"
    temp_dir: str = "./tmp"
    source_dir: str = "./source"
    platform: ExportPlatform = ExportPlatform.CUSTOM
    resolution: Resolution = field(default_factory=lambda: RESOLUTION_PRESETS["vertical_1080p"])
    export_format: ExportFormat = field(default_factory=ExportFormat)
    audio: AudioSettings = field(default_factory=AudioSettings)
    subtitles: SubtitleSettings = field(default_factory=SubtitleSettings)
    color_grading: ColorGrading = field(default_factory=ColorGrading)
    default_transition: TransitionSettings = field(default_factory=TransitionSettings)
    target_fps: int = 30
    max_concurrent_tasks: int = 4
    log_level: str = "INFO"
    enable_gpu_acceleration: bool = False
    custom: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.target_fps not in (24, 25, 30, 60):
            raise ValueError(f"target_fps must be one of 24, 25, 30, 60, got {self.target_fps}")
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}, got '{self.log_level}'")

    @classmethod
    def from_platform(cls, platform: ExportPlatform, **overrides: Any) -> ProjectConfig:
        """Create a config pre-filled with platform-specific defaults.

        Any keyword argument provided will override the platform defaults.
        """
        preset = PLATFORM_PRESETS[platform]
        kwargs: dict[str, Any] = {
            "platform": platform,
            "resolution": preset.resolution,
            "export_format": preset.export_format,
            "audio": preset.audio,
            "subtitles": preset.subtitles,
            "color_grading": preset.color_grading,
            "default_transition": preset.default_transition,
            "target_fps": preset.target_fps,
        }
        kwargs.update(overrides)
        return cls(**kwargs)

    def apply_env_overrides(self) -> ProjectConfig:
        """Return a new config with values overridden by environment variables.

        Environment variable names are prefixed with ``VE_`` (video engine).
        Nested objects use double-underscore separators, e.g.
        ``VE_RESOLUTION__WIDTH=1920``.
        """
        changes: dict[str, Any] = {}
        env_map: dict[str, str] = {
            "VE_PROJECT_NAME": "project_name",
            "VE_OUTPUT_DIR": "output_dir",
            "VE_TEMP_DIR": "temp_dir",
            "VE_SOURCE_DIR": "source_dir",
            "VE_PLATFORM": "platform",
            "VE_TARGET_FPS": "target_fps",
            "VE_MAX_CONCURRENT_TASKS": "max_concurrent_tasks",
            "VE_LOG_LEVEL": "log_level",
            "VE_ENABLE_GPU_ACCELERATION": "enable_gpu_acceleration",
        }
        for env_var, attr in env_map.items():
            val = os.environ.get(env_var)
            if val is None:
                continue
            if attr == "platform":
                changes[attr] = ExportPlatform(val)
            elif attr == "target_fps" or attr == "max_concurrent_tasks":
                changes[attr] = int(val)
            elif attr == "enable_gpu_acceleration":
                changes[attr] = val.lower() in ("1", "true", "yes")
            else:
                changes[attr] = val

        nested_map: dict[str, dict[str, str]] = {
            "VE_RESOLUTION": {"width": "resolution_width", "height": "resolution_height"},
        }
        for prefix, mapping in nested_map.items():
            for suffix, key in mapping.items():
                val = os.environ.get(f"{prefix}__{suffix}")
                if val is not None:
                    changes[key] = int(val)

        if "resolution_width" in changes or "resolution_height" in changes:
            cur = self.resolution
            new_w = changes.pop("resolution_width", cur.width)
            new_h = changes.pop("resolution_height", cur.height)
            changes["resolution"] = Resolution(width=new_w, height=new_h, aspect_ratio=cur.aspect_ratio)

        if not changes:
            return self

        data = asdict(self)
        data.update(changes)
        return ProjectConfig(**data)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _config_to_dict(obj: Any) -> Any:
    """Recursively convert dataclass instances and enums to serializable dicts."""
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, "__dataclass_fields__"):
        result: dict[str, Any] = {}
        for f in fields(obj):
            result[f.name] = _config_to_dict(getattr(obj, f.name))
        return result
    if isinstance(obj, list):
        return [_config_to_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _config_to_dict(v) for k, v in obj.items()}
    return obj


def _dict_to_config(data: dict[str, Any], cls: type) -> Any:
    """Reconstruct a dataclass from a dict, converting enum fields."""
    if not hasattr(cls, "__dataclass_fields__"):
        return data

    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        val = data[f.name]
        field_type = f.type

        # Resolve string type annotations
        if isinstance(field_type, str):
            field_type = _resolve_type_name(field_type, cls)

        if isinstance(field_type, type) and issubclass(field_type, Enum) and isinstance(val, str):
            kwargs[f.name] = field_type(val)
        elif isinstance(field_type, type) and hasattr(field_type, "__dataclass_fields__") and isinstance(val, dict):
            kwargs[f.name] = _dict_to_config(val, field_type)
        else:
            kwargs[f.name] = val

    return cls(**kwargs)


def _resolve_type_name(name: str, parent_cls: type) -> Any:
    """Attempt to resolve a string type annotation from the module or parent scope."""
    import sys
    module = sys.modules.get(parent_cls.__module__, None)
    if module is not None:
        return getattr(module, name, name)
    return name


def save_config(config: ProjectConfig, path: str | Path, fmt: str = "auto") -> Path:
    """Save a ``ProjectConfig`` to a file.

    Args:
        config: The configuration to save.
        path: Destination file path.
        fmt: ``"json"``, ``"yaml"``, or ``"auto"`` (detect from extension).

    Returns:
        The resolved path that was written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _config_to_dict(config)

    if fmt == "auto":
        fmt = path.suffix.lstrip(".").lower()
        if fmt not in ("json", "yaml", "yml"):
            fmt = "json"

    if fmt == "json":
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    elif fmt in ("yaml", "yml"):
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required for YAML output. Install it with: pip install pyyaml"
            ) from exc
        path.write_text(yaml.safe_dump(data, default_flow_style=False, sort_keys=False), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported format '{fmt}'. Use 'json', 'yaml', or 'auto'.")

    return path


def load_config(path: str | Path, fmt: str = "auto") -> ProjectConfig:
    """Load a ``ProjectConfig`` from a JSON or YAML file.

    Args:
        path: Source file path.
        fmt: ``"json"``, ``"yaml"``, or ``"auto"`` (detect from extension).

    Returns:
        A fully hydrated ``ProjectConfig`` instance.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    if fmt == "auto":
        fmt = path.suffix.lstrip(".").lower()
        if fmt not in ("json", "yaml", "yml"):
            fmt = "json"

    raw = path.read_text(encoding="utf-8")

    if fmt == "json":
        data: dict[str, Any] = json.loads(raw)
    elif fmt in ("yaml", "yml"):
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required for YAML input. Install it with: pip install pyyaml"
            ) from exc
        data = yaml.safe_load(raw)
        if not isinstance(data, dict):
            raise ValueError(f"Expected a mapping at the top level of {path}")
    else:
        raise ValueError(f"Unsupported format '{fmt}'. Use 'json', 'yaml', or 'auto'.")

    return _dict_to_config(data, ProjectConfig)
