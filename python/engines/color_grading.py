"""
Color Grading Engine for Video Editing.

Provides comprehensive color grading capabilities including brightness, contrast,
saturation, temperature, gamma, color curves, LUT application, cinematic looks,
sharpening, noise reduction, and automatic color correction.

Uses FFmpeg filter complex for efficient processing and numpy for LUT generation
and histogram analysis.
"""

import os
import struct
import subprocess
import json
import math
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any

import numpy as np


class ColorPreset(Enum):
    """Predefined color grading presets."""
    CINEMATIC = "cinematic"
    WARM = "warm"
    COOL = "cool"
    VINTAGE = "vintage"
    DRAMATIC = "dramatic"
    VIVID = "vivid"
    DESATURATED = "desaturated"
    HIGH_CONTRAST = "high_contrast"
    FILM_NOIR = "film_noir"
    SUNSET = "sunset"
    MORNING = "morning"
    NIGHT = "night"


@dataclass
class ColorGrading:
    """Data class holding all color grading parameters."""
    brightness: float = 0.0
    contrast: float = 1.0
    saturation: float = 1.0
    temperature: float = 0.0
    tint: float = 0.0
    gamma: float = 1.0
    highlights: float = 0.0
    shadows: float = 0.0
    sharpen: float = 0.0
    noise_reduction: float = 0.0
    lut_path: Optional[str] = None
    color_curve: Optional[str] = None
    cinematic_look: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ColorGrading":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class ColorGradingEngine:
    """
    Engine for applying color grading operations to video files.

    Uses FFmpeg for video processing with filter_complex for combining
    multiple filters in a single pass when possible.
    """

    # Preset definitions mapping to ColorGrading parameters
    PRESET_DEFINITIONS: Dict[ColorPreset, ColorGrading] = {
        ColorPreset.CINEMATIC: ColorGrading(
            brightness=-0.05, contrast=1.2, saturation=0.85,
            temperature=5.0, gamma=0.95, highlights=-0.1,
            shadows=0.05, sharpen=0.3
        ),
        ColorPreset.WARM: ColorGrading(
            brightness=0.02, contrast=1.05, saturation=1.1,
            temperature=15.0, tint=2.0, gamma=1.0
        ),
        ColorPreset.COOL: ColorGrading(
            brightness=0.0, contrast=1.05, saturation=0.95,
            temperature=-15.0, tint=-1.0, gamma=1.0
        ),
        ColorPreset.VINTAGE: ColorGrading(
            brightness=0.03, contrast=0.9, saturation=0.7,
            temperature=8.0, gamma=1.1, highlights=0.05,
            shadows=-0.05
        ),
        ColorPreset.DRAMATIC: ColorGrading(
            brightness=-0.08, contrast=1.4, saturation=0.8,
            temperature=0.0, gamma=0.9, highlights=-0.15,
            shadows=-0.1, sharpen=0.5
        ),
        ColorPreset.VIVID: ColorGrading(
            brightness=0.05, contrast=1.15, saturation=1.4,
            temperature=3.0, gamma=1.0, sharpen=0.2
        ),
        ColorPreset.DESATURATED: ColorGrading(
            brightness=0.0, contrast=1.1, saturation=0.3,
            temperature=0.0, gamma=1.0
        ),
        ColorPreset.HIGH_CONTRAST: ColorGrading(
            brightness=-0.03, contrast=1.5, saturation=1.0,
            temperature=0.0, gamma=0.85, highlights=-0.1,
            shadows=-0.15, sharpen=0.3
        ),
        ColorPreset.FILM_NOIR: ColorGrading(
            brightness=-0.1, contrast=1.6, saturation=0.0,
            temperature=0.0, gamma=0.8, highlights=-0.2,
            shadows=-0.2, sharpen=0.6
        ),
        ColorPreset.SUNSET: ColorGrading(
            brightness=0.03, contrast=1.1, saturation=1.2,
            temperature=20.0, tint=5.0, gamma=1.0,
            highlights=-0.05
        ),
        ColorPreset.MORNING: ColorGrading(
            brightness=0.08, contrast=1.0, saturation=1.05,
            temperature=8.0, tint=1.0, gamma=1.05,
            highlights=0.05, shadows=0.03
        ),
        ColorPreset.NIGHT: ColorGrading(
            brightness=-0.15, contrast=1.3, saturation=0.7,
            temperature=-10.0, tint=-2.0, gamma=0.85,
            highlights=-0.2, shadows=-0.15, sharpen=0.4
        ),
    }

    def __init__(self, ffmpeg_path: str = "ffmpeg", ffprobe_path: str = "ffprobe"):
        """
        Initialize the color grading engine.

        Args:
            ffmpeg_path: Path to the FFmpeg executable.
            ffprobe_path: Path to the FFprobe executable.
        """
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path
        self._validate_ffmpeg()

    def _validate_ffmpeg(self) -> None:
        """Verify that FFmpeg is available."""
        try:
            result = subprocess.run(
                [self.ffmpeg_path, "-version"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                raise RuntimeError("FFmpeg is not properly installed.")
        except FileNotFoundError:
            raise RuntimeError(
                f"FFmpeg not found at '{self.ffmpeg_path}'. "
                "Please install FFmpeg or provide the correct path."
            )

    def _probe_video(self, video_path: str) -> Dict[str, Any]:
        """
        Get video metadata using FFprobe.

        Args:
            video_path: Path to the video file.

        Returns:
            Dictionary containing video metadata.
        """
        cmd = [
            self.ffprobe_path, "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"FFprobe failed: {result.stderr}")
        return json.loads(result.stdout)

    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration in seconds."""
        metadata = self._probe_video(video_path)
        return float(metadata.get("format", {}).get("duration", 0))

    def _build_filter_complex(
        self,
        brightness: float = 0.0,
        contrast: float = 1.0,
        saturation: float = 1.0,
        temperature: float = 0.0,
        tint: float = 0.0,
        gamma: float = 1.0,
        highlights: float = 0.0,
        shadows: float = 0.0,
        sharpen: float = 0.0,
        noise_reduction: float = 0.0,
        lut_path: Optional[str] = None,
        color_curve: Optional[str] = None,
    ) -> str:
        """
        Build a combined FFmpeg filter_complex string from individual parameters.

        Args:
            brightness: Brightness offset (-1.0 to 1.0).
            contrast: Contrast multiplier (0.0 to 3.0).
            saturation: Saturation multiplier (0.0 to 3.0).
            temperature: Color temperature shift (-100 to 100).
            tint: Tint shift (-100 to 100).
            gamma: Gamma correction (0.1 to 5.0).
            highlights: Highlights adjustment (-1.0 to 1.0).
            shadows: Shadows adjustment (-1.0 to 1.0).
            sharpen: Sharpening amount (0.0 to 5.0).
            noise_reduction: Noise reduction strength (0.0 to 10.0).
            lut_path: Path to a .cube LUT file.
            color_curve: Curve type name.

        Returns:
            FFmpeg filter_complex string.
        """
        filters: List[str] = []
        current_label = "[0:v]"

        # EQ filter for brightness, contrast, gamma
        needs_eq = (
            brightness != 0.0 or contrast != 1.0 or
            gamma != 1.0 or saturation != 1.0
        )
        if needs_eq:
            eq_parts = []
            if brightness != 0.0:
                eq_parts.append(f"brightness={brightness}")
            if contrast != 1.0:
                eq_parts.append(f"contrast={contrast}")
            if gamma != 1.0:
                eq_parts.append(f"gamma={gamma}")
            if saturation != 1.0:
                eq_parts.append(f"saturation={saturation}")
            eq_filter = ":".join(eq_parts)
            next_label = "[eq_out]"
            filters.append(f"{current_label}eq={eq_filter}{next_label}")
            current_label = next_label

        # Color balance filter for temperature and tint
        needs_colorbalance = temperature != 0.0 or tint != 0.0
        if needs_colorbalance:
            # Temperature: positive = warm (red shift), negative = cool (blue shift)
            # Normalize to -1.0 to 1.0 range (assuming input is -100 to 100)
            temp_norm = max(-1.0, min(1.0, temperature / 100.0))
            tint_norm = max(-1.0, min(1.0, tint / 100.0))

            # Red/cyan balance from temperature
            rc = temp_norm * 0.5
            # Blue/yellow balance from temperature
            bc = -temp_norm * 0.5
            # Green/magenta from tint
            gm = tint_norm * 0.5

            cb_filter = f"colorbalance=rs={rc}:gs={gm}:bs={bc}:rm={rc*0.3}:gm={gm*0.3}:bm={bc*0.3}:rh={rc*0.2}:gh={gm*0.2}:bh={bc*0.2}"
            next_label = "[cb_out]"
            filters.append(f"{current_label}{cb_filter}{next_label}")
            current_label = next_label

        # Curves filter for highlights/shadows and color curves
        needs_curves = highlights != 0.0 or shadows != 0.0 or color_curve is not None
        if needs_curves:
            curve_parts = []
            if color_curve:
                curve_map = {
                    "s_curve": "0/0 0.25/0.20 0.5/0.55 0.75/0.80 1/1",
                    "low_contrast": "0/0 0.5/0.45 1/1",
                    "high_contrast": "0/0 0.25/0.15 0.5/0.55 0.75/0.85 1/1",
                    "filmic": "0/0.02 0.1/0.08 0.2/0.15 0.4/0.38 0.6/0.62 0.8/0.85 1/0.98",
                    "lift_gamma_gain": "0/0.05 0.15/0.12 0.5/0.5 0.85/0.88 1/0.95",
                    "faded": "0/0.05 0.25/0.22 0.5/0.5 0.75/0.78 1/0.95",
                    "crushed_blacks": "0/0 0.1/0.02 0.25/0.15 0.5/0.5 0.75/0.8 1/1",
                    "lifted_blacks": "0/0.08 0.25/0.25 0.5/0.5 0.75/0.75 1/1",
                }
                if color_curve in curve_map:
                    curve_parts.append(f"master={curve_map[color_curve]}")
                else:
                    curve_parts.append(f"master={color_curve}")

            if highlights != 0.0 or shadows != 0.0:
                # Adjust highlights and shadows via the master curve
                # Generate a curve that modifies the high/low regions
                h_shift = highlights * 0.2
                s_shift = shadows * 0.2
                points = []
                for i in range(256):
                    x = i / 255.0
                    y = x
                    # Shadow adjustment (affects lower range)
                    if x < 0.5:
                        shadow_factor = 1.0 - (x / 0.5)
                        y += s_shift * shadow_factor * shadow_factor
                    # Highlight adjustment (affects upper range)
                    if x > 0.5:
                        highlight_factor = (x - 0.5) / 0.5
                        y += h_shift * highlight_factor * highlight_factor
                    y = max(0.0, min(1.0, y))
                    points.append(f"{x:.4f}/{y:.4f}")
                if not curve_parts:
                    curve_parts.append(f"master={' '.join(points)}")

            curves_filter = f"curves={':'.join(curve_parts)}"
            next_label = "[curves_out]"
            filters.append(f"{current_label}{curves_filter}{next_label}")
            current_label = next_label

        # LUT 3D filter
        if lut_path and os.path.isfile(lut_path):
            lut_filter = f"lut3d=file='{lut_path.replace(chr(39), chr(39)+chr(92)+chr(39))}'"
            next_label = "[lut_out]"
            filters.append(f"{current_label}{lut_filter}{next_label}")
            current_label = next_label

        # Sharpen (unsharp mask)
        if sharpen > 0.0:
            # unsharp=luma_msize_x:luma_msize_y:luma_amount:chroma_msize_x:chroma_msize_y:chroma_amount
            amount = min(5.0, sharpen)
            luma_size = 5
            sharp_filter = f"unsharp={luma_size}:{luma_size}:{amount}:{luma_size}:{luma_size}:{amount*0.5}"
            next_label = "[sharp_out]"
            filters.append(f"{current_label}{sharp_filter}{next_label}")
            current_label = next_label

        # Noise reduction (nlmeans)
        if noise_reduction > 0.0:
            strength = min(10.0, noise_reduction)
            # nlmeans=s=h_pca:h=sigma:r=patch_size
            nr_filter = f"nlmeans=s={strength}:p=3:r=9"
            next_label = "[nr_out]"
            filters.append(f"{current_label}{nr_filter}{next_label}")
            current_label = next_label

        if not filters:
            return ""

        # Chain filters together
        filter_complex = ";".join(filters)
        return filter_complex

    def _run_ffmpeg(
        self,
        input_path: str,
        output_path: str,
        filter_complex: str,
        extra_args: Optional[List[str]] = None
    ) -> subprocess.CompletedProcess:
        """
        Execute an FFmpeg command with the given filter complex.

        Args:
            input_path: Input video file path.
            output_path: Output video file path.
            filter_complex: FFmpeg filter_complex string.
            extra_args: Additional FFmpeg arguments.

        Returns:
            CompletedProcess instance.
        """
        cmd = [
            self.ffmpeg_path, "-y",
            "-i", input_path,
        ]

        if filter_complex:
            cmd.extend(["-filter_complex", filter_complex])
            cmd.extend(["-map", "[out]"] if "[out]" in filter_complex else ["-map", "0:v:0", "-map", "0:a?"])

        if extra_args:
            cmd.extend(extra_args)

        # Preserve audio and other streams
        if "-map" not in cmd:
            cmd.extend(["-map", "0"])

        cmd.extend(["-c:v", "libx264", "-preset", "medium", "-crf", "18"])
        cmd.extend(["-c:a", "copy"])
        cmd.append(output_path)

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600
        )
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")
        return result

    def _apply_combined_filters(
        self,
        video_path: str,
        output_path: str,
        grading: ColorGrading
    ) -> str:
        """
        Apply all color grading parameters in a single FFmpeg pass.

        Args:
            video_path: Input video path.
            output_path: Output video path.
            grading: ColorGrading instance with all parameters.

        Returns:
            Path to the output file.
        """
        filter_complex = self._build_filter_complex(
            brightness=grading.brightness,
            contrast=grading.contrast,
            saturation=grading.saturation,
            temperature=grading.temperature,
            tint=grading.tint,
            gamma=grading.gamma,
            highlights=grading.highlights,
            shadows=grading.shadows,
            sharpen=grading.sharpen,
            noise_reduction=grading.noise_reduction,
            lut_path=grading.lut_path,
            color_curve=grading.color_curve,
        )

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # Fix the final output label for -map
        if filter_complex and not filter_complex.rstrip().endswith("[out]"):
            # Rename last output label to [out]
            parts = filter_complex.rsplit("[", 1)
            if len(parts) == 2:
                last_label = parts[1].rstrip("]")
                filter_complex = parts[0] + "[out]"
                # Update map
                cmd = [
                    self.ffmpeg_path, "-y",
                    "-i", video_path,
                    "-filter_complex", filter_complex,
                    "-map", f"[out]", "-map", "0:a?",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                    "-c:a", "copy",
                    output_path
                ]
            else:
                cmd = [
                    self.ffmpeg_path, "-y",
                    "-i", video_path,
                    "-filter_complex", filter_complex,
                    "-map", "0",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                    "-c:a", "copy",
                    output_path
                ]
        elif filter_complex:
            cmd = [
                self.ffmpeg_path, "-y",
                "-i", video_path,
                "-filter_complex", filter_complex,
                "-map", "[out]", "-map", "0:a?",
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-c:a", "copy",
                output_path
            ]
        else:
            # No filters, just copy
            cmd = [
                self.ffmpeg_path, "-y",
                "-i", video_path,
                "-c", "copy",
                output_path
            ]

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")

        return output_path

    # -------------------------------------------------------------------------
    # Individual filter methods
    # -------------------------------------------------------------------------

    def apply_brightness(
        self,
        video_path: str,
        value: float,
        output_path: str
    ) -> str:
        """
        Adjust video brightness.

        Args:
            video_path: Path to the input video file.
            value: Brightness adjustment value (-1.0 to 1.0, 0.0 = no change).
            output_path: Path for the output video file.

        Returns:
            Path to the output file.
        """
        filter_complex = self._build_filter_complex(brightness=value)
        return self._apply_combined_filters(video_path, output_path, ColorGrading(brightness=value))

    def apply_contrast(
        self,
        video_path: str,
        value: float,
        output_path: str
    ) -> str:
        """
        Adjust video contrast.

        Args:
            video_path: Path to the input video file.
            value: Contrast multiplier (0.0 to 3.0, 1.0 = no change).
            output_path: Path for the output video file.

        Returns:
            Path to the output file.
        """
        return self._apply_combined_filters(video_path, output_path, ColorGrading(contrast=value))

    def apply_saturation(
        self,
        video_path: str,
        value: float,
        output_path: str
    ) -> str:
        """
        Adjust video color saturation.

        Args:
            video_path: Path to the input video file.
            value: Saturation multiplier (0.0 to 3.0, 1.0 = no change, 0.0 = grayscale).
            output_path: Path for the output video file.

        Returns:
            Path to the output file.
        """
        return self._apply_combined_filters(video_path, output_path, ColorGrading(saturation=value))

    def apply_temperature(
        self,
        video_path: str,
        warmth: float,
        output_path: str
    ) -> str:
        """
        Adjust video color temperature.

        Args:
            video_path: Path to the input video file.
            warmth: Temperature shift (-100 to 100, 0 = neutral, positive = warm, negative = cool).
            output_path: Path for the output video file.

        Returns:
            Path to the output file.
        """
        return self._apply_combined_filters(video_path, output_path, ColorGrading(temperature=warmth))

    def apply_gamma(
        self,
        video_path: str,
        value: float,
        output_path: str
    ) -> str:
        """
        Apply gamma correction.

        Args:
            video_path: Path to the input video file.
            value: Gamma value (0.1 to 5.0, 1.0 = no change).
            output_path: Path for the output video file.

        Returns:
            Path to the output file.
        """
        return self._apply_combined_filters(video_path, output_path, ColorGrading(gamma=value))

    def apply_color_curve(
        self,
        video_path: str,
        curve_type: str,
        output_path: str
    ) -> str:
        """
        Apply a predefined or custom color curve.

        Args:
            video_path: Path to the input video file.
            curve_type: Curve type name or custom curve string.
                Preset names: s_curve, low_contrast, high_contrast, filmic,
                lift_gamma_gain, faded, crushed_blacks, lifted_blacks.
                Or a custom curve string like "0/0 0.5/0.6 1/1".
            output_path: Path for the output video file.

        Returns:
            Path to the output file.
        """
        return self._apply_combined_filters(
            video_path, output_path, ColorGrading(color_curve=curve_type)
        )

    def apply_lut(
        self,
        video_path: str,
        lut_path: str,
        output_path: str
    ) -> str:
        """
        Apply a 3D LUT file to the video.

        Args:
            video_path: Path to the input video file.
            lut_path: Path to the .cube LUT file.
            output_path: Path for the output video file.

        Returns:
            Path to the output file.

        Raises:
            FileNotFoundError: If the LUT file does not exist.
        """
        if not os.path.isfile(lut_path):
            raise FileNotFoundError(f"LUT file not found: {lut_path}")
        return self._apply_combined_filters(
            video_path, output_path, ColorGrading(lut_path=lut_path)
        )

    def apply_cinematic_look(
        self,
        video_path: str,
        look_type: str,
        output_path: str
    ) -> str:
        """
        Apply a cinematic color look to the video.

        Combines multiple grading parameters to achieve a specific cinematic aesthetic.

        Args:
            video_path: Path to the input video file.
            look_type: Type of cinematic look.
                Options: "blockbuster", "indie", "horror", "romance",
                "documentary", "noir", "teal_orange", "bleach_bypass".
            output_path: Path for the output video file.

        Returns:
            Path to the output file.
        """
        cinematic_presets = {
            "blockbuster": ColorGrading(
                brightness=-0.03, contrast=1.25, saturation=0.9,
                temperature=5.0, gamma=0.95, highlights=-0.08,
                shadows=0.05, sharpen=0.4
            ),
            "indie": ColorGrading(
                brightness=0.02, contrast=0.95, saturation=0.8,
                temperature=3.0, gamma=1.05, highlights=0.03,
                shadows=-0.03, sharpen=0.1
            ),
            "horror": ColorGrading(
                brightness=-0.12, contrast=1.4, saturation=0.5,
                temperature=-8.0, gamma=0.85, highlights=-0.15,
                shadows=-0.2, sharpen=0.3
            ),
            "romance": ColorGrading(
                brightness=0.05, contrast=1.0, saturation=1.1,
                temperature=10.0, tint=3.0, gamma=1.05,
                highlights=0.05, shadows=0.03
            ),
            "documentary": ColorGrading(
                brightness=0.0, contrast=1.1, saturation=0.9,
                temperature=2.0, gamma=1.0, highlights=-0.03,
                shadows=0.02, sharpen=0.2
            ),
            "noir": ColorGrading(
                brightness=-0.08, contrast=1.5, saturation=0.0,
                temperature=0.0, gamma=0.85, highlights=-0.2,
                shadows=-0.15, sharpen=0.5
            ),
            "teal_orange": ColorGrading(
                brightness=-0.02, contrast=1.2, saturation=1.15,
                temperature=8.0, tint=-3.0, gamma=0.95,
                highlights=-0.05, shadows=0.03, sharpen=0.3
            ),
            "bleach_bypass": ColorGrading(
                brightness=-0.05, contrast=1.5, saturation=0.6,
                temperature=0.0, gamma=0.9, highlights=-0.1,
                shadows=-0.1, sharpen=0.4
            ),
        }

        look_type_lower = look_type.lower()
        if look_type_lower not in cinematic_presets:
            available = ", ".join(cinematic_presets.keys())
            raise ValueError(
                f"Unknown cinematic look '{look_type}'. Available: {available}"
            )

        grading = cinematic_presets[look_type_lower]
        return self._apply_combined_filters(video_path, output_path, grading)

    def sharpen_video(
        self,
        video_path: str,
        amount: float,
        output_path: str
    ) -> str:
        """
        Sharpen the video using an unsharp mask.

        Args:
            video_path: Path to the input video file.
            amount: Sharpening strength (0.0 to 5.0).
            output_path: Path for the output video file.

        Returns:
            Path to the output file.
        """
        return self._apply_combined_filters(
            video_path, output_path, ColorGrading(sharpen=amount)
        )

    def reduce_noise(
        self,
        video_path: str,
        strength: float,
        output_path: str
    ) -> str:
        """
        Reduce video noise using NLMeans denoising.

        Args:
            video_path: Path to the input video file.
            strength: Denoising strength (0.0 to 10.0).
            output_path: Path for the output video file.

        Returns:
            Path to the output file.
        """
        return self._apply_combined_filters(
            video_path, output_path, ColorGrading(noise_reduction=strength)
        )

    # -------------------------------------------------------------------------
    # Full grading and presets
    # -------------------------------------------------------------------------

    def apply_full_grade(
        self,
        video_path: str,
        grading: ColorGrading,
        output_path: str
    ) -> str:
        """
        Apply a complete color grading configuration to the video.

        All parameters are applied in a single FFmpeg pass for optimal performance.

        Args:
            video_path: Path to the input video file.
            grading: ColorGrading instance with all grading parameters.
            output_path: Path for the output video file.

        Returns:
            Path to the output file.
        """
        return self._apply_combined_filters(video_path, output_path, grading)

    def apply_preset(
        self,
        video_path: str,
        preset: ColorPreset,
        output_path: str
    ) -> str:
        """
        Apply a predefined color grading preset.

        Args:
            video_path: Path to the input video file.
            preset: ColorPreset enum value.
            output_path: Path for the output video file.

        Returns:
            Path to the output file.
        """
        grading = self.PRESET_DEFINITIONS[preset]
        return self._apply_combined_filters(video_path, output_path, grading)

    # -------------------------------------------------------------------------
    # LUT generation
    # -------------------------------------------------------------------------

    def create_lut_from_preset(
        self,
        preset_name: str,
        size: int = 33,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate a 3D LUT (.cube) file from a preset or ColorGrading configuration.

        Args:
            preset_name: Name of the preset (must match a ColorPreset name)
                or "custom" to use default values.
            size: LUT grid size (default 33, typically 17, 33, or 65).
            output_path: Output path for the .cube file. If None, auto-generated.

        Returns:
            Path to the generated .cube file.
        """
        # Find the preset
        grading = None
        for preset in ColorPreset:
            if preset.value == preset_name.lower() or preset.name == preset_name.upper():
                grading = self.PRESET_DEFINITIONS[preset]
                break

        if grading is None:
            raise ValueError(
                f"Unknown preset '{preset_name}'. "
                f"Available: {[p.value for p in ColorPreset]}"
            )

        return self._generate_cube_lut(grading, size, output_path)

    def create_lut_from_grading(
        self,
        grading: ColorGrading,
        size: int = 33,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate a 3D LUT from a ColorGrading configuration.

        Args:
            grading: ColorGrading instance defining the color transformation.
            size: LUT grid size.
            output_path: Output path for the .cube file.

        Returns:
            Path to the generated .cube file.
        """
        return self._generate_cube_lut(grading, size, output_path)

    def _generate_cube_lut(
        self,
        grading: ColorGrading,
        size: int = 33,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate a .cube 3D LUT file using numpy.

        Creates a identity LUT and applies the color transformations
        to each grid point, then writes the result in .cube format.

        Args:
            grading: ColorGrading configuration to bake into the LUT.
            size: Number of grid points per axis.
            output_path: Output file path.

        Returns:
            Path to the generated .cube file.
        """
        if output_path is None:
            output_path = f"lut_{grading.to_dict().get('brightness', 0):+.2f}_{size}x{size}x{size}.cube"

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # Create identity LUT grid: values from 0 to 1 along each axis
        axis = np.linspace(0.0, 1.0, size, dtype=np.float64)
        # Create 3D meshgrid: shape (size, size, size, 3) for R, G, B
        r_grid, g_grid, b_grid = np.meshgrid(axis, axis, axis, indexing="ij")
        lut = np.stack([r_grid, g_grid, b_grid], axis=-1)  # shape: (size, size, size, 3)

        # Apply brightness
        if grading.brightness != 0.0:
            lut = np.clip(lut + grading.brightness, 0.0, 1.0)

        # Apply contrast
        if grading.contrast != 1.0:
            lut = np.clip((lut - 0.5) * grading.contrast + 0.5, 0.0, 1.0)

        # Apply gamma
        if grading.gamma != 1.0 and grading.gamma > 0:
            lut = np.power(np.clip(lut, 0.0, 1.0), 1.0 / grading.gamma)

        # Apply saturation
        if grading.saturation != 1.0:
            # Convert to luminance (BT.709)
            luminance = 0.2126 * lut[..., 0] + 0.7152 * lut[..., 1] + 0.0722 * lut[..., 2]
            luminance = luminance[..., np.newaxis]
            lut = np.clip(
                luminance + grading.saturation * (lut - luminance),
                0.0, 1.0
            )

        # Apply temperature
        if grading.temperature != 0.0:
            temp_factor = grading.temperature / 100.0
            lut[..., 0] = np.clip(lut[..., 0] + temp_factor * 0.1, 0.0, 1.0)  # Red
            lut[..., 2] = np.clip(lut[..., 2] - temp_factor * 0.1, 0.0, 1.0)  # Blue

        # Apply tint
        if grading.tint != 0.0:
            tint_factor = grading.tint / 100.0
            lut[..., 1] = np.clip(lut[..., 1] + tint_factor * 0.1, 0.0, 1.0)  # Green

        # Apply highlights adjustment
        if grading.highlights != 0.0:
            # Affect brighter regions more
            highlight_mask = np.clip((lut - 0.5) * 2.0, 0.0, 1.0)
            lut = np.clip(lut + highlight_mask * grading.highlights * 0.3, 0.0, 1.0)

        # Apply shadows adjustment
        if grading.shadows != 0.0:
            # Affect darker regions more
            shadow_mask = np.clip(1.0 - lut * 2.0, 0.0, 1.0)
            lut = np.clip(lut + shadow_mask * grading.shadows * 0.3, 0.0, 1.0)

        # Flatten to list of RGB triplets
        flat_lut = lut.reshape(-1, 3)

        # Write .cube file
        with open(output_path, "w") as f:
            f.write(f"TITLE \"Generated LUT\"\n")
            f.write(f"LUT_3D_SIZE {size}\n")
            f.write(f"DOMAIN_MIN 0.0 0.0 0.0\n")
            f.write(f"DOMAIN_MAX 1.0 1.0 1.0\n")
            f.write(f"\n")

            for i in range(flat_lut.shape[0]):
                r, g, b = flat_lut[i]
                f.write(f"{r:.6f} {g:.6f} {b:.6f}\n")

        return output_path

    # -------------------------------------------------------------------------
    # Analysis and auto-correction
    # -------------------------------------------------------------------------

    def analyze_color_distribution(
        self,
        video_path: str,
        sample_frames: int = 30
    ) -> Dict[str, Any]:
        """
        Analyze the color distribution of a video file.

        Samples frames from the video and computes histogram statistics
        for each color channel.

        Args:
            video_path: Path to the video file.
            sample_frames: Number of frames to sample for analysis.

        Returns:
            Dictionary containing color statistics:
                - mean_r, mean_g, mean_b: Average channel values (0-255).
                - std_r, std_g, std_b: Standard deviation per channel.
                - min_r, min_g, min_b: Minimum values per channel.
                - max_r, max_g, max_b: Maximum values per channel.
                - luminance_mean: Average luminance.
                - luminance_std: Luminance standard deviation.
                - color_balance: RGB balance ratios relative to average.
                - histogram_r, histogram_g, histogram_b: 256-bin histograms.
                - temperature_estimate: Estimated color temperature.
                - overall_brightness: Average brightness (0-1 scale).
                - contrast_estimate: Contrast estimate.
                - saturation_estimate: Average saturation estimate.
        """
        duration = self._get_video_duration(video_path)
        if duration <= 0:
            raise RuntimeError("Could not determine video duration.")

        # Calculate evenly-spaced timestamps to sample
        sample_interval = duration / (sample_frames + 1)
        timestamps = [sample_interval * (i + 1) for i in range(sample_frames)]

        all_r = []
        all_g = []
        all_b = []
        hist_r = np.zeros(256, dtype=np.float64)
        hist_g = np.zeros(256, dtype=np.float64)
        hist_b = np.zeros(256, dtype=np.float64)

        for ts in timestamps:
            frame_data = self._extract_frame_as_raw(video_path, ts)
            if frame_data is None:
                continue

            r, g, b = frame_data["r"], frame_data["g"], frame_data["b"]
            all_r.append(r)
            all_g.append(g)
            all_b.append(b)

            hist_r += np.bincount(r, minlength=256).astype(np.float64)
            hist_g += np.bincount(g, minlength=256).astype(np.float64)
            hist_b += np.bincount(b, minlength=256).astype(np.float64)

        if not all_r:
            raise RuntimeError("Failed to extract any frames from the video.")

        all_r = np.concatenate(all_r)
        all_g = np.concatenate(all_g)
        all_b = np.concatenate(all_b)

        # Luminance (BT.709)
        luminance = 0.2126 * all_r.astype(np.float64) + 0.7152 * all_g.astype(np.float64) + 0.0722 * all_b.astype(np.float64)

        # Saturation estimation (HSL-like)
        max_c = np.maximum(np.maximum(all_r.astype(np.float64), all_g.astype(np.float64)), all_b.astype(np.float64))
        min_c = np.minimum(np.minimum(all_r.astype(np.float64), all_g.astype(np.float64)), all_b.astype(np.float64))
        chroma = max_c - min_c
        saturation_est = np.where(max_c > 0, chroma / max_c, 0)

        mean_r = float(np.mean(all_r))
        mean_g = float(np.mean(all_g))
        mean_b = float(np.mean(all_b))

        overall_mean = (mean_r + mean_g + mean_b) / 3.0

        # Temperature estimate: positive = warm (more red), negative = cool (more blue)
        temp_estimate = 0.0
        if overall_mean > 0:
            temp_estimate = ((mean_r - mean_b) / overall_mean) * 100.0

        return {
            "mean_r": round(mean_r, 2),
            "mean_g": round(mean_g, 2),
            "mean_b": round(mean_b, 2),
            "std_r": round(float(np.std(all_r)), 2),
            "std_g": round(float(np.std(all_g)), 2),
            "std_b": round(float(np.std(all_b)), 2),
            "min_r": int(np.min(all_r)),
            "min_g": int(np.min(all_g)),
            "min_b": int(np.min(all_b)),
            "max_r": int(np.max(all_r)),
            "max_g": int(np.max(all_g)),
            "max_b": int(np.max(all_b)),
            "luminance_mean": round(float(np.mean(luminance)), 2),
            "luminance_std": round(float(np.std(luminance)), 2),
            "color_balance": {
                "r_ratio": round(mean_r / overall_mean, 4) if overall_mean > 0 else 1.0,
                "g_ratio": round(mean_g / overall_mean, 4) if overall_mean > 0 else 1.0,
                "b_ratio": round(mean_b / overall_mean, 4) if overall_mean > 0 else 1.0,
            },
            "histogram_r": hist_r.tolist(),
            "histogram_g": hist_g.tolist(),
            "histogram_b": hist_b.tolist(),
            "temperature_estimate": round(temp_estimate, 2),
            "overall_brightness": round(float(np.mean(luminance)) / 255.0, 4),
            "contrast_estimate": round(float(np.std(luminance)) / 128.0, 4),
            "saturation_estimate": round(float(np.mean(saturation_est)), 4),
            "frames_analyzed": len(timestamps),
        }

    def _extract_frame_at_second(
        self, video_path: str, timestamp: float
    ) -> Optional[np.ndarray]:
        """
        Extract a single frame from a video as a numpy RGB array.

        Args:
            video_path: Path to the video file.
            timestamp: Time in seconds to extract the frame.

        Returns:
            numpy array of shape (height, width, 3) in RGB, or None on failure.
        """
        cmd = [
            self.ffmpeg_path,
            "-ss", f"{timestamp:.3f}",
            "-i", video_path,
            "-vframes", "1",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-"
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, timeout=30
            )
            if result.returncode != 0:
                return None

            # Get frame dimensions
            metadata = self._probe_video(video_path)
            width, height = 1920, 1080  # defaults
            for stream in metadata.get("streams", []):
                if stream.get("codec_type") == "video":
                    width = int(stream.get("width", 1920))
                    height = int(stream.get("height", 1080))
                    break

            frame = np.frombuffer(result.stdout, dtype=np.uint8)
            expected_size = width * height * 3
            if frame.size != expected_size:
                return None
            return frame.reshape((height, width, 3))
        except Exception:
            return None

    def _extract_frame_as_raw(
        self, video_path: str, timestamp: float
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract a single frame and return separate R, G, B channel arrays.

        Args:
            video_path: Path to the video file.
            timestamp: Time in seconds.

        Returns:
            Dictionary with 'r', 'g', 'b' numpy arrays, or None.
        """
        frame = self._extract_frame_at_second(video_path, timestamp)
        if frame is None:
            return None
        return {
            "r": frame[:, :, 0].ravel(),
            "g": frame[:, :, 1].ravel(),
            "b": frame[:, :, 2].ravel(),
        }

    def auto_color_correct(
        self,
        video_path: str,
        output_path: str,
        strength: float = 1.0
    ) -> str:
        """
        Automatically color-correct a video based on histogram analysis.

        Analyzes the video's color distribution and applies corrections to:
        - Balance RGB channels toward neutral
        - Normalize brightness to a standard level
        - Adjust contrast to use the full dynamic range
        - Correct temperature/tint deviations

        Args:
            video_path: Path to the input video file.
            output_path: Path for the output video file.
            strength: Correction strength multiplier (0.0 to 1.0).
                0.0 = no correction, 1.0 = full correction.

        Returns:
            Path to the output file.
        """
        stats = self.analyze_color_distribution(video_path)

        # Calculate corrections
        brightness_correction = 0.0
        contrast_correction = 1.0
        saturation_correction = 1.0
        temperature_correction = 0.0
        tint_correction = 0.0

        # Brightness correction: target ~0.45 average luminance
        current_brightness = stats["overall_brightness"]
        target_brightness = 0.45
        brightness_correction = (target_brightness - current_brightness) * 0.5

        # Contrast correction: target std/128 ~ 0.3
        current_contrast = stats["contrast_estimate"]
        target_contrast = 0.3
        if current_contrast > 0.01:
            contrast_correction = target_contrast / current_contrast

        # Clamp contrast
        contrast_correction = max(0.5, min(2.0, contrast_correction))

        # Temperature correction: push toward neutral
        temp_estimate = stats["temperature_estimate"]
        if abs(temp_estimate) > 2.0:
            temperature_correction = -temp_estimate * 0.3

        # Tint correction based on G vs R/B balance
        color_balance = stats["color_balance"]
        g_ratio = color_balance["g_ratio"]
        avg_rb = (color_balance["r_ratio"] + color_balance["b_ratio"]) / 2.0
        if abs(g_ratio - avg_rb) > 0.02:
            tint_correction = (avg_rb - g_ratio) * 50.0

        # Saturation correction: boost slightly if too low, reduce if too high
        current_sat = stats["saturation_estimate"]
        if current_sat < 0.2:
            saturation_correction = 1.3
        elif current_sat > 0.6:
            saturation_correction = 0.85

        # Apply strength multiplier
        brightness_correction *= strength
        contrast_correction = 1.0 + (contrast_correction - 1.0) * strength
        saturation_correction = 1.0 + (saturation_correction - 1.0) * strength
        temperature_correction *= strength
        tint_correction *= strength

        grading = ColorGrading(
            brightness=max(-0.5, min(0.5, brightness_correction)),
            contrast=max(0.5, min(2.0, contrast_correction)),
            saturation=max(0.0, min(2.0, saturation_correction)),
            temperature=max(-50.0, min(50.0, temperature_correction)),
            tint=max(-50.0, min(50.0, tint_correction)),
            gamma=1.0,
        )

        return self._apply_combined_filters(video_path, output_path, grading)
