"""
Subtitle Generation Engine for Video Editing.

Provides SRT and ASS subtitle generation, word-level highlighting,
animated captions (karaoke, typewriter, pop-in, bounce), subtitle
burning via ffmpeg, style application, and format conversion.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class SubtitleStyle:
    """Styling properties for subtitles in ASS / SRT rendering."""

    font: str = "Arial"
    font_size: int = 48
    primary_color: str = "&H00FFFFFF"
    secondary_color: str = "&H0000FFFF"
    outline_color: str = "&H00000000"
    back_color: str = "&H80000000"
    highlight_color: str = "&H0000FFFF"
    bold: bool = True
    italic: bool = False
    outline_width: float = 2.0
    shadow_depth: float = 1.0
    alignment: int = 2
    margin_v: int = 40
    margin_l: int = 20
    margin_r: int = 20
    position: Optional[str] = None


@dataclass
class Subtitle:
    """A single subtitle entry with timing and optional style."""

    text: str
    start_time: float
    end_time: float
    style: Optional[SubtitleStyle] = None


@dataclass
class Caption:
    """A single word-level caption entry for animated / highlighted subtitles."""

    word: str
    start_time: float
    end_time: float
    is_highlighted: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _temp_path(suffix: str = ".srt") -> str:
    """Return a temporary file path inside a dedicated working directory."""
    work_dir = os.path.join(tempfile.gettempdir(), "subtitle_engine")
    os.makedirs(work_dir, exist_ok=True)
    return os.path.join(work_dir, f"{uuid.uuid4().hex}{suffix}")


def _ensure_dir(path: str) -> None:
    """Create parent directories for *path* if they do not exist."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _format_srt_time(seconds: float) -> str:
    """Convert *seconds* to SRT timestamp ``HH:MM:SS,mmm``."""
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms >= 1000:
        ms = 999
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_ass_time(seconds: float) -> str:
    """Convert *seconds* to ASS timestamp ``H:MM:SS.cc`` (centiseconds)."""
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    if cs >= 100:
        cs = 99
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _parse_srt_time(timestamp: str) -> float:
    """Parse an SRT timestamp ``HH:MM:SS,mmm`` to seconds."""
    timestamp = timestamp.strip().replace(",", ".")
    parts = timestamp.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid SRT timestamp: {timestamp}")
    h, m, s = parts
    return int(h) * 3600 + int(m) * 60 + float(s)


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


def _hex_to_ass_color(hex_color: str) -> str:
    """Convert ``#RRGGBB`` or ``&HBBGGRR`` to ASS ``&HAABBGGRR`` format.

    If the input already looks like an ASS colour (starts with ``&H``),
    it is returned as-is.
    """
    if hex_color.startswith("&H") or hex_color.startswith("&h"):
        return hex_color
    h = hex_color.lstrip("#")
    if len(h) == 6:
        r, g, b = h[0:2], h[2:4], h[4:6]
        return f"&H00{b}{g}{r}".upper()
    raise ValueError(f"Unsupported colour format: {hex_color}")


def _ass_color_to_hex(ass_color: str) -> str:
    """Convert ASS ``&HBBGGRR`` or ``&HAABBGGRR`` to ``#RRGGBB``."""
    c = ass_color.replace("&H", "").replace("&h", "")
    if len(c) == 8:
        c = c[2:]
    if len(c) != 6:
        raise ValueError(f"Invalid ASS colour: {ass_color}")
    b, g, r = c[0:2], c[2:4], c[4:6]
    return f"#{r}{g}{b}"


# ---------------------------------------------------------------------------
# SubtitleEngine
# ---------------------------------------------------------------------------

class SubtitleEngine:
    """Full-featured subtitle generation engine.

    Supports SRT and ASS output, word-level highlighting, animated caption
    styles, and ffmpeg-based subtitle burning.
    """

    def __init__(
        self,
        ffmpeg_path: str = "ffmpeg",
        ffprobe_path: str = "ffprobe",
    ) -> None:
        self._ffmpeg = ffmpeg_path
        self._ffprobe = ffprobe_path
        self._verify_tools()

    # -- internal helpers ---------------------------------------------------

    def _verify_tools(self) -> None:
        """Raise if ffmpeg binary is not reachable."""
        if shutil.which(self._ffmpeg) is None:
            raise FileNotFoundError(
                f"ffmpeg not found at '{self._ffmpeg}'. "
                "Please install ffmpeg and ensure it is on PATH."
            )

    def _run_ffmpeg(self, cmd: List[str], description: str = "ffmpeg") -> None:
        """Execute an ffmpeg command and raise on failure."""
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"{description} failed: {result.stderr}")

    # -----------------------------------------------------------------------
    # generate_subtitles
    # -----------------------------------------------------------------------

    def generate_subtitles(
        self,
        video_path: str,
        text: str,
        output_path: str = "output.srt",
    ) -> str:
        """Generate an SRT subtitle file from plain text.

        Splits *text* into sentences and distributes them evenly across the
        video duration.

        Args:
            video_path: Path to the source video.
            text: Full subtitle text (sentences separated by ``. ``, ``! ``,
                  or ``? ``).
            output_path: Path for the output ``.srt`` file.

        Returns:
            The *output_path* string.
        """
        _ensure_dir(output_path)
        duration = _get_video_duration(video_path)
        sentences = self._split_into_sentences(text)

        if not sentences:
            sentences = [text]

        time_per_sentence = duration / max(len(sentences), 1)
        subtitles: List[Subtitle] = []

        for i, sentence in enumerate(sentences):
            start = i * time_per_sentence
            end = min((i + 1) * time_per_sentence, duration)
            subtitles.append(Subtitle(text=sentence.strip(), start_time=start, end_time=end))

        srt_content = self._build_srt(subtitles)
        Path(output_path).write_text(srt_content, encoding="utf-8")
        return output_path

    @staticmethod
    def _split_into_sentences(text: str) -> List[str]:
        """Split *text* on sentence boundaries."""
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p for p in parts if p.strip()]

    # -----------------------------------------------------------------------
    # create_ass_subtitles
    # -----------------------------------------------------------------------

    def create_ass_subtitles(
        self,
        subtitles: List[Subtitle],
        style: Optional[SubtitleStyle] = None,
        output_path: str = "output.ass",
    ) -> str:
        """Create an ASS (Advanced SubStation Alpha) subtitle file.

        Args:
            subtitles: List of :class:`Subtitle` entries.
            style: Global :class:`SubtitleStyle`. Uses defaults if *None*.
            output_path: Path for the output ``.ass`` file.

        Returns:
            The *output_path* string.
        """
        _ensure_dir(output_path)
        if style is None:
            style = SubtitleStyle()

        ass_content = self._build_ass(subtitles, style)
        Path(output_path).write_text(ass_content, encoding="utf-8-sig")
        return output_path

    def _build_ass(
        self,
        subtitles: List[Subtitle],
        global_style: SubtitleStyle,
    ) -> str:
        """Build a complete ASS file string."""
        lines: List[str] = []

        lines.append("[Script Info]")
        lines.append("ScriptType: v4.00+")
        lines.append("Collisions: Normal")
        lines.append("PlayDepth: 0")
        lines.append("Timer: 100.0000")
        lines.append("WrapStyle: 0")
        lines.append("ScaledBorderAndShadow: yes")
        lines.append("YCbCr Matrix: None")
        lines.append("")

        lines.append("[V4+ Styles]")
        lines.append("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
                       "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
                       "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
                       "Alignment, MarginL, MarginR, MarginV, Encoding")
        lines.append(self._build_ass_style_line("Default", global_style))
        lines.append("")

        lines.append("[Events]")
        lines.append("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text")

        for sub in subtitles:
            start_ts = _format_ass_time(sub.start_time)
            end_ts = _format_ass_time(sub.end_time)
            style_name = "Default"
            text = sub.text.replace("\n", "\\N")
            lines.append(
                f"Dialogue: 0,{start_ts},{end_ts},{style_name},,0,0,0,,{text}"
            )

        lines.append("")
        return "\r\n".join(lines)

    @staticmethod
    def _build_ass_style_line(name: str, s: SubtitleStyle) -> str:
        """Build a single ASS style definition line."""
        bold_val = -1 if s.bold else 0
        italic_val = -1 if s.italic else 0
        return (
            f"Style: {name},"
            f"{s.font},{s.font_size},"
            f"{s.primary_color},{s.secondary_color},"
            f"{s.outline_color},{s.back_color},"
            f"{bold_val},{italic_val},0,0,"
            f"100,100,0,0,"
            f"1,{s.outline_width:.1f},{s.shadow_depth:.1f},"
            f"{s.alignment},{s.margin_l},{s.margin_r},{s.margin_v},1"
        )

    # -----------------------------------------------------------------------
    # burn_subtitles
    # -----------------------------------------------------------------------

    def burn_subtitles(
        self,
        video_path: str,
        subtitle_path: str,
        style: Optional[SubtitleStyle] = None,
        output_path: str = "output_burned.mp4",
    ) -> str:
        """Burn (hardcode) subtitles into a video using ffmpeg.

        Supports both ``.srt`` and ``.ass`` subtitle files. For ASS files
        the ``ass`` filter is used; for SRT the ``subtitles`` filter is used.

        Args:
            video_path: Path to the source video.
            subtitle_path: Path to the ``.srt`` or ``.ass`` subtitle file.
            style: Optional :class:`SubtitleStyle` for SRT override styling.
            output_path: Path for the rendered output file.

        Returns:
            The *output_path* string.
        """
        _ensure_dir(output_path)
        ext = Path(subtitle_path).suffix.lower()

        if ext == ".ass":
            filter_str = self._build_ass_burn_filter(subtitle_path)
        else:
            filter_str = self._build_srt_burn_filter(subtitle_path, style)

        cmd = [
            self._ffmpeg, "-y",
            "-i", video_path,
            "-vf", filter_str,
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "copy",
            output_path,
        ]
        self._run_ffmpeg(cmd, "burn_subtitles")
        return output_path

    @staticmethod
    def _build_ass_burn_filter(subtitle_path: str) -> str:
        """Build an ffmpeg ``-vf`` string for ASS subtitles."""
        escaped = subtitle_path.replace("\\", "/").replace(":", "\\:")
        escaped = escaped.replace("'", "\\'")
        return f"ass='{escaped}'"

    @staticmethod
    def _build_srt_burn_filter(
        subtitle_path: str,
        style: Optional[SubtitleStyle] = None,
    ) -> str:
        """Build an ffmpeg ``-vf`` string for SRT subtitles."""
        escaped = subtitle_path.replace("\\", "/").replace(":", "\\:")
        escaped = escaped.replace("'", "\\'")
        parts = [f"subtitles='{escaped}'"]

        if style is not None:
            overrides: List[str] = []
            overrides.append(f"force_style='FontName={style.font},"
                             f"FontSize={style.font_size},"
                             f"PrimaryColour={style.primary_color},"
                             f"OutlineColour={style.outline_color},"
                             f"BackColour={style.back_color},"
                             f"Bold={1 if style.bold else 0},"
                             f"Italic={1 if style.italic else 0},"
                             f"Outline={style.outline_width},"
                             f"Shadow={style.shadow_depth},"
                             f"Alignment={style.alignment},"
                             f"MarginV={style.margin_v},"
                             f"MarginL={style.margin_l},"
                             f"MarginR={style.margin_r}'")
            parts.append(":".join(overrides))

        return ":".join(parts)

    # -----------------------------------------------------------------------
    # create_word_highlight_subtitles
    # -----------------------------------------------------------------------

    def create_word_highlight_subtitles(
        self,
        words: List[str],
        timestamps: List[Tuple[float, float]],
        highlight_color: str = "#FFFF00",
        output_path: str = "output_highlight.ass",
    ) -> str:
        """Create ASS subtitles with word-level highlighting.

        At any given time only one word is highlighted. The highlighted word
        uses *highlight_color* while remaining words use the default colour.

        Args:
            words: List of words in the subtitle.
            timestamps: ``(start, end)`` for each word in *words*.
            highlight_color: Hex colour for the currently active word.
            output_path: Path for the output ``.ass`` file.

        Returns:
            The *output_path* string.

        Raises:
            ValueError: If *words* and *timestamps* have different lengths.
        """
        _ensure_dir(output_path)

        if len(words) != len(timestamps):
            raise ValueError(
                f"words ({len(words)}) and timestamps ({len(timestamps)}) "
                "must have the same length."
            )

        style = SubtitleStyle(
            font_size=52,
            bold=True,
            primary_color="&H00FFFFFF",
            outline_color="&H00000000",
            back_color="&H80000000",
            outline_width=3.0,
            shadow_depth=1.0,
            alignment=2,
            margin_v=60,
        )

        highlight_ass = _hex_to_ass_color(highlight_color)

        lines: List[str] = []
        lines.append("[Script Info]")
        lines.append("ScriptType: v4.00+")
        lines.append("Collisions: Normal")
        lines.append("PlayDepth: 0")
        lines.append("Timer: 100.0000")
        lines.append("WrapStyle: 0")
        lines.append("ScaledBorderAndShadow: yes")
        lines.append("YCbCr Matrix: None")
        lines.append("")

        lines.append("[V4+ Styles]")
        lines.append("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
                       "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
                       "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
                       "Alignment, MarginL, MarginR, MarginV, Encoding")

        base_style = (
            f"Style: Default,"
            f"{style.font},{style.font_size},"
            f"{style.primary_color},{style.secondary_color},"
            f"{style.outline_color},{style.back_color},"
            f"-1,0,0,0,"
            f"100,100,0,0,"
            f"1,{style.outline_width:.1f},{style.shadow_depth:.1f},"
            f"{style.alignment},{style.margin_l},{style.margin_r},{style.margin_v},1"
        )
        lines.append(base_style)

        highlight_style = (
            f"Style: Highlight,"
            f"{style.font},{style.font_size},"
            f"{highlight_ass},{style.secondary_color},"
            f"{style.outline_color},{style.back_color},"
            f"-1,0,0,0,"
            f"100,100,0,0,"
            f"1,{style.outline_width:.1f},{style.shadow_depth:.1f},"
            f"{style.alignment},{style.margin_l},{style.margin_r},{style.margin_v},1"
        )
        lines.append(highlight_style)
        lines.append("")

        lines.append("[Events]")
        lines.append("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text")

        for idx, (word, (start, end)) in enumerate(zip(words, timestamps)):
            start_ts = _format_ass_time(start)
            end_ts = _format_ass_time(end)

            before = " ".join(words[:idx])
            after = " ".join(words[idx + 1:])
            parts: List[str] = []
            if before:
                parts.append(f"{{\\c{style.primary_color}}}{before}")
            parts.append(f"{{\\c{highlight_ass}}}{word}")
            if after:
                parts.append(f"{{\\c{style.primary_color}}}{after}")
            text = " ".join(parts)

            lines.append(
                f"Dialogue: 0,{start_ts},{end_ts},Default,,0,0,0,,{text}"
            )

        lines.append("")
        Path(output_path).write_text("\r\n".join(lines), encoding="utf-8-sig")
        return output_path

    # -----------------------------------------------------------------------
    # create_animated_captions
    # -----------------------------------------------------------------------

    def create_animated_captions(
        self,
        video_path: str,
        captions: List[Caption],
        animation_style: str = "typewriter",
        output_path: str = "output_animated.ass",
    ) -> str:
        """Create animated caption subtitles with various effects.

        Supported *animation_style* values:
            * ``"karaoke"``    – Words fill in one at a time using ``\\kf``.
            * ``"typewriter"`` – Characters appear one at a time with ``\\k``.
            * ``"pop-in"``     – Words scale from 0 to 100% using ``\\fscx``/``\\fscy``.
            * ``"bounce"``     – Words drop in with a bounce using ``\\frz`` and ``\\fay``.

        Args:
            video_path: Path to the source video (used to determine duration
                for fallback timing if caption timings are missing).
            captions: List of :class:`Caption` entries with word-level timing.
            animation_style: Name of the animation effect.
            output_path: Path for the output ``.ass`` file.

        Returns:
            The *output_path* string.
        """
        _ensure_dir(output_path)

        style = SubtitleStyle(
            font_size=52,
            bold=True,
            primary_color="&H00FFFFFF",
            highlight_color="&H0000FFFF",
            outline_color="&H00000000",
            back_color="&H80000000",
            outline_width=3.0,
            shadow_depth=1.0,
            alignment=2,
            margin_v=60,
        )

        if animation_style == "karaoke":
            ass_content = self._build_karaoke_ass(captions, style)
        elif animation_style == "typewriter":
            ass_content = self._build_typewriter_ass(captions, style)
        elif animation_style == "pop-in":
            ass_content = self._build_pop_in_ass(captions, style)
        elif animation_style == "bounce":
            ass_content = self._build_bounce_ass(captions, style)
        else:
            raise ValueError(
                f"Unknown animation_style '{animation_style}'. "
                "Supported: karaoke, typewriter, pop-in, bounce."
            )

        Path(output_path).write_text(ass_content, encoding="utf-8-sig")
        return output_path

    def _build_karaoke_ass(
        self,
        captions: List[Caption],
        style: SubtitleStyle,
    ) -> str:
        """Build ASS content with karaoke fill (\\kf) animation.

        Words fill from left to right in the highlight colour, then snap
        to the primary colour when the next word begins.
        """
        highlight = style.highlight_color

        lines = self._ass_preamble(style)

        group = self._group_captions_into_lines(captions)
        for line_captions in group:
            text_parts: List[str] = []
            for cap in line_captions:
                duration_cs = int((cap.end_time - cap.start_time) * 100)
                duration_cs = max(duration_cs, 1)
                text_parts.append(
                    f"{{\\kf{duration_cs}\\c{highlight}}}"
                    f"{cap.word}"
                    f"{{\\c{style.primary_color}}}"
                )
            text = " ".join(text_parts)
            start_ts = _format_ass_time(line_captions[0].start_time)
            end_ts = _format_ass_time(line_captions[-1].end_time)
            lines.append(
                f"Dialogue: 0,{start_ts},{end_ts},Default,,0,0,0,,{text}"
            )

        lines.append("")
        return "\r\n".join(lines)

    def _build_typewriter_ass(
        self,
        captions: List[Caption],
        style: SubtitleStyle,
    ) -> str:
        """Build ASS content with typewriter animation (\\k per syllable).

        Each word is broken into per-character timing. A ``\\k`` tag is
        inserted before each character to make it appear sequentially.
        """
        highlight = style.highlight_color

        lines = self._ass_preamble(style)

        group = self._group_captions_into_lines(captions)
        for line_captions in group:
            text_parts: List[str] = []
            for cap in line_captions:
                word_dur = max(cap.end_time - cap.start_time, 0.01)
                chars = list(cap.word)
                char_dur = word_dur / max(len(chars), 1)
                for i, ch in enumerate(chars):
                    delay_cs = int(char_dur * 100)
                    if i == 0:
                        text_parts.append(f"{{\\c{highlight}}}{ch}")
                    else:
                        text_parts.append(f"{{\\k{delay_cs}\\c{highlight}}}{ch}")
                text_parts.append(f"{{\\c{style.primary_color}}}")
            text = "".join(text_parts)
            start_ts = _format_ass_time(line_captions[0].start_time)
            end_ts = _format_ass_time(line_captions[-1].end_time)
            lines.append(
                f"Dialogue: 0,{start_ts},{end_ts},Default,,0,0,0,,{text}"
            )

        lines.append("")
        return "\r\n".join(lines)

    def _build_pop_in_ass(
        self,
        captions: List[Caption],
        style: SubtitleStyle,
    ) -> str:
        """Build ASS content with pop-in scale animation.

        Each word scales from 0% to 100% over a short duration using
        ``\\fscx`` and ``\\fscy`` tags with ``\\t`` transforms.
        """
        lines = self._ass_preamble(style)

        group = self._group_captions_into_lines(captions)
        for line_captions in group:
            text_parts: List[str] = []
            for cap in line_captions:
                dur_ms = int((cap.end_time - cap.start_time) * 1000)
                dur_ms = max(dur_ms, 1)
                transform_end = min(dur_ms, 300)
                text_parts.append(
                    f"{{\\fscx0\\fscy0\\t(0,{transform_end},\\fscx100\\fscy100)"
                    f"}}{cap.word}"
                )
            text = " ".join(text_parts)
            start_ts = _format_ass_time(line_captions[0].start_time)
            end_ts = _format_ass_time(line_captions[-1].end_time)
            lines.append(
                f"Dialogue: 0,{start_ts},{end_ts},Default,,0,0,0,,{text}"
            )

        lines.append("")
        return "\r\n".join(lines)

    def _build_bounce_ass(
        self,
        captions: List[Caption],
        style: SubtitleStyle,
    ) -> str:
        """Build ASS content with bounce drop-in animation.

        Each word drops from above with a rotation using ``\\frz`` and
        ``\\fay`` transform tags, with ``\\t`` for easing.
        """
        lines = self._ass_preamble(style)

        group = self._group_captions_into_lines(captions)
        for line_captions in group:
            text_parts: List[str] = []
            for cap in line_captions:
                dur_ms = int((cap.end_time - cap.start_time) * 1000)
                dur_ms = max(dur_ms, 1)
                mid = min(dur_ms // 2, 200)
                text_parts.append(
                    f"{{\\frz15\\fay50\\t(0,{mid},\\frz-5\\fay0)"
                    f"\\t({mid},{dur_ms},\\frz0\\fay0)"
                    f"}}{cap.word}"
                )
            text = " ".join(text_parts)
            start_ts = _format_ass_time(line_captions[0].start_time)
            end_ts = _format_ass_time(line_captions[-1].end_time)
            lines.append(
                f"Dialogue: 0,{start_ts},{end_ts},Default,,0,0,0,,{text}"
            )

        lines.append("")
        return "\r\n".join(lines)

    def _ass_preamble(self, style: SubtitleStyle) -> List[str]:
        """Return common ASS header + style lines."""
        lines: List[str] = []
        lines.append("[Script Info]")
        lines.append("ScriptType: v4.00+")
        lines.append("Collisions: Normal")
        lines.append("PlayDepth: 0")
        lines.append("Timer: 100.0000")
        lines.append("WrapStyle: 0")
        lines.append("ScaledBorderAndShadow: yes")
        lines.append("YCbCr Matrix: None")
        lines.append("")

        lines.append("[V4+ Styles]")
        lines.append("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
                       "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
                       "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
                       "Alignment, MarginL, MarginR, MarginV, Encoding")
        lines.append(self._build_ass_style_line("Default", style))
        lines.append("")

        lines.append("[Events]")
        lines.append("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text")
        return lines

    @staticmethod
    def _group_captions_into_lines(
        captions: List[Caption],
        max_words: int = 8,
    ) -> List[List[Caption]]:
        """Group captions into display lines, splitting on large gaps.

        A new line is started when the gap between consecutive captions
        exceeds 0.4 seconds or when *max_words* is reached.
        """
        if not captions:
            return []

        groups: List[List[Caption]] = []
        current: List[Caption] = [captions[0]]

        for prev, cap in zip(captions, captions[1:]):
            gap = cap.start_time - prev.end_time
            if gap > 0.4 or len(current) >= max_words:
                groups.append(current)
                current = [cap]
            else:
                current.append(cap)

        if current:
            groups.append(current)

        return groups

    # -----------------------------------------------------------------------
    # split_text_to_captions
    # -----------------------------------------------------------------------

    def split_text_to_captions(
        self,
        text: str,
        max_chars_per_line: int = 40,
        max_lines: int = 2,
    ) -> List[Caption]:
        """Split a text string into :class:`Caption` objects with estimated timing.

        Timing is estimated based on average speaking rate (150 wpm). Each
        word gets an equal share of the total estimated duration.

        Args:
            text: Input text to split.
            max_chars_per_line: Maximum characters per display line.
            max_lines: Maximum number of display lines per caption.

        Returns:
            List of :class:`Caption` entries with estimated timings.
        """
        words = text.split()
        if not words:
            return []

        words_per_second = 150 / 60.0
        total_duration = len(words) / words_per_second
        time_per_word = total_duration / max(len(words), 1)

        captions: List[Caption] = []
        current_time = 0.0

        for word in words:
            end_time = current_time + time_per_word
            captions.append(Caption(
                word=word,
                start_time=current_time,
                end_time=end_time,
                is_highlighted=False,
            ))
            current_time = end_time

        return captions

    # -----------------------------------------------------------------------
    # sync_captions_to_audio
    # -----------------------------------------------------------------------

    def sync_captions_to_audio(
        self,
        video_path: str,
        captions: List[Caption],
    ) -> List[Caption]:
        """Attempt to synchronize captions to audio using silence detection.

        Uses ``ffmpeg``'s ``silencedetect`` filter to find speech segments
        and adjusts caption timings to fit within detected speech regions.

        Args:
            video_path: Path to the source video/audio.
            captions: List of :class:`Caption` entries to adjust.

        Returns:
            New list of :class:`Caption` entries with adjusted timings.
        """
        if not captions:
            return []

        speech_segments = self._detect_speech_segments(video_path)
        if not speech_segments:
            return list(captions)

        total_speech_duration = sum(e - s for s, e in speech_segments)
        total_caption_duration = captions[-1].end_time - captions[0].start_time

        if total_speech_duration <= 0 or total_caption_duration <= 0:
            return list(captions)

        scale = total_speech_duration / total_caption_duration

        adjusted: List[Caption] = []
        offset = speech_segments[0][0] - captions[0].start_time * scale

        for cap in captions:
            new_start = cap.start_time * scale + offset
            new_end = cap.end_time * scale + offset

            new_start = max(speech_segments[0][0], new_start)
            new_end = min(speech_segments[-1][1], new_end)

            adjusted.append(Caption(
                word=cap.word,
                start_time=new_start,
                end_time=new_end,
                is_highlighted=cap.is_highlighted,
            ))

        return adjusted

    def _detect_speech_segments(
        self,
        video_path: str,
        noise_threshold: float = -30.0,
        min_silence_duration: float = 0.1,
    ) -> List[Tuple[float, float]]:
        """Detect speech segments by identifying silence gaps.

        Uses ``ffmpeg -af silencedetect`` to find periods of silence and
        inverts them to get speech regions.

        Returns:
            List of ``(start, end)`` tuples in seconds.
        """
        cmd = [
            self._ffmpeg, "-i", video_path,
            "-af", f"silencedetect=noise={noise_threshold}dB:d={min_silence_duration}",
            "-f", "null", "-",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        stderr = result.stderr

        silences: List[Tuple[float, float]] = []
        start_pattern = re.compile(r"silence_start:\s*([\d.]+)")
        end_pattern = re.compile(r"silence_end:\s*([\d.]+)")

        starts = [float(m.group(1)) for m in start_pattern.finditer(stderr)]
        ends = [float(m.group(1)) for m in end_pattern.finditer(stderr)]

        duration = _get_video_duration(video_path)

        speech: List[Tuple[float, float]] = []
        prev_end = 0.0

        for s, e in zip(starts, ends):
            if s > prev_end + 0.01:
                speech.append((prev_end, s))
            prev_end = e

        if prev_end < duration - 0.01:
            speech.append((prev_end, duration))

        return speech

    # -----------------------------------------------------------------------
    # apply_subtitle_style
    # -----------------------------------------------------------------------

    def apply_subtitle_style(
        self,
        subtitle_path: str,
        style: SubtitleStyle,
        output_path: str = "output_styled.ass",
    ) -> str:
        """Apply a :class:`SubtitleStyle` to an existing subtitle file.

        If the input is an ``.ass`` file, the style definitions are
        rewritten. If the input is an ``.srt`` file, it is converted to
        ASS with the given style.

        Args:
            subtitle_path: Path to the ``.srt`` or ``.ass`` source file.
            style: The :class:`SubtitleStyle` to apply.
            output_path: Path for the output ``.ass`` file.

        Returns:
            The *output_path* string.
        """
        _ensure_dir(output_path)
        ext = Path(subtitle_path).suffix.lower()

        if ext == ".srt":
            subtitles = self._parse_srt(subtitle_path)
            return self.create_ass_subtitles(subtitles, style, output_path)

        content = Path(subtitle_path).read_text(encoding="utf-8-sig")
        content = self._rewrite_ass_styles(content, style)
        Path(output_path).write_text(content, encoding="utf-8-sig")
        return output_path

    def _rewrite_ass_styles(
        self,
        ass_content: str,
        new_style: SubtitleStyle,
    ) -> str:
        """Replace style definitions in an ASS file with *new_style*."""
        lines = ass_content.splitlines()
        result: List[str] = []
        in_styles = False
        format_parsed = False
        style_columns: List[str] = []

        for line in lines:
            stripped = line.strip()

            if stripped == "[V4+ Styles]":
                in_styles = True
                result.append(line)
                continue

            if stripped.startswith("[") and stripped.endswith("]") and in_styles:
                in_styles = False
                format_parsed = False
                result.append(line)
                continue

            if in_styles and stripped.startswith("Format:"):
                format_parsed = True
                result.append(line)
                continue

            if in_styles and format_parsed and stripped.startswith("Style:"):
                style_name = stripped.split(":", 1)[1].strip().split(",")[0].strip()
                result.append(self._build_ass_style_line(style_name, new_style))
                continue

            result.append(line)

        return "\n".join(result)

    def _parse_srt(self, srt_path: str) -> List[Subtitle]:
        """Parse an SRT file into :class:`Subtitle` objects."""
        content = Path(srt_path).read_text(encoding="utf-8")
        blocks = re.split(r"\n\s*\n", content.strip())
        subtitles: List[Subtitle] = []

        for block in blocks:
            lines = block.strip().splitlines()
            if len(lines) < 2:
                continue

            time_line = None
            text_lines: List[str] = []

            for line in lines:
                if "-->" in line:
                    time_line = line
                elif not line.strip().isdigit() and time_line is not None:
                    text_lines.append(line)

            if time_line and text_lines:
                parts = time_line.split("-->")
                if len(parts) == 2:
                    start = _parse_srt_time(parts[0])
                    end = _parse_srt_time(parts[1])
                    text = "\n".join(text_lines).strip()
                    subtitles.append(Subtitle(text=text, start_time=start, end_time=end))

        return subtitles

    # -----------------------------------------------------------------------
    # generate_srt_from_ass
    # -----------------------------------------------------------------------

    def generate_srt_from_ass(
        self,
        ass_path: str,
        output_path: str = "output.srt",
    ) -> str:
        """Convert an ASS subtitle file to SRT format.

        ASS-specific tags (karaoke, animations) are stripped and only the
        plain text is preserved.

        Args:
            ass_path: Path to the ``.ass`` source file.
            output_path: Path for the output ``.srt`` file.

        Returns:
            The *output_path* string.
        """
        _ensure_dir(output_path)
        content = Path(ass_path).read_text(encoding="utf-8-sig")

        events_section = False
        subtitles: List[Subtitle] = []

        for line in content.splitlines():
            stripped = line.strip()

            if stripped == "[Events]":
                events_section = True
                continue

            if stripped.startswith("[") and stripped.endswith("]") and events_section:
                events_section = False
                continue

            if not events_section or not stripped.startswith("Dialogue:"):
                continue

            parts = stripped.split(":", 1)
            if len(parts) < 2:
                continue

            fields = parts[1].split(",", 9)
            if len(fields) < 10:
                continue

            start_str = fields[1].strip()
            end_str = fields[2].strip()
            raw_text = fields[9].strip()

            start = self._parse_ass_time(start_str)
            end = self._parse_ass_time(end_str)
            text = self._strip_ass_tags(raw_text)

            if text.strip():
                subtitles.append(Subtitle(
                    text=text.strip(),
                    start_time=start,
                    end_time=end,
                ))

        srt_content = self._build_srt(subtitles)
        Path(output_path).write_text(srt_content, encoding="utf-8")
        return output_path

    @staticmethod
    def _parse_ass_time(timestamp: str) -> float:
        """Parse an ASS timestamp ``H:MM:SS.cc`` to seconds."""
        timestamp = timestamp.strip()
        parts = timestamp.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid ASS timestamp: {timestamp}")
        h = int(parts[0])
        m = int(parts[1])
        s_cs = parts[2].split(".")
        s = int(s_cs[0])
        cs = int(s_cs[1]) if len(s_cs) > 1 else 0
        return h * 3600 + m * 60 + s + cs / 100.0

    @staticmethod
    def _strip_ass_tags(text: str) -> str:
        """Remove all ASS override tags from *text*.

        Strips ``{...}`` blocks and ``\\N`` / ``\\n`` line breaks are
        converted to newlines.
        """
        cleaned = re.sub(r"\{[^}]*\}", "", text)
        cleaned = cleaned.replace("\\N", "\n").replace("\\n", "\n")
        cleaned = re.sub(r"\\[kK]\d+", "", cleaned)
        return cleaned.strip()

    def _build_srt(self, subtitles: List[Subtitle]) -> str:
        """Build SRT content from a list of :class:`Subtitle` objects."""
        blocks: List[str] = []
        for i, sub in enumerate(subtitles, start=1):
            start_ts = _format_srt_time(sub.start_time)
            end_ts = _format_srt_time(sub.end_time)
            blocks.append(f"{i}\n{start_ts} --> {end_ts}\n{sub.text}\n")
        return "\n".join(blocks)
