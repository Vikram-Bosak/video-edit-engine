"""Text animation engine for professional text overlays and animated captions."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from python.core.config import TextAnimation as TextAnimEnum
except ImportError:
    class TextAnimEnum(Enum):
        POP = "pop"
        BOUNCE = "bounce"
        FADE = "fade"
        SCALE = "scale"
        TYPEWRITER = "typewriter"
        SLIDE_LEFT = "slide_left"
        SLIDE_RIGHT = "slide_right"
        SLIDE_UP = "slide_up"
        SLIDE_DOWN = "slide_down"
        ELASTIC = "elastic"
        ROTATE = "rotate"
        NEON_GLOW = "neon_glow"
        ZOOM_IN = "zoom_in"
        WAVE = "wave"


class TextAnimation(Enum):
    POP = "pop"
    BOUNCE = "bounce"
    FADE = "fade"
    SCALE = "scale"
    TYPEWRITER = "typewriter"
    SLIDE_LEFT = "slide_left"
    SLIDE_RIGHT = "slide_right"
    SLIDE_UP = "slide_up"
    SLIDE_DOWN = "slide_down"
    ELASTIC = "elastic"
    ROTATE = "rotate"
    NEON_GLOW = "neon_glow"
    ZOOM_IN = "zoom_in"
    WAVE = "wave"


class TextAlignment(Enum):
    LEFT = 1
    CENTER = 2
    RIGHT = 7
    TOP_LEFT = 9
    TOP_CENTER = 10
    TOP_RIGHT = 11
    BOTTOM_LEFT = 5
    BOTTOM_CENTER = 6
    BOTTOM_RIGHT = 7


@dataclass
class TextStyle:
    font: str = "Arial"
    font_size: int = 72
    color: str = "&H00FFFFFF"
    secondary_color: str = "&H0000FFFF"
    bg_color: str = "&H80000000"
    stroke_color: str = "&H00000000"
    stroke_width: int = 3
    shadow_color: str = "&H80000000"
    shadow_offset: int = 2
    bold: bool = True
    italic: bool = False
    alignment: TextAlignment = TextAlignment.BOTTOM_CENTER
    margin_v: int = 60
    margin_l: int = 40
    margin_r: int = 40
    position: Optional[Tuple[int, int]] = None
    opacity: float = 1.0


@dataclass
class AnimatedText:
    text: str
    start_time: float
    end_time: float
    animation: TextAnimation = TextAnimation.FADE
    style: Optional[TextStyle] = None
    position: Optional[Tuple[int, int]] = None


@dataclass
class CaptionWord:
    word: str
    start_time: float
    end_time: float
    is_highlighted: bool = False


class TextEngine:
    """Engine for creating and animating text overlays."""

    def __init__(self, temp_dir: Optional[str] = None, ffmpeg_path: str = "ffmpeg"):
        self.ffmpeg_path = ffmpeg_path
        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix="text_engine_")
        os.makedirs(self.temp_dir, exist_ok=True)

    def _run_ffmpeg(self, args: List[str]) -> subprocess.CompletedProcess:
        cmd = [self.ffmpeg_path, "-y", "-hide_banner", "-loglevel", "error"] + args
        return subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    def _get_duration(self, path: str) -> float:
        cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        try:
            return float(result.stdout.strip())
        except ValueError:
            return 0.0

    def _hex_to_ass(self, hex_color: str) -> str:
        hex_color = hex_color.lstrip("#")
        if len(hex_color) == 6:
            r, g, b = hex_color[0:2], hex_color[2:4], hex_color[4:6]
            return f"&H00{b}{g}{r}&"
        return hex_color if hex_color.startswith("&") else "&H00FFFFFF&"

    def _position_to_ass_alignment(self, position: Optional[Tuple[int, int]], alignment: TextAlignment) -> int:
        return alignment.value

    def _generate_drawtext_filter(self, text: str, style: TextStyle, duration: float, animation: TextAnimation) -> str:
        font_size = style.font_size
        color = self._hex_to_ass(style.color).replace("&", "\\&")
        text_escaped = text.replace("'", "'\\''").replace(":", "\\:")
        x_expr = "w/2-tw/2"
        y_expr = f"h-th-{style.margin_v}"

        if style.position:
            x_expr = str(style.position[0])
            y_expr = str(style.position[1])

        enable = f"between(t,0,{duration})"

        base = (
            f"drawtext=text='{text_escaped}':fontsize={font_size}:fontcolor={color}"
            f":x={x_expr}:y={y_expr}"
            f":borderw={style.stroke_width}:bordercolor={self._hex_to_ass(style.stroke_color).replace('&', '\\&')}"
            f":enable='{enable}'"
        )

        if style.bold:
            base += f":fontfile="

        return base

    def _generate_ass_animation_tag(self, animation: TextAnimation, duration: float, frame_count: int = 0) -> str:
        fade_in_frames = 8
        fade_out_frames = 8
        duration_cs = int(duration * 100)

        if animation == TextAnimation.FADE:
            return f"\\fad(300,300)"
        elif animation == TextAnimation.POP:
            return (
                f"\\t(0,100,\\fscx50\\fscy50)"
                f"\\t(100,200,\\fscx110\\fscy110)"
                f"\\t(200,300,\\fscx100\\fscy100)"
                f"\\fad(0,{fade_out_frames * 10})"
            )
        elif animation == TextAnimation.BOUNCE:
            return (
                f"\\t(0,80,\\pos(0,-50)\\fscx120\\fscy80)"
                f"\\t(80,160,\\fscx90\\fscy110)"
                f"\\t(160,240,\\fscx105\\fscy95)"
                f"\\t(240,300,\\fscx100\\fscy100)"
            )
        elif animation == TextAnimation.SCALE:
            return f"\\t(0,{duration_cs // 3},\\fscx30\\fscy30)\\t({duration_cs // 3},{duration_cs * 2 // 3},\\fscx100\\fscy100)"
        elif animation == TextAnimation.TYPEWRITER:
            return f"\\fad(0,0)"
        elif animation == TextAnimation.SLIDE_LEFT:
            return f"\\t(0,150,\\pos(-200,0)\\fad(150,0))"
        elif animation == TextAnimation.SLIDE_RIGHT:
            return f"\\t(0,150,\\fad(150,0))"
        elif animation == TextAnimation.SLIDE_UP:
            return f"\\t(0,150,\\fad(150,0))"
        elif animation == TextAnimation.SLIDE_DOWN:
            return f"\\t(0,150,\\fad(150,0))"
        elif animation == TextAnimation.ELASTIC:
            return (
                f"\\t(0,100,\\fscx200\\fscy200)"
                f"\\t(100,200,\\fscx80\\fscy80)"
                f"\\t(200,300,\\fscx115\\fscy115)"
                f"\\t(300,400,\\fscx95\\fscy95)"
                f"\\t(400,500,\\fscx100\\fscy100)"
            )
        elif animation == TextAnimation.ROTATE:
            return f"\\t(0,{duration_cs // 2},\\frz360)\\t({duration_cs // 2},{duration_cs},\\frz0)"
        elif animation == TextAnimation.NEON_GLOW:
            return f"\\bord8\\shad4\\3c&H00FFFF&\\4c&H00FFFF&"
        elif animation == TextAnimation.ZOOM_IN:
            return f"\\t(0,200,\\fscx10\\fscy10)\\t(200,400,\\fscx100\\fscy100)"
        elif animation == TextAnimation.WAVE:
            return f"\\t(0,{duration_cs},\\fscy80)\\t({duration_cs // 4},{duration_cs * 3 // 4},\\fscy120)"
        return ""

    def create_ass_subtitles(
        self,
        texts: List[AnimatedText],
        width: int = 1080,
        height: int = 1920,
        output_path: Optional[str] = None,
    ) -> str:
        if not output_path:
            output_path = os.path.join(self.temp_dir, f"subs_{uuid.uuid4().hex[:8]}.ass")

        style = texts[0].style if texts[0].style else TextStyle()
        ass_color = self._hex_to_ass(style.color)
        ass_outline = self._hex_to_ass(style.stroke_color)
        ass_shadow = self._hex_to_ass(style.shadow_color)

        header = f"""[Script Info]
Title: Animated Text
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{style.font},{style.font_size},{ass_color},{self._hex_to_ass(style.secondary_color)},{ass_outline},{ass_shadow},{int(style.bold)},{int(style.italic)},0,0,100,100,0,0,1,{style.stroke_width},{style.shadow_offset},{style.alignment.value},{style.margin_l},{style.margin_r},{style.margin_v},1
Style: Highlight,{style.font},{style.font_size},&H0000FFFF&,&H00FFFFFF&,{ass_outline},{ass_shadow},{int(style.bold)},0,0,0,100,100,0,0,1,{style.stroke_width},{style.shadow_offset},{style.alignment.value},{style.margin_l},{style.margin_r},{style.margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"""

        lines = [header]
        for t in texts:
            start = self._seconds_to_ass_time(t.start_time)
            end = self._seconds_to_ass_time(t.end_time)
            anim_style = t.animation if hasattr(t, 'animation') else TextAnimation.FADE
            anim_tag = self._generate_ass_animation_tag(anim_style, t.end_time - t.start_time)
            pos_tag = ""
            if t.position:
                pos_tag = f"\\pos({t.position[0]},{t.position[1]})"
            elif style.position:
                pos_tag = f"\\pos({style.position[0]},{style.position[1]})"

            effect = anim_tag + pos_tag
            lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{effect}{t.text}")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return output_path

    def _seconds_to_ass_time(self, seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        return f"{h}:{m:02d}:{s:05.2f}"

    def _seconds_to_srt_time(self, seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    def create_text_clip(
        self,
        text: str,
        style: Optional[TextStyle] = None,
        duration: float = 3.0,
        width: int = 1080,
        height: int = 1920,
        output_path: Optional[str] = None,
    ) -> str:
        if not style:
            style = TextStyle()
        if not output_path:
            output_path = os.path.join(self.temp_dir, f"text_{uuid.uuid4().hex[:8]}.mp4")

        x = f"(w-tw)/2" if style.position is None else str(style.position[0])
        y = f"(h-th)/2" if style.position is None else str(style.position[1])

        color = self._hex_to_ass(style.color).replace("&H00", "").rstrip("&")
        text_escaped = text.replace("'", "'\\''").replace(":", "\\:").replace("%", "%%")

        filters = [
            f"color=c=black@0:s={width}x{height}:d={duration}:r=30",
            f"drawtext=text='{text_escaped}':fontsize={style.font_size}"
            f":fontcolor=0x{color}:x={x}:y={y}"
            f":borderw={style.stroke_width}:bordercolor=black"
        ]

        filter_str = ",".join(filters)
        args = [
            "-f", "lavfi", "-i", filter_str,
            "-t", str(duration),
            "-c:v", "libx264", "-pix_fmt", "yuva420p",
            "-preset", "fast",
            output_path
        ]

        result = self._run_ffmpeg(args)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to create text clip: {result.stderr}")
        return output_path

    def animate_text(
        self,
        text_clip_path: str,
        animation: TextAnimation = TextAnimation.FADE,
        duration: float = 3.0,
        output_path: Optional[str] = None,
    ) -> str:
        if not output_path:
            output_path = os.path.join(self.temp_dir, f"animated_{uuid.uuid4().hex[:8]}.mp4")

        vf_parts = []
        anim_frames = int(duration * 30)

        if animation == TextAnimation.FADE:
            vf_parts.append(f"fade=t=in:st=0:d=0.5,fade=t=out:st={duration - 0.5}:d=0.5")
        elif animation == TextAnimation.POP:
            vf_parts.append(
                f"select='between(n,0,{anim_frames})',"
                f"setpts=PTS-STARTPTS,"
                f"scale=trunc(({anim_frames}/2)*t/{max(duration, 0.01)})*iw/{anim_frames}:ih"
            )
        elif animation == TextAnimation.BOUNCE:
            vf_parts.append(
                f"select='between(n,0,{anim_frames})',"
                f"setpts=PTS-STARTPTS"
            )
        elif animation == TextAnimation.SCALE:
            vf_parts.append(
                f"select='between(n,0,{anim_frames})',"
                f"setpts=PTS-STARTPTS,"
                f"scale=iw*min(1,t/{max(duration * 0.3, 0.01)}):ih*min(1,t/{max(duration * 0.3, 0.01)})"
            )
        elif animation in (TextAnimation.SLIDE_LEFT, TextAnimation.SLIDE_RIGHT,
                          TextAnimation.SLIDE_UP, TextAnimation.SLIDE_DOWN):
            vf_parts.append(
                f"select='between(n,0,{anim_frames})',"
                f"setpts=PTS-STARTPTS,"
                f"crop=iw:ih:t='min(1,t/0.3)*iw*0.5'"
            )
        else:
            vf_parts.append(f"fade=t=in:st=0:d=0.3,fade=t=out:st={duration - 0.3}:d=0.3")

        vf = ",".join(vf_parts) if vf_parts else "null"
        args = [
            "-i", text_clip_path,
            "-vf", vf,
            "-t", str(duration),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "fast",
            output_path
        ]

        result = self._run_ffmpeg(args)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to animate text: {result.stderr}")
        return output_path

    def burn_text_into_video(
        self,
        video_path: str,
        text: str,
        style: Optional[TextStyle] = None,
        start_time: float = 0,
        duration: float = 3.0,
        animation: TextAnimation = TextAnimation.FADE,
        output_path: Optional[str] = None,
    ) -> str:
        if not style:
            style = TextStyle()
        if not output_path:
            output_path = os.path.join(self.temp_dir, f"burned_{uuid.uuid4().hex[:8]}.mp4")

        text_escaped = text.replace("'", "'\\''").replace(":", "\\:").replace("%", "%%")
        color = self._hex_to_ass(style.color).replace("&H00", "").rstrip("&")

        x_expr = "w/2-tw/2"
        y_expr = f"h-th-{style.margin_v}"
        if style.position:
            x_expr = str(style.position[0])
            y_expr = str(style.position[1])

        enable = f"between(t\\,{start_time}\\,{start_time + duration})"

        vf = (
            f"drawtext=text='{text_escaped}'"
            f":fontsize={style.font_size}"
            f":fontcolor=0x{color}"
            f":x={x_expr}:y={y_expr}"
            f":borderw={style.stroke_width}"
            f":bordercolor=black"
            f":enable='{enable}'"
            f":alpha='if(between(t,{start_time},{start_time + 0.1}),(t-{start_time})/0.1,if(between(t,{start_time + duration - 0.1},{start_time + duration}),({start_time + duration}-t)/0.1,1))'"
        )

        args = [
            "-i", video_path,
            "-vf", vf,
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "copy",
            "-pix_fmt", "yuv420p",
            output_path
        ]

        result = self._run_ffmpeg(args)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to burn text: {result.stderr}")
        return output_path

    def create_title_card(
        self,
        title: str,
        subtitle: str = "",
        style: Optional[TextStyle] = None,
        duration: float = 4.0,
        width: int = 1080,
        height: int = 1920,
        output_path: Optional[str] = None,
    ) -> str:
        if not style:
            style = TextStyle()
        if not output_path:
            output_path = os.path.join(self.temp_dir, f"title_{uuid.uuid4().hex[:8]}.mp4")

        title_esc = title.replace("'", "'\\''").replace(":", "\\:")
        sub_esc = subtitle.replace("'", "'\\''").replace(":", "\\:")
        title_color = self._hex_to_ass(style.color).replace("&H00", "").rstrip("&")

        filters = [
            f"color=c=black:s={width}x{height}:d={duration}:r=30",
            f"drawtext=text='{title_esc}':fontsize={style.font_size}:fontcolor=0x{title_color}"
            f":x=(w-tw)/2:y=(h-th)/2-50"
            f":borderw={style.stroke_width}:bordercolor=black"
            f":enable='between(t,0.3,{duration})'"
            f":alpha='min(1,(t-0.3)/0.5)'",
        ]

        if subtitle:
            sub_color = self._hex_to_ass(style.secondary_color).replace("&H00", "").rstrip("&")
            filters.append(
                f"drawtext=text='{sub_esc}':fontsize={style.font_size // 2}"
                f":fontcolor=0x{sub_color}"
                f":x=(w-tw)/2:y=(h-th)/2+80"
                f":enable='between(t,0.8,{duration})'"
                f":alpha='min(1,(t-0.8)/0.5)'"
            )

        filter_str = ",".join(filters)
        args = [
            "-f", "lavfi", "-i", filter_str,
            "-t", str(duration),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "fast",
            output_path
        ]

        result = self._run_ffmpeg(args)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to create title card: {result.stderr}")
        return output_path

    def create_animated_captions(
        self,
        video_path: str,
        words: List[CaptionWord],
        animation: TextAnimation = TextAnimation.POP,
        style: Optional[TextStyle] = None,
        output_path: Optional[str] = None,
    ) -> str:
        if not style:
            style = TextStyle()
        if not output_path:
            output_path = os.path.join(self.temp_dir, f"captions_{uuid.uuid4().hex[:8]}.mp4")

        # Build ASS file for word-level captions
        ass_path = os.path.join(self.temp_dir, f"word_caps_{uuid.uuid4().hex[:8]}.ass")
        ass_color = self._hex_to_ass(style.color)
        ass_outline = self._hex_to_ass(style.stroke_color)

        header = f"""[Script Info]
Title: Word Captions
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{style.font},{style.font_size},{ass_color},{self._hex_to_ass(style.secondary_color)},{ass_outline},{self._hex_to_ass(style.shadow_color)},{int(style.bold)},{int(style.italic)},0,0,100,100,0,0,1,{style.stroke_width},{style.shadow_offset},2,40,40,80,1
Style: Highlight,{style.font},{style.font_size},&H0000FFFF&,&H00FFFFFF&,{ass_outline},{self._hex_to_ass(style.shadow_color)},{int(style.bold)},0,0,0,100,100,0,0,1,{style.stroke_width},{style.shadow_offset},2,40,40,80,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"""

        lines = [header]
        for word in words:
            start = self._seconds_to_ass_time(word.start_time)
            end = self._seconds_to_ass_time(word.end_time)
            style_name = "Highlight" if word.is_highlighted else "Default"
            lines.append(f"Dialogue: 0,{start},{end},{style_name},,0,0,0,,{word.word}")

        with open(ass_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        args = [
            "-i", video_path,
            "-vf", f"ass={ass_path}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "copy",
            "-pix_fmt", "yuv420p",
            output_path
        ]

        result = self._run_ffmpeg(args)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to create animated captions: {result.stderr}")
        return output_path

    def split_text_to_captions(
        self,
        text: str,
        words_per_caption: int = 6,
        duration_per_word: float = 0.4,
        start_time: float = 0,
    ) -> List[CaptionWord]:
        words = text.split()
        captions = []
        current_time = start_time

        for i, word in enumerate(words):
            word_duration = duration_per_word
            is_highlighted = (i % words_per_caption) == (words_per_caption - 1)
            captions.append(CaptionWord(
                word=word,
                start_time=current_time,
                end_time=current_time + word_duration,
                is_highlighted=is_highlighted,
            ))
            current_time += word_duration

        return captions

    def position_text(
        self,
        text_alignment: TextAlignment,
        text_width: int,
        text_height: int,
        video_width: int = 1080,
        video_height: int = 1920,
        margin: int = 40,
    ) -> Tuple[int, int]:
        positions = {
            TextAlignment.TOP_LEFT: (margin, margin),
            TextAlignment.TOP_CENTER: ((video_width - text_width) // 2, margin),
            TextAlignment.TOP_RIGHT: (video_width - text_width - margin, margin),
            TextAlignment.CENTER_LEFT: (margin, (video_height - text_height) // 2),
            TextAlignment.CENTER: ((video_width - text_width) // 2, (video_height - text_height) // 2),
            TextAlignment.RIGHT: (video_width - text_width - margin, (video_height - text_height) // 2),
            TextAlignment.BOTTOM_LEFT: (margin, video_height - text_height - margin),
            TextAlignment.BOTTOM_CENTER: ((video_width - text_width) // 2, video_height - text_height - margin),
            TextAlignment.BOTTOM_RIGHT: (video_width - text_width - margin, video_height - text_height - margin),
        }
        return positions.get(text_alignment, ((video_width - text_width) // 2, video_height - text_height - margin))

    def auto_resize_text(
        self,
        text: str,
        max_width: int = 1000,
        max_font_size: int = 80,
        min_font_size: int = 24,
    ) -> int:
        font_size = max_font_size
        chars_per_line = max_width // (font_size * 0.6)
        while font_size > min_font_size:
            chars_per_line = max_width // (font_size * 0.6)
            lines = len(text) / chars_per_line
            if lines <= 3:
                break
            font_size -= 4
        return max(font_size, min_font_size)
