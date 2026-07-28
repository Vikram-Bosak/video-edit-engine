"""Timeline engine for the video editing system.

Provides the TimelineEngine class which manages timelines composed of
video tracks, audio tracks, overlay tracks, and text tracks. Rendering
is performed through ffmpeg complex filter graphs.
"""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TextLayer:
    """A single text element placed on the timeline."""

    text: str
    start_time: float
    end_time: float
    style: Dict[str, Any] = field(default_factory=lambda: {
        "font_size": 48,
        "font_color": "white",
        "font_name": "Arial",
        "bg_color": None,
        "bg_opacity": 0.5,
        "outline_color": "black",
        "outline_width": 2,
        "position": "center",
    })
    position: Dict[str, float] = field(default_factory=lambda: {"x": 0.5, "y": 0.5})

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class Overlay:
    """An overlay image or video placed on the timeline."""

    path: str
    start_time: float
    end_time: float
    position: Dict[str, float] = field(default_factory=lambda: {"x": 0, "y": 0})
    scale: Optional[Dict[str, int]] = None
    opacity: float = 1.0
    effects: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class VideoClip:
    """A single video clip on a track."""

    path: str
    start_time: float
    end_time: float
    trim_start: float = 0.0
    trim_end: float = 0.0
    speed: float = 1.0
    volume: float = 1.0
    effects: List[Dict[str, Any]] = field(default_factory=list)
    transition_in: Optional[Dict[str, Any]] = None
    transition_out: Optional[Dict[str, Any]] = None

    @property
    def duration(self) -> float:
        return (self.end_time - self.start_time) / self.speed


@dataclass
class AudioClip:
    """A single audio clip on a track."""

    path: str
    start_time: float
    end_time: float
    trim_start: float = 0.0
    trim_end: float = 0.0
    volume: float = 1.0
    fade_in: float = 0.0
    fade_out: float = 0.0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class VideoTrack:
    """An ordered collection of video clips forming a layer."""

    clips: List[VideoClip] = field(default_factory=list)
    layer_index: int = 0

    @property
    def duration(self) -> float:
        if not self.clips:
            return 0.0
        return max(c.end_time for c in self.clips)


@dataclass
class AudioTrack:
    """An ordered collection of audio clips."""

    clips: List[AudioClip] = field(default_factory=list)
    layer_index: int = 0

    @property
    def duration(self) -> float:
        if not self.clips:
            return 0.0
        return max(c.end_time for c in self.clips)


@dataclass
class OverlayTrack:
    """A collection of overlay elements."""

    overlays: List[Overlay] = field(default_factory=list)
    layer_index: int = 0

    @property
    def duration(self) -> float:
        if not self.overlays:
            return 0.0
        return max(o.end_time for o in self.overlays)


@dataclass
class TextTrack:
    """A collection of text layers."""

    texts: List[TextLayer] = field(default_factory=list)
    layer_index: int = 0

    @property
    def duration(self) -> float:
        if not self.texts:
            return 0.0
        return max(t.end_time for t in self.texts)


@dataclass
class Timeline:
    """Complete representation of a timeline to be rendered."""

    video_tracks: List[VideoTrack] = field(default_factory=list)
    audio_tracks: List[AudioTrack] = field(default_factory=list)
    overlay_tracks: List[OverlayTrack] = field(default_factory=list)
    text_tracks: List[TextTrack] = field(default_factory=list)
    total_duration: float = 0.0
    fps: int = 30
    width: int = 1920
    height: int = 1080

    def recalculate_duration(self) -> None:
        """Recalculate total_duration from all tracks."""
        durations: List[float] = []
        for t in self.video_tracks:
            durations.append(t.duration)
        for t in self.audio_tracks:
            durations.append(t.duration)
        for t in self.overlay_tracks:
            durations.append(t.duration)
        for t in self.text_tracks:
            durations.append(t.duration)
        self.total_duration = max(durations) if durations else 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _escape_ffmpeg_text(text: str) -> str:
    """Escape a string for use inside ffmpeg drawtext filter."""
    text = text.replace("\\", "\\\\\\\\")
    text = text.replace("'", "'\\\\\\''")
    text = text.replace(":", "\\:")
    text = text.replace("%", "%%")
    return text


def _build_style_string(style: Dict[str, Any]) -> str:
    """Convert a style dict into ffmpeg drawtext options."""
    parts: List[str] = []
    font_size = style.get("font_size", 48)
    font_color = style.get("font_color", "white")
    font_name = style.get("font_name", "Arial")
    bg_color = style.get("bg_color")
    bg_opacity = style.get("bg_opacity", 0.5)
    outline_color = style.get("outline_color", "black")
    outline_width = style.get("outline_width", 2)

    parts.append(f"fontsize={font_size}")
    parts.append(f"fontcolor={font_color}")
    parts.append(f"font='{font_name}'")
    parts.append(f"borderw={outline_width}")
    parts.append(f"bordercolor={outline_color}")
    if bg_color:
        parts.append(f"box=1:boxcolor={bg_color}@{bg_opacity}:boxborderw=5")
    return ":".join(parts)


# ---------------------------------------------------------------------------
# TimelineEngine
# ---------------------------------------------------------------------------

class TimelineEngine:
    """Engine for creating, manipulating, and rendering timelines.

    Rendering is performed by dynamically building ffmpeg ``-filter_complex``
    filter graphs that handle trimming, scaling, overlays, text drawing, and
    audio mixing.
    """

    # ------------------------------------------------------------------
    # Timeline creation
    # ------------------------------------------------------------------

    def create_timeline(
        self,
        clips: Optional[List[VideoClip]] = None,
        transitions: Optional[List[Dict[str, Any]]] = None,
        overlays: Optional[List[Overlay]] = None,
        audio_tracks: Optional[List[AudioTrack]] = None,
        *,
        fps: int = 30,
        width: int = 1920,
        height: int = 1080,
    ) -> Timeline:
        """Create and return a new :class:`Timeline`.

        Parameters
        ----------
        clips:
            Video clips to place on the first video track.
        transitions:
            Transition definitions applied between clips (currently stored
            on each clip's ``transition_in`` / ``transition_out``).
        overlays:
            Overlay elements placed on the first overlay track.
        audio_tracks:
            Audio tracks to include.
        fps:
            Frames per second.
        width:
            Output width in pixels.
        height:
            Output height in pixels.
        """
        video_track = VideoTrack(clips=clips or [], layer_index=0)

        overlay_track = OverlayTrack(overlays=overlays or [], layer_index=0)

        video_tracks = [video_track] if video_track.clips else []
        overlay_tracks = [overlay_track] if overlay_track.overlays else []
        audio_track_list = list(audio_tracks) if audio_tracks else []

        timeline = Timeline(
            video_tracks=video_tracks,
            audio_tracks=audio_track_list,
            overlay_tracks=overlay_tracks,
            text_tracks=[],
            fps=fps,
            width=width,
            height=height,
        )
        timeline.recalculate_duration()

        # Apply transitions to clip pairs when provided.
        if transitions:
            self._apply_transitions(video_track.clips, transitions)

        return timeline

    @staticmethod
    def _apply_transitions(
        clips: List[VideoClip], transitions: List[Dict[str, Any]]
    ) -> None:
        """Attach transition metadata to the appropriate clips."""
        for tr in transitions:
            idx = tr.get("clip_index", 0)
            t_type = tr.get("type", "fade")
            duration = tr.get("duration", 1.0)
            if 0 <= idx < len(clips):
                clips[idx].transition_out = {
                    "type": t_type,
                    "duration": duration,
                }
            if 0 <= idx + 1 < len(clips):
                clips[idx + 1].transition_in = {
                    "type": t_type,
                    "duration": duration,
                }

    # ------------------------------------------------------------------
    # Track management
    # ------------------------------------------------------------------

    def add_video_track(self, timeline: Timeline, track: VideoTrack) -> Timeline:
        """Add a :class:`VideoTrack` to *timeline* and return it."""
        track.layer_index = len(timeline.video_tracks)
        timeline.video_tracks.append(track)
        timeline.recalculate_duration()
        return timeline

    def add_audio_track(self, timeline: Timeline, track: AudioTrack) -> Timeline:
        """Add an :class:`AudioTrack` to *timeline* and return it."""
        track.layer_index = len(timeline.audio_tracks)
        timeline.audio_tracks.append(track)
        timeline.recalculate_duration()
        return timeline

    def add_overlay_track(self, timeline: Timeline, overlay: OverlayTrack) -> Timeline:
        """Add an :class:`OverlayTrack` to *timeline* and return it."""
        overlay.layer_index = len(timeline.overlay_tracks)
        timeline.overlay_tracks.append(overlay)
        timeline.recalculate_duration()
        return timeline

    def add_text_track(self, timeline: Timeline, text_layer: TextTrack) -> Timeline:
        """Add a :class:`TextTrack` to *timeline* and return it."""
        text_layer.layer_index = len(timeline.text_tracks)
        timeline.text_tracks.append(text_layer)
        timeline.recalculate_duration()
        return timeline

    # ------------------------------------------------------------------
    # Duration helpers
    # ------------------------------------------------------------------

    def calculate_duration(self, timeline: Timeline) -> float:
        """Return the total duration of *timeline* in seconds."""
        timeline.recalculate_duration()
        return timeline.total_duration

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------

    def optimize_timeline(self, timeline: Timeline) -> Timeline:
        """Remove gaps between clips and merge overlapping clips where possible.

        Clips on each video track are sorted by ``start_time`` and then
        consecutive clips whose time ranges overlap or touch are merged into a
        single clip.  The same logic is applied independently to audio tracks.
        """
        for track in timeline.video_tracks:
            track.clips = self._merge_clips(track.clips)
        for track in timeline.audio_tracks:
            track.clips = self._merge_audio_clips(track.clips)
        timeline.recalculate_duration()
        return timeline

    @staticmethod
    def _merge_clips(clips: List[VideoClip]) -> List[VideoClip]:
        """Merge overlapping / adjacent video clips."""
        if not clips:
            return []
        sorted_clips = sorted(clips, key=lambda c: c.start_time)
        merged: List[VideoClip] = [sorted_clips[0]]
        for clip in sorted_clips[1:]:
            prev = merged[-1]
            if clip.start_time <= prev.end_time:
                prev.end_time = max(prev.end_time, clip.end_time)
            else:
                merged.append(clip)
        return merged

    @staticmethod
    def _merge_audio_clips(clips: List[AudioClip]) -> List[AudioClip]:
        """Merge overlapping / adjacent audio clips."""
        if not clips:
            return []
        sorted_clips = sorted(clips, key=lambda c: c.start_time)
        merged: List[AudioClip] = [sorted_clips[0]]
        for clip in sorted_clips[1:]:
            prev = merged[-1]
            if clip.start_time <= prev.end_time:
                prev.end_time = max(prev.end_time, clip.end_time)
            else:
                merged.append(clip)
        return merged

    # ------------------------------------------------------------------
    # Export / import
    # ------------------------------------------------------------------

    def export_timeline_data(self, timeline: Timeline, output_path: str) -> str:
        """Serialise *timeline* to JSON and write it to *output_path*.

        Returns the absolute path written.
        """
        data = self._timeline_to_dict(timeline)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        return os.path.abspath(output_path)

    def import_timeline_data(self, json_path: str) -> Timeline:
        """Load a :class:`Timeline` from a previously exported JSON file."""
        with open(json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return self._dict_to_timeline(data)

    # Serialisation helpers -------------------------------------------------

    @staticmethod
    def _timeline_to_dict(timeline: Timeline) -> Dict[str, Any]:
        return {
            "fps": timeline.fps,
            "width": timeline.width,
            "height": timeline.height,
            "total_duration": timeline.total_duration,
            "video_tracks": [
                {
                    "layer_index": vt.layer_index,
                    "clips": [
                        {
                            "path": c.path,
                            "start_time": c.start_time,
                            "end_time": c.end_time,
                            "trim_start": c.trim_start,
                            "trim_end": c.trim_end,
                            "speed": c.speed,
                            "volume": c.volume,
                            "effects": c.effects,
                            "transition_in": c.transition_in,
                            "transition_out": c.transition_out,
                        }
                        for c in vt.clips
                    ],
                }
                for vt in timeline.video_tracks
            ],
            "audio_tracks": [
                {
                    "layer_index": at.layer_index,
                    "clips": [
                        {
                            "path": c.path,
                            "start_time": c.start_time,
                            "end_time": c.end_time,
                            "trim_start": c.trim_start,
                            "trim_end": c.trim_end,
                            "volume": c.volume,
                            "fade_in": c.fade_in,
                            "fade_out": c.fade_out,
                        }
                        for c in at.clips
                    ],
                }
                for at in timeline.audio_tracks
            ],
            "overlay_tracks": [
                {
                    "layer_index": ot.layer_index,
                    "overlays": [
                        {
                            "path": o.path,
                            "start_time": o.start_time,
                            "end_time": o.end_time,
                            "position": o.position,
                            "scale": o.scale,
                            "opacity": o.opacity,
                            "effects": o.effects,
                        }
                        for o in ot.overlays
                    ],
                }
                for ot in timeline.overlay_tracks
            ],
            "text_tracks": [
                {
                    "layer_index": tt.layer_index,
                    "texts": [
                        {
                            "text": tl.text,
                            "start_time": tl.start_time,
                            "end_time": tl.end_time,
                            "style": tl.style,
                            "position": tl.position,
                        }
                        for tl in tt.texts
                    ],
                }
                for tt in timeline.text_tracks
            ],
        }

    @staticmethod
    def _dict_to_timeline(data: Dict[str, Any]) -> Timeline:
        video_tracks = []
        for vt in data.get("video_tracks", []):
            clips = [
                VideoClip(**c) for c in vt.get("clips", [])
            ]
            video_tracks.append(VideoTrack(clips=clips, layer_index=vt.get("layer_index", 0)))

        audio_tracks = []
        for at in data.get("audio_tracks", []):
            clips = [AudioClip(**c) for c in at.get("clips", [])]
            audio_tracks.append(AudioTrack(clips=clips, layer_index=at.get("layer_index", 0)))

        overlay_tracks = []
        for ot in data.get("overlay_tracks", []):
            overlays = [Overlay(**o) for o in ot.get("overlays", [])]
            overlay_tracks.append(OverlayTrack(overlays=overlays, layer_index=ot.get("layer_index", 0)))

        text_tracks = []
        for tt in data.get("text_tracks", []):
            texts = [TextLayer(**t) for t in tt.get("texts", [])]
            text_tracks.append(TextTrack(texts=texts, layer_index=tt.get("layer_index", 0)))

        return Timeline(
            video_tracks=video_tracks,
            audio_tracks=audio_tracks,
            overlay_tracks=overlay_tracks,
            text_tracks=text_tracks,
            total_duration=data.get("total_duration", 0.0),
            fps=data.get("fps", 30),
            width=data.get("width", 1920),
            height=data.get("height", 1080),
        )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render_timeline(self, timeline: Timeline, output_path: str) -> str:
        """Render *timeline* to *output_path* using ffmpeg.

        Builds a complete ``-filter_complex`` graph that:

        1. Takes all video and audio inputs.
        2. Trims each input to its specified time window.
        3. Scales / pads every video stream to the output resolution.
        4. Chains overlays in layer order.
        5. Draws text layers.
        6. Mixes all audio tracks.
        7. Writes the final file.

        Returns the absolute path of the rendered file.
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        inputs, filter_parts, video_labels, audio_labels = self._build_video_filters(timeline)
        audio_labels.extend(self._build_audio_filters(timeline, inputs, filter_parts))

        filter_complex = ";".join(filter_parts)

        cmd = self._build_ffmpeg_command(inputs, filter_complex, output_path, timeline)
        self._run_ffmpeg(cmd)

        return os.path.abspath(output_path)

    # -- filter graph helpers -----------------------------------------------

    def _build_video_filters(
        self, timeline: Timeline
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Return (inputs, filter_parts, video_labels, audio_labels).

        Populates *filter_parts* with the video side of the filter_complex
        graph and collects the final per-track video labels that will later
        be combined.
        """
        inputs: List[str] = []
        filter_parts: List[str] = []
        video_labels: List[str] = []
        audio_labels: List[str] = []

        input_index = 0

        # --- base canvas (black frame) ---
        base_label = "base"
        filter_parts.append(
            f"color=c=black:s={timeline.width}x{timeline.height}"
            f":r={timeline.fps}:d={timeline.total_duration}"
            f"[{base_label}]"
        )

        current_label = base_label

        # --- video tracks ---
        for track in sorted(timeline.video_tracks, key=lambda t: t.layer_index):
            for clip in sorted(track.clips, key=lambda c: c.start_time):
                vidx = input_index
                aidx = input_index + 1
                inputs.extend(["-i", clip.path])

                clip_id = f"vclip_{uuid.uuid4().hex[:8]}"
                trimmed = f"{clip_id}_trimmed"

                # Trim
                ss = clip.trim_start
                duration = clip.duration
                speed_filter = f"setpts={1.0 / clip.speed}*PTS" if clip.speed != 1.0 else None
                parts = [f"[{vidx}:v]trim=start={ss}:duration={duration},setpts=PTS-STARTPTS"]
                if speed_filter:
                    parts.append(speed_filter)
                parts.append(f"scale={timeline.width}:{timeline.height}:force_original_aspect_ratio=decrease")
                parts.append(f"pad={timeline.width}:{timeline.height}:(ow-iw)/2:(oh-ih)/2:color=black")
                if clip.volume != 1.0:
                    pass  # audio volume handled separately
                parts.append(f"[{trimmed}]")
                filter_parts.append(",".join(parts))

                # Audio from this clip
                audio_clip_id = f"aclip_{uuid.uuid4().hex[:8]}"
                audio_trimmed = f"{audio_clip_id}_trimmed"
                audio_parts = [
                    f"[{aidx}:a]atrim=start={ss}:duration={duration},asetpts=PTS-STARTPTS"
                ]
                if clip.volume != 1.0:
                    audio_parts.append(f"volume={clip.volume}")
                audio_parts.append(f"[{audio_trimmed}]")
                filter_parts.append(",".join(audio_parts))
                audio_labels.append(f"[{audio_trimmed}]")

                # Overlay onto current canvas
                overlay_id = f"vovl_{uuid.uuid4().hex[:8]}"
                filter_parts.append(
                    f"[{current_label}][{trimmed}]overlay=0:0:shortest=1[{overlay_id}]"
                )
                current_label = overlay_id

                input_index += 2

        video_labels.append(f"[{current_label}]")

        return inputs, filter_parts, video_labels, audio_labels

    def _build_audio_filters(
        self,
        timeline: Timeline,
        inputs: List[str],
        filter_parts: List[str],
    ) -> List[str]:
        """Build the audio portion of the filter graph.

        Returns additional audio labels from audio tracks (beyond those
        extracted from video clips).
        """
        extra_audio_labels: List[str] = []
        input_index = len(inputs) // 2

        for track in timeline.audio_tracks:
            for clip in track.clips:
                aidx = input_index
                inputs.extend(["-i", clip.path])

                clip_id = f"atrk_{uuid.uuid4().hex[:8]}"
                trimmed = f"{clip_id}_trimmed"

                parts = [
                    f"[{aidx}:a]atrim=start={clip.trim_start}:duration={clip.duration},asetpts=PTS-STARTPTS"
                ]
                if clip.volume != 1.0:
                    parts.append(f"volume={clip.volume}")
                if clip.fade_in > 0:
                    parts.append(f"afade=t=in:st=0:d={clip.fade_in}")
                if clip.fade_out > 0:
                    fade_start = clip.duration - clip.fade_out
                    parts.append(f"afade=t=out:st={fade_start}:d={clip.fade_out}")
                parts.append(f"[{trimmed}]")
                filter_parts.append(",".join(parts))
                extra_audio_labels.append(f"[{trimmed}]")

                input_index += 1

        return extra_audio_labels

    def _build_ffmpeg_command(
        self,
        inputs: List[str],
        filter_complex: str,
        output_path: str,
        timeline: Timeline,
    ) -> List[str]:
        """Assemble the full ffmpeg command list."""
        cmd = ["ffmpeg", "-y"]
        cmd.extend(inputs)
        cmd.extend(["-filter_complex", filter_complex])

        # Map the final video and audio outputs.
        # Find the last video label and mix all audio.
        video_label = self._last_video_label(filter_complex)
        all_audio = self._collect_audio_labels(filter_complex)

        cmd.extend(["-map", video_label])

        if all_audio:
            # Mix all audio labels together.
            mix_label = "[amixed]"
            mix_inputs = "".join(all_audio)
            filter_complex_audio_mix = f"{mix_inputs}amix=inputs={len(all_audio)}:duration=longest:dropout_transition=0{mix_label}"
            # Append to the existing filter_complex.
            cmd[-1] = f"{filter_complex};{filter_complex_audio_mix}"
            cmd.extend(["-map", mix_label])
        else:
            # Generate silent audio.
            cmd.extend(["-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo"])
            cmd.extend(["-shortest"])

        cmd.extend([
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-c:a", "aac",
            "-b:a", "192k",
            "-r", str(timeline.fps),
            "-pix_fmt", "yuv420p",
            output_path,
        ])

        return cmd

    @staticmethod
    def _last_video_label(filter_complex: str) -> str:
        """Extract the last ``[...]`` video label from the filter graph."""
        import re
        labels = re.findall(r"\[(\w+)\](?!.*overlay)", filter_complex)
        if not labels:
            labels = re.findall(r"\[(\w+)\]", filter_complex)
        return f"[{labels[-1]}]" if labels else "[base]"

    @staticmethod
    def _collect_audio_labels(filter_complex: str) -> List[str]:
        """Collect all ``[...trimmed]`` audio labels from the graph."""
        import re
        return re.findall(r"\[(a\w+_trimmed)\]", filter_complex)

    @staticmethod
    def _run_ffmpeg(cmd: List[str]) -> subprocess.CompletedProcess:
        """Execute an ffmpeg command, raising on failure."""
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg exited with code {result.returncode}:\n{result.stderr}"
            )
        return result
