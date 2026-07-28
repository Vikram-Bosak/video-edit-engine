"""
Scene Detection Module for Video Editing Engine.

Provides content-based, motion-based, and audio-level scene detection
with multi-threaded processing and memory-efficient frame sampling.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from python.core.config import SceneConfig
except ImportError:

    @dataclass
    class SceneConfig:
        """Fallback configuration for scene detection."""

        default_threshold: float = 30.0
        min_scene_duration: float = 0.5
        max_scene_duration: float = 300.0
        highlight_threshold: float = 0.8
        motion_threshold: float = 25.0
        audio_threshold: float = 0.7
        sample_fps: float = 2.0
        max_threads: int = 4
        histogram_bins: int = 64
        merge_gap: float = 0.3

logger = logging.getLogger(__name__)

Callback = Optional[Callable[[str, float, Dict[str, Any]], None]]


@dataclass
class Scene:
    """Represents a detected scene in a video."""

    start_time: float
    end_time: float
    duration: float = field(init=False)
    score: float = 0.0
    is_highlight: bool = False
    motion_intensity: float = 0.0

    def __post_init__(self) -> None:
        self.duration = round(self.end_time - self.start_time, 4)

    def overlaps(self, other: Scene) -> bool:
        return self.start_time < other.end_time and other.start_time < self.end_time

    def contains(self, timestamp: float) -> bool:
        return self.start_time <= timestamp <= self.end_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "score": self.score,
            "is_highlight": self.is_highlight,
            "motion_intensity": self.motion_intensity,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Scene:
        return cls(
            start_time=data["start_time"],
            end_time=data["end_time"],
            score=data.get("score", 0.0),
            is_highlight=data.get("is_highlight", False),
            motion_intensity=data.get("motion_intensity", 0.0),
        )


class _FFmpegProbe:
    """Lightweight wrapper around ffprobe for media metadata."""

    @staticmethod
    def probe(video_path: str) -> Dict[str, Any]:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            video_path,
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            return json.loads(result.stdout)
        except FileNotFoundError:
            raise RuntimeError("ffprobe not found. Install ffmpeg and ensure it is on PATH.")
        except subprocess.TimeoutExpired:
            raise RuntimeError("ffprobe timed out while probing the video.")
        except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"ffprobe failed: {exc}")

    @staticmethod
    def get_duration(video_path: str) -> float:
        info = _FFmpegProbe.probe(video_path)
        try:
            return float(info["format"]["duration"])
        except (KeyError, TypeError, ValueError):
            raise RuntimeError(f"Could not determine duration for {video_path}")

    @staticmethod
    def get_video_stream(video_path: str) -> Dict[str, Any]:
        info = _FFmpegProbe.probe(video_path)
        for stream in info.get("streams", []):
            if stream.get("codec_type") == "video":
                return stream
        raise RuntimeError(f"No video stream found in {video_path}")

    @staticmethod
    def get_audio_stream(video_path: str) -> Optional[Dict[str, Any]]:
        info = _FFmpegProbe.probe(video_path)
        for stream in info.get("streams", []):
            if stream.get("codec_type") == "audio":
                return stream
        return None


class ContentDetector:
    """Histogram-based scene change detector using OpenCV."""

    def __init__(
        self,
        threshold: float = 30.0,
        histogram_bins: int = 64,
    ) -> None:
        self.threshold = threshold
        self.histogram_bins = histogram_bins

    def _compute_histogram(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv],
            [0, 1],
            None,
            [self.histogram_bins, self.histogram_bins],
            [0, 180, 0, 256],
        )
        cv2.normalize(hist, hist)
        return hist.flatten()

    def _histogram_distance(self, hist_a: np.ndarray, hist_b: np.ndarray) -> float:
        dist = cv2.compareHist(
            hist_a.astype(np.float32),
            hist_b.astype(np.float32),
            cv2.HISTCMP_CHISQR,
        )
        return float(dist)

    def detect(
        self,
        video_path: str,
        sample_fps: float = 2.0,
        progress_callback: Callback = None,
    ) -> List[Scene]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        try:
            native_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if native_fps <= 0 or total_frames <= 0:
                raise RuntimeError("Invalid video metadata.")

            frame_interval = max(1, int(native_fps / sample_fps))
            estimated_samples = total_frames // frame_interval

            scenes: List[Scene] = []
            prev_hist: Optional[np.ndarray] = None
            prev_frame_idx: int = 0
            frame_idx: int = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_interval == 0:
                    hist = self._compute_histogram(frame)
                    if prev_hist is not None:
                        distance = self._histogram_distance(prev_hist, hist)
                        if distance > self.threshold:
                            start_t = prev_frame_idx / native_fps
                            end_t = frame_idx / native_fps
                            scenes.append(Scene(start_time=start_t, end_time=end_t))
                            if progress_callback:
                                progress = frame_idx / max(total_frames, 1)
                                progress_callback(
                                    "content_detect",
                                    progress,
                                    {"scenes_found": len(scenes), "frame": frame_idx},
                                )
                    prev_hist = hist
                    prev_frame_idx = frame_idx

                frame_idx += 1

            duration = _FFmpegProbe.get_duration(video_path)
            if scenes:
                last_scene = scenes[-1]
                if last_scene.end_time < duration - 0.1:
                    scenes.append(
                        Scene(start_time=last_scene.end_time, end_time=duration)
                    )
                elif last_scene.end_time < duration:
                    scenes[-1] = Scene(
                        start_time=last_scene.start_time, end_time=duration
                    )
            else:
                scenes.append(Scene(start_time=0.0, end_time=duration))

            if progress_callback:
                progress_callback(
                    "content_detect", 1.0, {"scenes_found": len(scenes)}
                )

            return scenes

        finally:
            cap.release()


class MotionDetector:
    """Detects high-motion segments in video using optical flow magnitude."""

    def __init__(self, threshold: float = 25.0) -> None:
        self.threshold = threshold

    def _compute_motion_score(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return float(np.mean(magnitude))

    def detect(
        self,
        video_path: str,
        sample_fps: float = 2.0,
        progress_callback: Callback = None,
    ) -> List[Tuple[float, float]]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        try:
            native_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if native_fps <= 0 or total_frames <= 0:
                raise RuntimeError("Invalid video metadata.")

            frame_interval = max(1, int(native_fps / sample_fps))
            high_motion_ranges: List[Tuple[float, float]] = []
            in_high_motion = False
            motion_start: float = 0.0
            prev_gray: Optional[np.ndarray] = None
            frame_idx: int = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_interval == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if prev_gray is not None:
                        score = self._compute_motion_score(prev_gray, gray)
                        current_time = frame_idx / native_fps

                        if score > self.threshold and not in_high_motion:
                            in_high_motion = True
                            motion_start = current_time
                        elif score <= self.threshold and in_high_motion:
                            in_high_motion = False
                            high_motion_ranges.append((motion_start, current_time))

                        if progress_callback:
                            progress_callback(
                                "motion_detect",
                                frame_idx / max(total_frames, 1),
                                {"high_motion_ranges": len(high_motion_ranges)},
                            )

                    prev_gray = gray
                frame_idx += 1

            if in_high_motion:
                duration = _FFmpegProbe.get_duration(video_path)
                high_motion_ranges.append((motion_start, duration))

            if progress_callback:
                progress_callback(
                    "motion_detect", 1.0, {"high_motion_ranges": len(high_motion_ranges)}
                )

            return high_motion_ranges

        finally:
            cap.release()


class AudioLevelDetector:
    """Detects exciting audio moments by extracting loudness via ffmpeg."""

    @staticmethod
    def extract_audio_levels(
        video_path: str,
        window_seconds: float = 0.5,
    ) -> List[Tuple[float, float]]:
        """Extract mean volume per time window using ffmpeg volumedetect."""
        duration = _FFmpegProbe.get_duration(video_path)
        levels: List[Tuple[float, float]] = []
        tmp_dir = tempfile.mkdtemp(prefix="scene_audio_")

        try:
            wav_path = os.path.join(tmp_dir, "audio.wav")
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                video_path,
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                wav_path,
            ]
            subprocess.run(cmd, capture_output=True, check=True, timeout=120)

            data = np.frombuffer(open(wav_path, "rb").read(), dtype=np.int16)
            sample_rate = 16000
            samples_per_window = int(sample_rate * window_seconds)
            num_windows = int(math.ceil(len(data) / samples_per_window))

            for i in range(num_windows):
                chunk = data[i * samples_per_window : (i + 1) * samples_per_window]
                if len(chunk) == 0:
                    continue
                rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
                if rms > 0:
                    db = 20.0 * math.log10(rms / 32768.0 + 1e-10)
                else:
                    db = -100.0
                time_start = i * window_seconds
                levels.append((time_start, db))

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        return levels

    def detect(
        self,
        video_path: str,
        threshold_db: float = -20.0,
        progress_callback: Callback = None,
    ) -> List[Tuple[float, float]]:
        if progress_callback:
            progress_callback("audio_detect", 0.0, {"status": "extracting audio levels"})

        levels = self.extract_audio_levels(video_path)
        if not levels:
            return []

        if progress_callback:
            progress_callback("audio_detect", 0.5, {"windows": len(levels)})

        exciting_ranges: List[Tuple[float, float]] = []
        in_exciting = False
        exciting_start: float = 0.0
        window_duration = 0.5

        for time_pos, db in levels:
            if db > threshold_db and not in_exciting:
                in_exciting = True
                exciting_start = time_pos
            elif db <= threshold_db and in_exciting:
                in_exciting = False
                exciting_ranges.append((exciting_start, time_pos))

        if in_exciting:
            exciting_ranges.append((exciting_start, levels[-1][0] + window_duration))

        if progress_callback:
            progress_callback(
                "audio_detect", 1.0, {"exciting_segments": len(exciting_ranges)}
            )

        return exciting_ranges


class SceneDetector:
    """Main scene detection engine coordinating content, motion, and audio analysis."""

    def __init__(
        self,
        config: Optional[SceneConfig] = None,
        progress_callback: Callback = None,
    ) -> None:
        self.config = config or SceneConfig()
        self.progress_callback = progress_callback
        self._content_detector = ContentDetector(
            threshold=self.config.default_threshold,
            histogram_bins=self.config.histogram_bins,
        )
        self._motion_detector = MotionDetector(
            threshold=self.config.motion_threshold,
        )
        self._audio_detector = AudioLevelDetector()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_scenes(
        self,
        video_path: str,
        threshold: Optional[float] = None,
    ) -> List[Scene]:
        """Detect scene boundaries using histogram-based content analysis.

        Args:
            video_path: Path to the source video file.
            threshold: Histogram distance threshold. Uses config default if None.

        Returns:
            Ordered list of Scene objects covering the full video duration.
        """
        self._validate_path(video_path)
        effective_threshold = threshold if threshold is not None else self.config.default_threshold

        self._content_detector.threshold = effective_threshold
        scenes = self._content_detector.detect(
            video_path,
            sample_fps=self.config.sample_fps,
            progress_callback=self.progress_callback,
        )

        scenes = self._validate_and_fix_scenes(scenes, video_path)
        self._log(f"Detected {len(scenes)} scenes in {video_path}")
        return scenes

    def split_scenes(
        self,
        video_path: str,
        scenes: List[Scene],
        output_dir: Optional[str] = None,
    ) -> List[str]:
        """Split video into individual clips for each scene using ffmpeg.

        Args:
            video_path: Path to the source video file.
            scenes: List of Scene objects defining cut points.
            output_dir: Directory for output clips. Created if needed.

        Returns:
            List of file paths to the created clip files.
        """
        self._validate_path(video_path)
        if not scenes:
            return []

        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(video_path), "scene_clips")
        os.makedirs(output_dir, exist_ok=True)

        stem = Path(video_path).stem
        clip_paths: List[str] = []
        total = len(scenes)

        def _split_single(idx: int, scene: Scene) -> str:
            out_path = os.path.join(
                output_dir,
                f"{stem}_scene_{idx + 1:04d}.mp4",
            )
            duration = scene.end_time - scene.start_time
            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                str(scene.start_time),
                "-i",
                video_path,
                "-t",
                str(duration),
                "-c",
                "copy",
                "-avoid_negative_ts",
                "make_zero",
                out_path,
            ]
            try:
                subprocess.run(
                    cmd,
                    capture_output=True,
                    check=True,
                    timeout=max(30, int(duration * 3)),
                )
            except subprocess.TimeoutExpired:
                logger.warning("ffmpeg timed out splitting scene %d at %.2fs", idx + 1, scene.start_time)
                raise
            except subprocess.CalledProcessError as exc:
                logger.error("ffmpeg failed for scene %d: %s", idx + 1, exc.stderr.decode(errors="replace"))
                raise

            if self.progress_callback:
                self.progress_callback(
                    "split_scenes",
                    (idx + 1) / total,
                    {"clip": out_path, "scene_index": idx},
                )
            return out_path

        with ThreadPoolExecutor(max_workers=min(self.config.max_threads, total)) as pool:
            futures = {
                pool.submit(_split_single, i, s): i for i, s in enumerate(scenes)
            }
            results: Dict[int, str] = {}
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()

        for i in range(total):
            clip_paths.append(results[i])

        self._log(f"Split into {len(clip_paths)} clips in {output_dir}")
        return clip_paths

    def calculate_scene_scores(
        self,
        video_path: str,
        scenes: List[Scene],
    ) -> List[float]:
        """Calculate interestingness scores for each scene.

        Scoring factors:
            - Motion intensity (optical flow magnitude)
            - Scene duration relative to the video median
            - Audio loudness peaks within the scene window

        Args:
            video_path: Path to the source video file.
            scenes: List of Scene objects to score.

        Returns:
            List of float scores in [0, 1] aligned with input scenes.
        """
        self._validate_path(video_path)
        if not scenes:
            return []

        motion_ranges = self._motion_detector.detect(
            video_path,
            sample_fps=self.config.sample_fps,
            progress_callback=self.progress_callback,
        )

        audio_levels: List[Tuple[float, float]] = []
        try:
            audio_levels = self._audio_detector.extract_audio_levels(video_path)
        except Exception:
            self._log("Audio level extraction failed; scoring without audio data.", level=logging.WARNING)

        durations = [s.duration for s in scenes]
        median_dur = float(np.median(durations)) if durations else 1.0
        if median_dur <= 0:
            median_dur = 1.0

        max_audio_db = max((db for _, db in audio_levels), default=-60.0)
        min_audio_db = min((db for _, db in audio_levels), default=-100.0)
        audio_range = max(max_audio_db - min_audio_db, 1.0)

        scores: List[float] = []
        for scene in scenes:
            motion_score = self._compute_scene_motion(scene, motion_ranges)
            duration_score = self._duration_score(scene.duration, median_dur)
            audio_score = self._compute_scene_audio(scene, audio_levels, min_audio_db, audio_range)

            combined = 0.45 * motion_score + 0.25 * duration_score + 0.30 * audio_score
            combined = max(0.0, min(1.0, combined))
            scores.append(round(combined, 4))

        if self.progress_callback:
            self.progress_callback(
                "scene_scores", 1.0, {"scored_scenes": len(scores)}
            )

        self._log(f"Scored {len(scores)} scenes")
        return scores

    def remove_boring_scenes(
        self,
        scenes: List[Scene],
        min_score: float = 0.3,
    ) -> List[Scene]:
        """Filter out scenes with score below the given threshold.

        Adjacent boring scenes are merged into the next interesting scene
        to avoid fragmented timelines.

        Args:
            scenes: List of Scene objects with scores already assigned.
            min_score: Minimum score to keep a scene.

        Returns:
            Filtered list of scenes.
        """
        if not scenes:
            return []

        filtered: List[Scene] = []
        skip_buffer: Optional[Scene] = None

        for scene in scenes:
            if scene.score >= min_score:
                if skip_buffer is not None:
                    merged_start = skip_buffer.start_time
                    skip_buffer = None
                    scene = Scene(
                        start_time=merged_start,
                        end_time=scene.end_time,
                        score=max(scene.score, min_score),
                        is_highlight=scene.is_highlight,
                        motion_intensity=scene.motion_intensity,
                    )
                filtered.append(scene)
            else:
                if skip_buffer is None:
                    skip_buffer = scene
                else:
                    skip_buffer = Scene(
                        start_time=skip_buffer.start_time,
                        end_time=scene.end_time,
                        score=skip_buffer.score,
                    )

        if filtered:
            first = filtered[0]
            if skip_buffer is not None and skip_buffer.start_time < first.start_time:
                filtered[0] = Scene(
                    start_time=skip_buffer.start_time,
                    end_time=first.end_time,
                    score=first.score,
                    is_highlight=first.is_highlight,
                    motion_intensity=first.motion_intensity,
                )

        self._log(
            f"Removed {len(scenes) - len(filtered)} boring scenes "
            f"(threshold={min_score}), {len(filtered)} remaining"
        )
        return filtered

    def detect_highlights(
        self,
        video_path: str,
        threshold: Optional[float] = None,
    ) -> List[float]:
        """Detect highlight timestamps by combining motion and audio peaks.

        A highlight is a timestamp where both motion and audio intensity
        exceed their respective thresholds.

        Args:
            video_path: Path to the source video file.
            threshold: Combined intensity threshold in [0, 1]. Defaults to config value.

        Returns:
            List of highlight timestamps in seconds.
        """
        self._validate_path(video_path)
        eff_threshold = threshold if threshold is not None else self.config.highlight_threshold

        motion_ranges = self._motion_detector.detect(
            video_path,
            sample_fps=self.config.sample_fps,
            progress_callback=self.progress_callback,
        )

        audio_levels: List[Tuple[float, float]] = []
        try:
            audio_levels = self._audio_detector.extract_audio_levels(video_path)
        except Exception:
            self._log("Audio extraction failed for highlight detection.", level=logging.WARNING)

        duration = _FFmpegProbe.get_duration(video_path)
        sample_interval = 1.0 / self.config.sample_fps
        num_samples = int(math.ceil(duration / sample_interval))

        highlights: List[float] = []

        for i in range(num_samples):
            t = i * sample_interval
            if t >= duration:
                break

            motion_intensity = self._point_motion_intensity(t, motion_ranges)
            audio_intensity = self._point_audio_intensity(t, audio_levels)

            combined = 0.55 * motion_intensity + 0.45 * audio_intensity
            if combined >= eff_threshold:
                highlights.append(round(t, 3))

        highlights = self._cluster_timestamps(highlights, cluster_gap=1.0)

        if self.progress_callback:
            self.progress_callback(
                "highlights", 1.0, {"highlight_count": len(highlights)}
            )

        self._log(f"Detected {len(highlights)} highlight moments in {video_path}")
        return highlights

    def merge_short_scenes(
        self,
        scenes: List[Scene],
        min_duration: Optional[float] = None,
    ) -> List[Scene]:
        """Merge scenes shorter than min_duration into neighbours.

        Merges into whichever adjacent scene has the higher score.

        Args:
            scenes: List of Scene objects.
            min_duration: Minimum duration in seconds. Uses config default if None.

        Returns:
            Merged list of scenes.
        """
        if not scenes:
            return []

        eff_min = min_duration if min_duration is not None else self.config.min_scene_duration
        if eff_min <= 0:
            return list(scenes)

        merged: List[Scene] = [scenes[0]]

        for scene in scenes[1:]:
            prev = merged[-1]
            if prev.duration < eff_min:
                if prev.score >= scene.score:
                    merged[-1] = Scene(
                        start_time=prev.start_time,
                        end_time=scene.end_time,
                        score=prev.score,
                        is_highlight=prev.is_highlight or scene.is_highlight,
                        motion_intensity=max(prev.motion_intensity, scene.motion_intensity),
                    )
                else:
                    merged[-1] = Scene(
                        start_time=prev.start_time,
                        end_time=scene.end_time,
                        score=scene.score,
                        is_highlight=prev.is_highlight or scene.is_highlight,
                        motion_intensity=max(prev.motion_intensity, scene.motion_intensity),
                    )
            else:
                merged.append(scene)

        if len(merged) > 1:
            last = merged[-1]
            if last.duration < eff_min:
                prev = merged[-2]
                if prev.score >= last.score:
                    merged[-2] = Scene(
                        start_time=prev.start_time,
                        end_time=last.end_time,
                        score=prev.score,
                        is_highlight=prev.is_highlight or last.is_highlight,
                        motion_intensity=max(prev.motion_intensity, last.motion_intensity),
                    )
                else:
                    merged[-2] = Scene(
                        start_time=prev.start_time,
                        end_time=last.end_time,
                        score=last.score,
                        is_highlight=prev.is_highlight or last.is_highlight,
                        motion_intensity=max(prev.motion_intensity, last.motion_intensity),
                    )
                merged.pop()

        self._log(
            f"Merged short scenes: {len(scenes)} -> {len(merged)} "
            f"(min_duration={eff_min:.2f}s)"
        )
        return merged

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_path(self, video_path: str) -> None:
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

    def _validate_and_fix_scenes(
        self,
        scenes: List[Scene],
        video_path: str,
    ) -> List[Scene]:
        if not scenes:
            duration = _FFmpegProbe.get_duration(video_path)
            return [Scene(start_time=0.0, end_time=duration)]

        duration = _FFmpegProbe.get_duration(video_path)
        fixed: List[Scene] = []

        for scene in scenes:
            s = max(0.0, scene.start_time)
            e = min(duration, scene.end_time)
            if e - s > 0.01:
                fixed.append(Scene(start_time=round(s, 4), end_time=round(e, 4)))

        if not fixed:
            fixed.append(Scene(start_time=0.0, end_time=duration))
        elif fixed[0].start_time > 0.01:
            fixed.insert(
                0, Scene(start_time=0.0, end_time=fixed[0].start_time)
            )
        if fixed[-1].end_time < duration - 0.01:
            fixed.append(
                Scene(start_time=fixed[-1].end_time, end_time=duration)
            )

        return fixed

    def _compute_scene_motion(
        self,
        scene: Scene,
        motion_ranges: List[Tuple[float, float]],
    ) -> float:
        if not motion_ranges:
            return 0.0

        motion_duration = 0.0
        for start, end in motion_ranges:
            overlap_start = max(scene.start_time, start)
            overlap_end = min(scene.end_time, end)
            if overlap_start < overlap_end:
                motion_duration += overlap_end - overlap_start

        if scene.duration <= 0:
            return 0.0
        ratio = motion_duration / scene.duration
        return max(0.0, min(1.0, ratio))

    @staticmethod
    def _duration_score(duration: float, median_duration: float) -> float:
        if median_duration <= 0:
            return 0.5
        ratio = duration / median_duration
        if ratio < 0.2:
            return 0.2
        if ratio > 3.0:
            return 0.3
        return max(0.0, min(1.0, 0.5 + 0.3 * math.log2(ratio + 0.1)))

    def _compute_scene_audio(
        self,
        scene: Scene,
        audio_levels: List[Tuple[float, float]],
        min_db: float,
        audio_range: float,
    ) -> float:
        if not audio_levels:
            return 0.5

        relevant_dbs = [
            db
            for t, db in audio_levels
            if scene.start_time <= t <= scene.end_time
        ]

        if not relevant_dbs:
            return 0.5

        max_db = max(relevant_dbs)
        normalized = (max_db - min_db) / audio_range
        return max(0.0, min(1.0, normalized))

    def _point_motion_intensity(
        self,
        timestamp: float,
        motion_ranges: List[Tuple[float, float]],
    ) -> float:
        for start, end in motion_ranges:
            if start <= timestamp <= end:
                return 1.0
        for start, end in motion_ranges:
            dist = min(abs(timestamp - start), abs(timestamp - end))
            if dist < 2.0:
                return max(0.0, 1.0 - dist / 2.0) * 0.5
        return 0.0

    def _point_audio_intensity(
        self,
        timestamp: float,
        audio_levels: List[Tuple[float, float]],
    ) -> float:
        if not audio_levels:
            return 0.0
        closest_db = min(
            audio_levels,
            key=lambda x: abs(x[0] - timestamp),
            default=(0.0, -100.0),
        )
        _, db = closest_db
        normalized = (db + 100.0) / 100.0
        return max(0.0, min(1.0, normalized))

    @staticmethod
    def _cluster_timestamps(
        timestamps: List[float],
        cluster_gap: float = 1.0,
    ) -> List[float]:
        if not timestamps:
            return []
        clusters: List[List[float]] = [[timestamps[0]]]
        for t in timestamps[1:]:
            if t - clusters[-1][-1] <= cluster_gap:
                clusters[-1].append(t)
            else:
                clusters.append([t])
        return [round(float(np.mean(c)), 3) for c in clusters]

    def _log(self, msg: str, level: int = logging.INFO) -> None:
        logger.log(level, msg)
