"""
Audio Processing Engine for Video Editing.

Provides comprehensive audio manipulation capabilities including extraction,
mixing, normalization, beat detection, noise reduction, and more.
All operations are performed via ffmpeg subprocess calls.
"""

import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class AudioTrack:
    """Configuration for a single audio track in a mix."""

    path: str
    volume: float = 1.0
    start_time: float = 0.0
    end_time: Optional[float] = None
    fade_in: float = 0.0
    fade_out: float = 0.0
    pan: float = 0.0


@dataclass
class AudioLevels:
    """Measured audio level information."""

    peak_db: float
    rms_db: float
    lufs: float
    true_peak: float


@dataclass
class Beat:
    """Detected beat information."""

    timestamp: float
    strength: float
    bpm: float


class AudioEngineError(Exception):
    """Raised when an audio processing operation fails."""


def _run_ffmpeg(args: List[str], check: bool = True) -> subprocess.CompletedProcess:
    """Execute an ffmpeg command and return the result.

    Args:
        args: Command arguments starting after 'ffmpeg'.
        check: If True, raise on non-zero exit code.

    Returns:
        CompletedProcess instance.

    Raises:
        AudioEngineError: If ffmpeg execution fails.
    """
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + args
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=check,
            timeout=600,
        )
        return result
    except FileNotFoundError:
        raise AudioEngineError("ffmpeg is not installed or not found on PATH.")
    except subprocess.CalledProcessError as exc:
        raise AudioEngineError(
            f"ffmpeg exited with code {exc.returncode}: {exc.stderr.strip()}"
        ) from exc
    except subprocess.TimeoutExpired:
        raise AudioEngineError("ffmpeg command timed out after 600 seconds.")


def _run_ffprobe(args: List[str]) -> str:
    """Execute an ffprobe command and return stdout.

    Args:
        args: Command arguments starting after 'ffprobe'.

    Returns:
        Stdout string from ffprobe.

    Raises:
        AudioEngineError: If ffprobe execution fails.
    """
    cmd = ["ffprobe", "-hide_banner"] + args
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        return result.stdout
    except FileNotFoundError:
        raise AudioEngineError("ffprobe is not installed or not found on PATH.")
    except subprocess.CalledProcessError as exc:
        raise AudioEngineError(
            f"ffprobe exited with code {exc.returncode}: {exc.stderr.strip()}"
        ) from exc


def _ensure_parent_dir(path: str) -> None:
    """Create parent directories for *path* if they do not exist."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


class AudioEngine:
    """Audio processing engine backed by ffmpeg.

    Every public method validates inputs, invokes ffmpeg via subprocess,
    and returns meaningful results.  Temporary files created during
    processing are cleaned up automatically.
    """

    # ------------------------------------------------------------------ #
    #  Extraction / Mixing                                                #
    # ------------------------------------------------------------------ #

    def extract_audio(
        self, video_path: str, output_path: str
    ) -> str:
        """Extract the audio stream from a video file.

        Args:
            video_path: Path to the source video.
            output_path: Desired path for the extracted audio file.

        Returns:
            The *output_path* on success.

        Raises:
            AudioEngineError: On missing input or ffmpeg failure.
        """
        if not os.path.isfile(video_path):
            raise AudioEngineError(f"Video file not found: {video_path}")
        _ensure_parent_dir(output_path)

        _run_ffmpeg([
            "-i", video_path,
            "-vn",
            "-acodec", "copy",
            output_path,
        ])
        return output_path

    def mix_audio(
        self,
        video_path: str,
        audio_tracks: List[AudioTrack],
        output_path: str,
    ) -> str:
        """Mix multiple audio tracks onto a video.

        Each *AudioTrack* may specify volume, timing, fades, and pan.
        The original video audio is kept and mixed with the supplied tracks.

        Args:
            video_path: Source video file.
            audio_tracks: List of AudioTrack configurations.
            output_path: Destination video path.

        Returns:
            The *output_path* on success.
        """
        if not os.path.isfile(video_path):
            raise AudioEngineError(f"Video file not found: {video_path}")
        if not audio_tracks:
            raise AudioEngineError("At least one audio track is required.")
        _ensure_parent_dir(output_path)

        inputs: List[str] = ["-i", video_path]
        filter_parts: List[str] = []

        for idx, track in enumerate(audio_tracks):
            if not os.path.isfile(track.path):
                raise AudioEngineError(f"Audio track not found: {track.path}")
            inputs += ["-i", track.path]

            filters: List[str] = []
            filters.append(f"volume={track.volume}")

            if track.fade_in > 0:
                filters.append(f"afade=t=in:st=0:d={track.fade_in}")
            if track.end_time is not None:
                duration = track.end_time - track.start_time
                if duration > 0:
                    filters.append(f"atrim=0:{duration}")
                if track.fade_out > 0:
                    safe_dur = max(duration, track.fade_out)
                    start = safe_dur - track.fade_out
                    filters.append(f"afade=t=out:st={start}:d={track.fade_out}")

            pan_val = track.pan
            if pan_val != 0.0:
                left = max(0.0, 1.0 - abs(pan_val))
                right = max(0.0, 1.0 + pan_val) if pan_val < 0 else 1.0
                right = max(0.0, 1.0 - abs(pan_val)) if pan_val > 0 else right
                left = 1.0 if pan_val < 0 else left
                filters.append(f"stereotools=balance_in={-pan_val}")

            chain = ",".join(filters) if filters else "anull"
            filter_parts.append(f"[{idx + 1}:a]{chain}[a{idx}]")

        mix_inputs = "".join(f"[a{i}]" for i in range(len(audio_tracks)))
        filter_parts.append(
            f"[0:a]{mix_inputs}amix=inputs={len(audio_tracks) + 1}"
            f":duration=first:dropout_transition=2[outa]"
        )

        filter_graph = ";".join(filter_parts)

        _run_ffmpeg(
            inputs
            + [
                "-filter_complex", filter_graph,
                "-map", "0:v",
                "-map", "[outa]",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
                output_path,
            ]
        )
        return output_path

    # ------------------------------------------------------------------ #
    #  Background Music / Voiceover                                       #
    # ------------------------------------------------------------------ #

    def add_background_music(
        self,
        video_path: str,
        music_path: str,
        volume: float = 0.3,
        fade_in: float = 1.0,
        fade_out: float = 2.0,
        output_path: str = "",
    ) -> str:
        """Overlay background music onto a video.

        Args:
            video_path: Source video.
            music_path: Background music file.
            volume: Music volume multiplier (0.0 – 1.0+).
            fade_in: Fade-in duration in seconds.
            fade_out: Fade-out duration in seconds.
            output_path: Destination path.  Defaults to ``<video>_bgmusic.<ext>``.

        Returns:
            Path to the output video.
        """
        if not os.path.isfile(video_path):
            raise AudioEngineError(f"Video file not found: {video_path}")
        if not os.path.isfile(music_path):
            raise AudioEngineError(f"Music file not found: {music_path}")
        if not output_path:
            base, ext = os.path.splitext(video_path)
            output_path = f"{base}_bgmusic{ext}"
        _ensure_parent_dir(output_path)

        duration_json = _run_ffprobe([
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json",
            video_path,
        ])
        info = json.loads(duration_json)
        video_duration = float(info["format"]["duration"])

        filters: List[str] = [
            f"volume={volume}",
            f"afade=t=in:st=0:d={fade_in}",
        ]
        if fade_out > 0:
            fo_start = max(0.0, video_duration - fade_out)
            filters.append(f"afade=t=out:st={fo_start}:d={fade_out}")

        chain = ",".join(filters)

        _run_ffmpeg([
            "-i", video_path,
            "-i", music_path,
            "-filter_complex",
            f"[1:a]{chain},atrim=0:{video_duration},asetpts=PTS-STARTPTS[m];"
            f"[0:a][m]amix=inputs=2:duration=first:dropout_transition=2[outa]",
            "-map", "0:v",
            "-map", "[outa]",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            output_path,
        ])
        return output_path

    def add_voiceover(
        self,
        video_path: str,
        voiceover_path: str,
        volume: float = 1.0,
        start_time: float = 0.0,
        output_path: str = "",
    ) -> str:
        """Add a voiceover track to a video.

        Args:
            video_path: Source video.
            voiceover_path: Voiceover audio file.
            volume: Voiceover volume multiplier.
            start_time: Offset in seconds before the voiceover begins.
            output_path: Destination path.

        Returns:
            Path to the output video.
        """
        if not os.path.isfile(video_path):
            raise AudioEngineError(f"Video file not found: {video_path}")
        if not os.path.isfile(voiceover_path):
            raise AudioEngineError(f"Voiceover file not found: {voiceover_path}")
        if not output_path:
            base, ext = os.path.splitext(video_path)
            output_path = f"{base}_vo{ext}"
        _ensure_parent_dir(output_path)

        delay_ms = int(start_time * 1000)

        _run_ffmpeg([
            "-i", video_path,
            "-i", voiceover_path,
            "-filter_complex",
            (
                f"[1:a]volume={volume},adelay={delay_ms}|{delay_ms},apad=whole_dur=999[vo];"
                f"[0:a][vo]amix=inputs=2:duration=first:dropout_transition=2[outa]"
            ),
            "-map", "0:v",
            "-map", "[outa]",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            output_path,
        ])
        return output_path

    # ------------------------------------------------------------------ #
    #  Normalization / Silence                                            #
    # ------------------------------------------------------------------ #

    def normalize_audio(
        self,
        audio_path: str,
        target_lufs: float = -23.0,
        output_path: str = "",
    ) -> str:
        """Normalize audio using EBU R128 loudness standards.

        Args:
            audio_path: Input audio/video file.
            target_lufs: Target integrated loudness in LUFS.
            output_path: Destination path.  Defaults to ``<input>_normalized.<ext>``.

        Returns:
            Path to the normalized file.
        """
        if not os.path.isfile(audio_path):
            raise AudioEngineError(f"Audio file not found: {audio_path}")
        if not output_path:
            base, ext = os.path.splitext(audio_path)
            output_path = f"{base}_normalized{ext}"
        _ensure_parent_dir(output_path)

        _run_ffmpeg([
            "-i", audio_path,
            "-af", f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11:print_format=json",
            "-f", "null",
            "-",
        ], check=False)

        _run_ffmpeg([
            "-i", audio_path,
            "-af",
            f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11:print_format=json",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            output_path,
        ])
        return output_path

    def remove_silence(
        self,
        audio_path: str,
        threshold_db: float = -40.0,
        min_duration: float = 0.5,
        output_path: str = "",
    ) -> str:
        """Remove silent segments from an audio/video file.

        Args:
            audio_path: Input file.
            threshold_db: Silence threshold in dB.
            min_duration: Minimum silence duration (seconds) to be removed.
            output_path: Destination path.

        Returns:
            Path to the processed file.
        """
        if not os.path.isfile(audio_path):
            raise AudioEngineError(f"Audio file not found: {audio_path}")
        if not output_path:
            base, ext = os.path.splitext(audio_path)
            output_path = f"{base}_nosilence{ext}"
        _ensure_parent_dir(output_path)

        _run_ffmpeg([
            "-i", audio_path,
            "-af", (
                f"silenceremove="
                f"start_periods=1:start_duration=0:start_threshold={threshold_db}dB:"
                f"stop_periods=-1:stop_duration={min_duration}:stop_threshold={threshold_db}dB"
            ),
            "-c:v", "copy",
            output_path,
        ])
        return output_path

    # ------------------------------------------------------------------ #
    #  Ducking                                                            #
    # ------------------------------------------------------------------ #

    def duck_audio(
        self,
        video_path: str,
        voice_track_path: str,
        duck_amount: float = 12.0,
        output_path: str = "",
    ) -> str:
        """Duck (lower) the mix when a voice track is active.

        Uses sidechain compression so the background audio is attenuated
        whenever the voice track contains speech.

        Args:
            video_path: Source video (mixed audio).
            voice_track_path: Voice/speech track used as the ducking trigger.
            duck_amount: Attenuation in dB applied during ducking.
            output_path: Destination path.

        Returns:
            Path to the ducked output.
        """
        if not os.path.isfile(video_path):
            raise AudioEngineError(f"Video file not found: {video_path}")
        if not os.path.isfile(voice_track_path):
            raise AudioEngineError(f"Voice track not found: {voice_track_path}")
        if not output_path:
            base, ext = os.path.splitext(video_path)
            output_path = f"{base}_ducked{ext}"
        _ensure_parent_dir(output_path)

        ratio = max(1.0, duck_amount / 6.0)

        _run_ffmpeg([
            "-i", video_path,
            "-i", voice_track_path,
            "-filter_complex",
            (
                f"[1:a]asplit=2[sc][voc];"
                f"[sc]showspectrumpic=s=1x1:legend=0,anull[trigger];"
                f"[0:a][voc]sidechaincompress=threshold=0.02"
                f":ratio={ratio}:attack=20:release=200[ducked];"
                f"[ducked][voc]amix=inputs=2:duration=first:dropout_transition=2[outa]"
            ),
            "-map", "0:v",
            "-map", "[outa]",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            output_path,
        ])
        return output_path

    # ------------------------------------------------------------------ #
    #  Beat Detection                                                     #
    # ------------------------------------------------------------------ #

    def detect_beats(
        self, audio_path: str
    ) -> List[Beat]:
        """Detect beats in an audio file using energy-based onset analysis.

        Extracts the raw PCM waveform via ffmpeg, computes short-time
        energy, identifies peaks, and estimates BPM.

        Args:
            audio_path: Path to an audio or video file.

        Returns:
            List of Beat objects sorted by timestamp.
        """
        if not os.path.isfile(audio_path):
            raise AudioEngineError(f"Audio file not found: {audio_path}")

        with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as tmp:
            raw_path = tmp.name

        try:
            _run_ffmpeg([
                "-i", audio_path,
                "-vn",
                "-ac", "1",
                "-ar", "22050",
                "-f", "s16le",
                raw_path,
            ])

            raw_data = Path(raw_path).read_bytes()
            if len(raw_data) < 2:
                return []

            import numpy as np

            samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float64)
            samples /= 32768.0

            frame_size = 1024
            hop_size = 512
            sample_rate = 22050.0

            num_frames = max(1, (len(samples) - frame_size) // hop_size)
            energy = np.zeros(num_frames)
            for i in range(num_frames):
                start = i * hop_size
                frame = samples[start: start + frame_size]
                energy[i] = np.sum(frame ** 2) / len(frame)

            if num_frames < 3:
                return []

            mean_energy = np.mean(energy)
            threshold = mean_energy * 1.5

            peaks: List[int] = []
            min_distance = int(0.15 * sample_rate / hop_size)
            for i in range(1, num_frames - 1):
                if energy[i] > threshold and energy[i] > energy[i - 1] and energy[i] > energy[i + 1]:
                    if not peaks or (i - peaks[-1]) >= min_distance:
                        peaks.append(i)

            if len(peaks) < 2:
                return []

            intervals = np.diff(peaks)
            median_interval = float(np.median(intervals))
            avg_interval_frames = median_interval * hop_size
            beat_duration = avg_interval_frames / sample_rate
            bpm = 60.0 / beat_duration if beat_duration > 0 else 120.0

            beats: List[Beat] = []
            for peak_idx in peaks:
                ts = (peak_idx * hop_size) / sample_rate
                strength = float(energy[peak_idx] / (mean_energy + 1e-10))
                beats.append(Beat(timestamp=round(ts, 4), strength=round(strength, 4), bpm=round(bpm, 2)))

            return beats
        finally:
            if os.path.isfile(raw_path):
                os.unlink(raw_path)

    # ------------------------------------------------------------------ #
    #  Silence Detection                                                  #
    # ------------------------------------------------------------------ #

    def detect_silence(
        self,
        audio_path: str,
        threshold_db: float = -40.0,
        min_duration: float = 0.5,
    ) -> List[Tuple[float, float]]:
        """Detect silent segments in an audio/video file.

        Uses ffmpeg's ``silencedetect`` filter.

        Args:
            audio_path: Input file.
            threshold_db: Silence threshold in dB.
            min_duration: Minimum silence duration to report (seconds).

        Returns:
            List of ``(start, end)`` tuples in seconds.
        """
        if not os.path.isfile(audio_path):
            raise AudioEngineError(f"Audio file not found: {audio_path}")

        result = subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner",
                "-i", audio_path,
                "-af", f"silencedetect=noise={threshold_db}dB:d={min_duration}",
                "-f", "null", "-",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

        silence_ranges: List[Tuple[float, float]] = []
        starts: dict[int, float] = {}

        for line in result.stderr.splitlines():
            m_start = re.search(r"silence_start:\s*([\d.]+)", line)
            m_end = re.search(r"silence_end:\s*([\d.]+)\s*\|\s*silence_duration:\s*([\d.]+)", line)
            if m_start:
                idx = len(starts)
                starts[idx] = float(m_start.group(1))
            if m_end:
                end_ts = float(m_end.group(1))
                duration = float(m_end.group(2))
                start_ts = end_ts - duration
                silence_ranges.append((round(start_ts, 4), round(end_ts, 4)))

        return silence_ranges

    # ------------------------------------------------------------------ #
    #  Fading / Volume / EQ                                               #
    # ------------------------------------------------------------------ #

    def fade_audio(
        self,
        audio_path: str,
        fade_in_duration: float = 0.0,
        fade_out_duration: float = 0.0,
        output_path: str = "",
    ) -> str:
        """Apply fade-in and/or fade-out to an audio/video file.

        Args:
            audio_path: Input file.
            fade_in_duration: Fade-in length in seconds.
            fade_out_duration: Fade-out length in seconds.
            output_path: Destination path.

        Returns:
            Path to the output file.
        """
        if not os.path.isfile(audio_path):
            raise AudioEngineError(f"Audio file not found: {audio_path}")
        if not output_path:
            base, ext = os.path.splitext(audio_path)
            output_path = f"{base}_faded{ext}"
        _ensure_parent_dir(output_path)

        filters: List[str] = []
        if fade_in_duration > 0:
            filters.append(f"afade=t=in:st=0:d={fade_in_duration}")
        if fade_out_duration > 0:
            dur_json = _run_ffprobe([
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "json",
                audio_path,
            ])
            total_dur = float(json.loads(dur_json)["format"]["duration"])
            fo_start = max(0.0, total_dur - fade_out_duration)
            filters.append(f"afade=t=out:st={fo_start}:d={fade_out_duration}")

        if not filters:
            _run_ffmpeg(["-i", audio_path, "-c", "copy", output_path])
            return output_path

        af = ",".join(filters)
        _run_ffmpeg([
            "-i", audio_path,
            "-af", af,
            "-c:v", "copy",
            output_path,
        ])
        return output_path

    def adjust_volume(
        self,
        audio_path: str,
        volume_factor: float = 1.0,
        output_path: str = "",
    ) -> str:
        """Adjust the volume of an audio/video file.

        Args:
            audio_path: Input file.
            volume_factor: Multiplier (1.0 = unchanged, 2.0 = double).
            output_path: Destination path.

        Returns:
            Path to the output file.
        """
        if not os.path.isfile(audio_path):
            raise AudioEngineError(f"Audio file not found: {audio_path}")
        if not output_path:
            base, ext = os.path.splitext(audio_path)
            output_path = f"{base}_vol{ext}"
        _ensure_parent_dir(output_path)

        _run_ffmpeg([
            "-i", audio_path,
            "-af", f"volume={volume_factor}",
            "-c:v", "copy",
            output_path,
        ])
        return output_path

    def apply_equalizer(
        self,
        audio_path: str,
        bands: List[Tuple[float, float, float]],
        output_path: str = "",
    ) -> str:
        """Apply a parametric equalizer.

        Args:
            audio_path: Input file.
            bands: List of ``(frequency_hz, width_hz, gain_db)`` tuples.
            output_path: Destination path.

        Returns:
            Path to the output file.
        """
        if not os.path.isfile(audio_path):
            raise AudioEngineError(f"Audio file not found: {audio_path}")
        if not output_path:
            base, ext = os.path.splitext(audio_path)
            output_path = f"{base}_eq{ext}"
        _ensure_parent_dir(output_path)

        eq_filters = []
        for freq, width, gain in bands:
            eq_filters.append(f"equalizer=f={freq}:width_type=h:w={width}:g={gain}")

        af = ",".join(eq_filters) if eq_filters else "anull"
        _run_ffmpeg([
            "-i", audio_path,
            "-af", af,
            "-c:v", "copy",
            output_path,
        ])
        return output_path

    # ------------------------------------------------------------------ #
    #  Noise Reduction                                                    #
    # ------------------------------------------------------------------ #

    def reduce_noise(
        self, audio_path: str, output_path: str = ""
    ) -> str:
        """Apply basic noise reduction using ffmpeg's afftdn filter.

        Args:
            audio_path: Input file.
            output_path: Destination path.

        Returns:
            Path to the denoised file.
        """
        if not os.path.isfile(audio_path):
            raise AudioEngineError(f"Audio file not found: {audio_path}")
        if not output_path:
            base, ext = os.path.splitext(audio_path)
            output_path = f"{base}_denoised{ext}"
        _ensure_parent_dir(output_path)

        _run_ffmpeg([
            "-i", audio_path,
            "-af", "afftdn=nf=-25:tn=1:om=o",
            "-c:v", "copy",
            output_path,
        ])
        return output_path

    # ------------------------------------------------------------------ #
    #  Audio Levels                                                       #
    # ------------------------------------------------------------------ #

    def get_audio_levels(self, audio_path: str) -> AudioLevels:
        """Measure audio levels of a file.

        Reports peak, RMS, integrated LUFS (EBU R128), and true peak.

        Args:
            audio_path: Input audio or video file.

        Returns:
            AudioLevels dataclass with measured values.
        """
        if not os.path.isfile(audio_path):
            raise AudioEngineError(f"Audio file not found: {audio_path}")

        vol_json = _run_ffprobe([
            "-v", "error",
            "-show_entries", "frame_tags=lavfi.rms_level,lavfi.peak_level",
            "-f", "lavfi",
            "-i", f"amovie='{audio_path}',ebur128=metadata=1",
            "-of", "json",
        ])

        peak_db = -math_inf()
        rms_db = -math_inf()
        lufs = -23.0
        true_peak = -math_inf()

        try:
            data = json.loads(vol_json)
            for frame in data.get("frames", []):
                tags = frame.get("tags", {})
                if "lavfi.rms_level" in tags:
                    val = float(tags["lavfi.rms_level"])
                    rms_db = max(rms_db, val)
                if "lavfi.peak_level" in tags:
                    val = float(tags["lavfi.peak_level"])
                    peak_db = max(peak_db, val)
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

        loudness_json = subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner",
                "-i", audio_path,
                "-af", "loudnorm=print_format=json",
                "-f", "null", "-",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        json_match = re.search(
            r"\{[^{}]*\"input_i\"[^{}]*\}", loudness_json.stderr, re.DOTALL
        )
        if json_match:
            try:
                lm = json.loads(json_match.group(0))
                lufs = float(lm.get("input_i", -23.0))
                true_peak = float(lm.get("input_tp", -math_inf()))
            except (json.JSONDecodeError, ValueError):
                pass

        if peak_db == -math_inf():
            peak_db = -70.0
        if rms_db == -math_inf():
            rms_db = -70.0
        if true_peak == -math_inf():
            true_peak = peak_db

        return AudioLevels(
            peak_db=round(peak_db, 2),
            rms_db=round(rms_db, 2),
            lufs=round(lufs, 2),
            true_peak=round(true_peak, 2),
        )

    # ------------------------------------------------------------------ #
    #  Track Factory                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def create_audio_track(
        audio_path: str,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
        volume: float = 1.0,
        fade_in: float = 0.0,
        fade_out: float = 0.0,
    ) -> AudioTrack:
        """Create an AudioTrack configuration with sensible defaults.

        Args:
            audio_path: Path to the audio file.
            start_time: Offset into the timeline (seconds).
            end_time: When the track ends (None = full length).
            volume: Volume multiplier.
            fade_in: Fade-in duration (seconds).
            fade_out: Fade-out duration (seconds).

        Returns:
            A fully populated AudioTrack dataclass.
        """
        if not os.path.isfile(audio_path):
            raise AudioEngineError(f"Audio file not found: {audio_path}")
        return AudioTrack(
            path=os.path.abspath(audio_path),
            volume=volume,
            start_time=start_time,
            end_time=end_time,
            fade_in=fade_in,
            fade_out=fade_out,
            pan=0.0,
        )


def _math_inf() -> float:
    """Return negative infinity."""
    return float("-inf")


__all__ = [
    "AudioEngine",
    "AudioEngineError",
    "AudioTrack",
    "AudioLevels",
    "Beat",
]
