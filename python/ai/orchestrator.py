"""AI orchestrator for automated video editing decisions."""

from __future__ import annotations

import logging
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    from python.core.config import (
        TransitionType, TextAnimation, MotionEffect, ColorPreset,
        CropAspect, ExportPlatform,
    )
except ImportError:
    class TransitionType(Enum):
        FADE = "fade"; FLASH = "flash"; WHIP = "whip"; BLUR = "blur"
        SLIDE = "slide"; ZOOM = "zoom"; WIPE_LEFT = "wipe_left"
    class TextAnimation(Enum):
        POP = "pop"; FADE = "fade"; BOUNCE = "bounce"; SCALE = "scale"
    class MotionEffect(Enum):
        ZOOM_IN = "zoom_in"; KEN_BURNS = "ken_burns"; CAMERA_SHAKE = "camera_shake"
    class ColorPreset(Enum):
        CINEMATIC = "cinematic"; WARM = "warm"; COOL = "cool"; VIVID = "vivid"
    class CropAspect(Enum):
        PORTRAIT_9_16 = "9:16"; SQUARE_1_1 = "1:1"; LANDSCAPE_16_9 = "16:9"
    class ExportPlatform(Enum):
        TIKTOK = "tiktok"; YOUTUBE_SHORT = "youtube_short"
        FACEBOOK_REEL = "facebook_reel"; INSTAGRAM_REEL = "instagram_reel"

logger = logging.getLogger(__name__)


@dataclass
class EditScene:
    start: float
    end: float
    score: float = 0.5
    effects: List[str] = field(default_factory=list)
    transition_in: TransitionType = TransitionType.FADE
    transition_out: TransitionType = TransitionType.FADE


@dataclass
class TransitionPlan:
    type: TransitionType = TransitionType.FADE
    duration: float = 0.5
    between_scenes: List[int] = field(default_factory=list)


@dataclass
class TextOverlayPlan:
    title: Optional[Dict[str, Any]] = None
    subtitles: bool = False
    lower_thirds: List[Dict[str, Any]] = field(default_factory=list)
    outro: Optional[Dict[str, Any]] = None
    word_highlight: bool = True


@dataclass
class MusicPlan:
    track_path: Optional[str] = None
    volume: float = 0.3
    fade_in: float = 1.0
    fade_out: float = 2.0
    ducking: bool = True


@dataclass
class EffectsPlan:
    motion_effects: List[Dict[str, Any]] = field(default_factory=list)
    color_preset: ColorPreset = ColorPreset.CINEMATIC
    overlays: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class CropPlan:
    target_aspect: CropAspect = CropAspect.PORTRAIT_9_16
    strategy: str = "smart"


@dataclass
class ColorPlan:
    preset: ColorPreset = ColorPreset.CINEMATIC
    custom_settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EditPlan:
    video_path: str = ""
    scenes: List[EditScene] = field(default_factory=list)
    transitions: List[TransitionPlan] = field(default_factory=list)
    text_overlay: TextOverlayPlan = field(default_factory=TextOverlayPlan)
    music: MusicPlan = field(default_factory=MusicPlan)
    effects: EffectsPlan = field(default_factory=EffectsPlan)
    crop: CropPlan = field(default_factory=CropPlan)
    color: ColorPlan = field(default_factory=ColorPlan)
    subtitles: bool = False
    motion_effects: List[Dict[str, Any]] = field(default_factory=list)
    export_settings: Dict[str, Any] = field(default_factory=dict)
    estimated_duration: float = 60.0


@dataclass
class EditResult:
    output_path: str = ""
    plan: Optional[EditPlan] = None
    success: bool = False
    error: str = ""
    duration: float = 0.0


PLATFORM_CONFIGS = {
    ExportPlatform.TIKTOK: {
        "aspect": CropAspect.PORTRAIT_9_16, "max_duration": 180,
        "text_size": 80, "transitions": [TransitionType.GLITCH, TransitionType.FLASH, TransitionType.WHIP],
        "color_preset": ColorPreset.VIVID, "music_volume": 0.3, "subtitles": True,
    },
    ExportPlatform.YOUTUBE_SHORT: {
        "aspect": CropAspect.PORTRAIT_9_16, "max_duration": 60,
        "text_size": 72, "transitions": [TransitionType.ZOOM, TransitionType.WHIP, TransitionType.FADE],
        "color_preset": ColorPreset.CINEMATIC, "music_volume": 0.25, "subtitles": True,
    },
    ExportPlatform.FACEBOOK_REEL: {
        "aspect": CropAspect.PORTRAIT_9_16, "max_duration": 90,
        "text_size": 68, "transitions": [TransitionType.FADE, TransitionType.SLIDE],
        "color_preset": ColorPreset.WARM, "music_volume": 0.3, "subtitles": True,
    },
    ExportPlatform.INSTAGRAM_REEL: {
        "aspect": CropAspect.PORTRAIT_9_16, "max_duration": 90,
        "text_size": 68, "transitions": [TransitionType.FADE, TransitionType.BLUR],
        "color_preset": ColorPreset.WARM, "music_volume": 0.3, "subtitles": True,
    },
}


class AIOrchestrator:
    """Makes automated editing decisions based on video analysis."""

    def __init__(self, temp_dir: Optional[str] = None):
        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix="ai_orchestrator_")
        os.makedirs(self.temp_dir, exist_ok=True)
        self._engines: Dict[str, Any] = {}

    def _get_engine(self, name: str) -> Any:
        if name not in self._engines:
            if name == "analyzer":
                from python.processors.video_analyzer import VideoAnalyzer
                self._engines[name] = VideoAnalyzer()
            elif name == "scene":
                from python.engines.scene_detection import SceneDetector
                self._engines[name] = SceneDetector()
            elif name == "crop":
                from python.engines.smart_crop import SmartCropper
                self._engines[name] = SmartCropper()
            elif name == "transitions":
                from python.engines.transitions import TransitionEngine
                self._engines[name] = TransitionEngine()
            elif name == "text":
                from python.engines.text_animation import TextEngine
                self._engines[name] = TextEngine()
            elif name == "graphics":
                from python.engines.graphics_overlay import GraphicsEngine
                self._engines[name] = GraphicsEngine()
            elif name == "audio":
                from python.engines.audio_processing import AudioEngine
                self._engines[name] = AudioEngine()
            elif name == "subtitle":
                from python.engines.subtitle_engine import SubtitleEngine
                self._engines[name] = SubtitleEngine()
            elif name == "color":
                from python.engines.color_grading import ColorGradingEngine
                self._engines[name] = ColorGradingEngine()
            elif name == "motion":
                from python.engines.motion_effects import MotionEffectsEngine
                self._engines[name] = MotionEffectsEngine()
            elif name == "export":
                from python.engines.export_engine import ExportEngine
                self._engines[name] = ExportEngine()
        return self._engines[name]

    def select_scenes(
        self, scenes: List[EditScene], target_duration: float
    ) -> List[EditScene]:
        sorted_scenes = sorted(scenes, key=lambda s: s.score, reverse=True)
        selected = []
        current_duration = 0.0
        for scene in sorted_scenes:
            dur = scene.end - scene.start
            if current_duration + dur <= target_duration * 1.1:
                selected.append(scene)
                current_duration += dur
            if current_duration >= target_duration:
                break
        selected.sort(key=lambda s: s.start)
        return selected if selected else sorted_scenes[:min(len(sorted_scenes), 10)]

    def choose_transitions(self, scenes: List[EditScene], platform: ExportPlatform) -> List[TransitionPlan]:
        config = PLATFORM_CONFIGS.get(platform, PLATFORM_CONFIGS[ExportPlatform.TIKTOK])
        available = config["transitions"]
        plans = []
        for i in range(len(scenes) - 1):
            t_type = available[i % len(available)]
            if scenes[i].score > 0.7:
                duration = 0.3
            elif scenes[i].score < 0.3:
                duration = 0.8
            else:
                duration = 0.5
            plans.append(TransitionPlan(type=t_type, duration=duration))
        return plans

    def choose_text_overlay(
        self, platform: ExportPlatform, has_speech: bool = False
    ) -> TextOverlayPlan:
        config = PLATFORM_CONFIGS.get(platform, PLATFORM_CONFIGS[ExportPlatform.TIKTOK])
        text_size = config.get("text_size", 72)
        return TextOverlayPlan(
            title={"text": "", "size": text_size, "animation": "pop"},
            subtitles=has_speech or config.get("subtitles", True),
            word_highlight=True,
        )

    def choose_music(self, audio_has_music: bool = False) -> MusicPlan:
        return MusicPlan(
            volume=0.25 if audio_has_music else 0.4,
            fade_in=1.0, fade_out=2.0, ducking=True,
        )

    def choose_effects(self, platform: ExportPlatform, video_duration: float) -> EffectsPlan:
        config = PLATFORM_CONFIGS.get(platform, PLATFORM_CONFIGS[ExportPlatform.TIKTOK])
        motion = []
        if video_duration > 30:
            motion.append({"type": "ken_burns", "intensity": 0.3})
        if video_duration > 10:
            motion.append({"type": "zoom_in", "intensity": 0.2})
        return EffectsPlan(motion_effects=motion, color_preset=config["color_preset"])

    def choose_crop(self, source_aspect: str, platform: ExportPlatform) -> CropPlan:
        config = PLATFORM_CONFIGS.get(platform, PLATFORM_CONFIGS[ExportPlatform.TIKTOK])
        return CropPlan(target_aspect=config["aspect"], strategy="smart")

    def choose_color(self, platform: ExportPlatform) -> ColorPlan:
        config = PLATFORM_CONFIGS.get(platform, PLATFORM_CONFIGS[ExportPlatform.TIKTOK])
        return ColorPlan(preset=config["color_preset"])

    def plan_edit(
        self, video_analysis: Any, platform: ExportPlatform = ExportPlatform.TIKTOK
    ) -> EditPlan:
        config = PLATFORM_CONFIGS.get(platform, PLATFORM_CONFIGS[ExportPlatform.TIKTOK])
        metadata = getattr(video_analysis, "metadata", None)
        duration = getattr(metadata, "duration", 60) if metadata else 60
        target_duration = min(duration, config["max_duration"])
        width = getattr(metadata, "width", 1080) if metadata else 1080
        height = getattr(metadata, "height", 1920) if metadata else 1920
        source_aspect = f"{width}:{height}"

        scenes = [
            EditScene(start=0, end=target_duration, score=0.5)
        ]

        plan = EditPlan(
            video_path="",
            scenes=scenes,
            transitions=[],
            text_overlay=self.choose_text_overlay(platform),
            music=self.choose_music(),
            effects=self.choose_effects(platform, target_duration),
            crop=self.choose_crop(source_aspect, platform),
            color=self.choose_color(platform),
            subtitles=config.get("subtitles", True),
            export_settings={"platform": platform.value},
            estimated_duration=target_duration,
        )

        plan.transitions = self.choose_transitions(plan.scenes, platform)
        return plan

    def generate_full_edit(self, video_path: str, platform: ExportPlatform = ExportPlatform.TIKTOK) -> EditPlan:
        analyzer = self._get_engine("analyzer")
        analysis = analyzer.analyze_video(video_path)
        plan = self.plan_edit(analysis, platform)
        plan.video_path = video_path
        return plan

    def execute_edit(self, plan: EditPlan, output_dir: Optional[str] = None) -> EditResult:
        if not output_dir:
            output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        try:
            current_path = plan.video_path
            output_path = os.path.join(output_dir, f"edited_{uuid.uuid4().hex[:8]}.mp4")

            # Step 1: Smart crop
            if plan.crop.strategy == "smart":
                try:
                    cropper = self._get_engine("crop")
                    cropped = os.path.join(self.temp_dir, f"cropped_{uuid.uuid4().hex[:8]}.mp4")
                    current_path = cropper.crop_video(
                        current_path, plan.crop.target_aspect, cropped
                    )
                    logger.info("Smart crop applied")
                except Exception as e:
                    logger.warning(f"Crop failed, continuing without: {e}")

            # Step 2: Color grading
            try:
                color_engine = self._get_engine("color")
                graded = os.path.join(self.temp_dir, f"graded_{uuid.uuid4().hex[:8]}.mp4")
                color_engine.apply_cinematic_look(
                    current_path, plan.color.preset.value, graded
                )
                current_path = graded
                logger.info("Color grading applied")
            except Exception as e:
                logger.warning(f"Color grading failed: {e}")

            # Step 3: Motion effects
            for effect in plan.motion_effects:
                try:
                    motion = self._get_engine("motion")
                    effected = os.path.join(self.temp_dir, f"motion_{uuid.uuid4().hex[:8]}.mp4")
                    if effect.get("type") == "zoom_in":
                        motion.apply_zoom_in(current_path, 1.0, 1.15, 0.5, 0.5, 3.0, 0, effected)
                    elif effect.get("type") == "ken_burns":
                        motion.apply_ken_burns(current_path, "center", "center", 1.0, 1.2, 5.0, effected)
                    current_path = effected
                    logger.info(f"Motion effect {effect['type']} applied")
                except Exception as e:
                    logger.warning(f"Motion effect failed: {e}")

            # Step 4: Text overlay
            if plan.text_overlay.title and plan.text_overlay.title.get("text"):
                try:
                    text_engine = self._get_engine("text")
                    titled = os.path.join(self.temp_dir, f"titled_{uuid.uuid4().hex[:8]}.mp4")
                    text_engine.burn_text_into_video(
                        current_path, plan.text_overlay.title["text"],
                        start_time=0, duration=3.0, output_path=titled
                    )
                    current_path = titled
                except Exception as e:
                    logger.warning(f"Text overlay failed: {e}")

            # Step 5: Audio processing
            if plan.music.track_path:
                try:
                    audio = self._get_engine("audio")
                    audio_out = os.path.join(self.temp_dir, f"audio_{uuid.uuid4().hex[:8]}.mp4")
                    audio.add_background_music(
                        current_path, plan.music.track_path,
                        plan.music.volume, plan.music.fade_in,
                        plan.music.fade_out, audio_out
                    )
                    current_path = audio_out
                except Exception as e:
                    logger.warning(f"Audio processing failed: {e}")

            # Step 6: Subtitles
            if plan.subtitles:
                logger.info("Subtitles requested (requires speech-to-text)")

            # Step 7: Export
            try:
                export = self._get_engine("export")
                platform_enum = ExportPlatform(plan.export_settings.get("platform", "tiktok"))
                result = export.export_for_platform(current_path, platform_enum, output_dir)
                if result:
                    final_path = list(result.values())[0] if isinstance(result, dict) else output_path
                else:
                    final_path = output_path
            except Exception:
                import shutil
                shutil.copy2(current_path, output_path)
                final_path = output_path

            return EditResult(
                output_path=final_path, plan=plan, success=True,
            )

        except Exception as e:
            logger.error(f"Edit execution failed: {e}")
            return EditResult(plan=plan, success=False, error=str(e))
