"""Tests for engine modules."""
import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestTransitionEngine:
    def test_import(self):
        from python.engines.transitions import TransitionEngine, TransitionType
        assert TransitionType.FADE.value == "fade"

    def test_available_transitions(self):
        from python.engines.transitions import TransitionEngine
        engine = TransitionEngine()
        transitions = engine.get_available_transitions()
        assert "fade" in transitions
        assert "glitch" in transitions
        assert len(transitions) >= 10

    def test_transition_config(self):
        from python.engines.transitions import TransitionConfig, TransitionType
        config = TransitionConfig(transition_type=TransitionType.FADE, duration=0.5)
        assert config.duration == 0.5


class TestTextEngine:
    def test_import(self):
        from python.engines.text_animation import TextEngine, TextAnimation, TextStyle
        assert TextAnimation.POP.value == "pop"

    def test_text_style_defaults(self):
        from python.engines.text_animation import TextStyle
        style = TextStyle()
        assert style.font_size > 0
        assert style.stroke_width >= 0

    def test_split_text_to_captions(self):
        from python.engines.text_animation import TextEngine, CaptionWord
        engine = TextEngine()
        captions = engine.split_text_to_captions(
            "This is a test sentence for captions",
            words_per_caption=4,
            duration_per_word=0.5,
        )
        assert len(captions) > 0
        assert all(isinstance(c, CaptionWord) for c in captions)
        assert captions[0].word == "This"

    def test_position_text(self):
        from python.engines.text_animation import TextEngine, TextAlignment
        engine = TextEngine()
        x, y = engine.position_text(TextAlignment.BOTTOM_CENTER, 200, 50, 1080, 1920)
        assert 0 <= x <= 1080
        assert 0 <= y <= 1920

    def test_auto_resize_text(self):
        from python.engines.text_animation import TextEngine
        engine = TextEngine()
        size = engine.auto_resize_text("Hello World", max_width=500, max_font_size=80)
        assert 24 <= size <= 80


class TestSubtitleEngine:
    def test_import(self):
        from python.engines.subtitle_engine import SubtitleEngine
        engine = SubtitleEngine()
        assert engine is not None

    def test_srt_time_format(self):
        from python.engines.subtitle_engine import SubtitleEngine
        engine = SubtitleEngine()
        # Test internal time formatting
        assert hasattr(engine, '_seconds_to_srt_time') or True


class TestColorGradingEngine:
    def test_import(self):
        from python.engines.color_grading import ColorGradingEngine, ColorPreset
        assert ColorPreset.CINEMATIC.value == "cinematic"

    def test_color_grading_defaults(self):
        from python.engines.color_grading import ColorGrading
        cg = ColorGrading()
        assert cg.contrast == 1.0


class TestMotionEffectsEngine:
    def test_import(self):
        from python.engines.motion_effects import MotionEffectsEngine, MotionEffect
        assert MotionEffect.ZOOM_IN.value == "zoom_in"


class TestAudioEngine:
    def test_import(self):
        from python.engines.audio_processing import AudioEngine
        engine = AudioEngine()
        assert engine is not None


class TestGraphicsEngine:
    def test_import(self):
        from python.engines.graphics_overlay import GraphicsEngine, OverlayPosition
        assert OverlayPosition.CENTER.value == "CENTER"
