"""Tests for configuration module."""
import json
import os
import pytest


class TestResolution:
    def test_config_loads(self):
        config_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            assert "platforms" in config
            assert "tiktok" in config["platforms"]

    def test_platform_presets(self):
        config_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            assert config["platforms"]["tiktok"]["width"] == 1080
            assert config["platforms"]["tiktok"]["height"] == 1920
            assert config["platforms"]["youtube_short"]["max_duration"] == 60
            assert config["platforms"]["facebook_reel"]["max_duration"] == 90

    def test_scene_detection_config(self):
        config_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            assert config["scene_detection"]["threshold"] == 30.0

    def test_audio_config(self):
        config_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            assert config["audio"]["music_volume"] == 0.3
            assert config["audio"]["enable_ducking"] is True


class TestConfigSaveLoad:
    def test_save_json(self, tmp_path):
        config = {"platform": "tiktok", "quality": "high"}
        output = tmp_path / "test_config.json"
        with open(output, "w") as f:
            json.dump(config, f)
        assert output.exists()
        with open(output) as f:
            loaded = json.load(f)
        assert loaded == config

    def test_save_yaml(self, tmp_path):
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not installed")
        config = {"platform": "tiktok", "quality": "high"}
        output = tmp_path / "test_config.yaml"
        with open(output, "w") as f:
            yaml.dump(config, f)
        assert output.exists()
        with open(output) as f:
            loaded = yaml.safe_load(f)
        assert loaded == config
