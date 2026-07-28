"""Tests for utility modules."""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestLogger:
    def test_setup_logger(self):
        from python.utils.logger import setup_logger
        logger = setup_logger("test_logger", verbose=False)
        assert logger is not None

    def test_logger_verbose(self):
        from python.utils.logger import setup_logger
        logger = setup_logger("test_verbose", verbose=True)
        assert logger is not None


class TestHelpers:
    def test_file_hash(self):
        from python.utils.helpers import compute_file_hash
        config_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
        if os.path.exists(config_path):
            hash_val = compute_file_hash(config_path)
            assert len(hash_val) == 64  # SHA256 hex

    def test_detect_file_type(self):
        from python.utils.helpers import detect_file_type
        result = detect_file_type("video.mp4")
        assert result in ("video", "audio", "image", "unknown", "subtitle", "config", "other")

    def test_ensure_directory(self):
        from python.utils.helpers import ensure_directory
        import tempfile
        test_dir = os.path.join(tempfile.mkdtemp(), "test", "nested")
        ensure_directory(test_dir)
        assert os.path.exists(test_dir)
