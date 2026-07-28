"""Integration tests for the pipeline."""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestPipeline:
    def test_import(self):
        from python.processors.pipeline import VideoPipeline, PipelineConfig, PipelineResult
        assert PipelineResult is not None

    def test_pipeline_config_defaults(self):
        from python.processors.pipeline import PipelineConfig
        config = PipelineConfig()
        assert config.platform == "tiktok"
        assert config.enable_subtitles is True

    def test_pipeline_nonexistent_file(self):
        from python.processors.pipeline import VideoPipeline, PipelineConfig
        pipeline = VideoPipeline()
        result = pipeline.process_video("/nonexistent/video.mp4")
        assert result.success is False
        assert len(result.errors) > 0


class TestBatchProcessor:
    def test_import(self):
        from python.processors.batch_processor import (
            BatchProcessor, BatchResult, ProcessResult, BatchConfig, QueueStatus
        )
        assert BatchProcessor is not None

    def test_queue_status(self):
        from python.processors.batch_processor import BatchProcessor
        bp = BatchProcessor()
        status = bp.get_queue_status()
        assert status.pending == 0


class TestReporter:
    def test_import(self):
        from python.processors.reporter import ReportGenerator, ReportSummary
        assert ReportGenerator is not None

    def test_report_summary(self):
        from python.processors.reporter import ReportSummary
        summary = ReportSummary(total_videos=10, successful=8, failed=2)
        assert summary.total_videos == 10
        assert summary.successful == 8


class TestOrchestrator:
    def test_import(self):
        from python.ai.orchestrator import AIOrchestrator, EditPlan
        assert AIOrchestrator is not None

    def test_edit_plan_defaults(self):
        from python.ai.orchestrator import EditPlan
        plan = EditPlan()
        assert plan.video_path == ""

    def test_scene_selection(self):
        from python.ai.orchestrator import AIOrchestrator, EditScene
        from python.core.config import ExportPlatform
        orchestrator = AIOrchestrator()
        scenes = [
            EditScene(start=0, end=5, score=0.8),
            EditScene(start=5, end=10, score=0.3),
            EditScene(start=10, end=15, score=0.9),
        ]
        selected = orchestrator.select_scenes(scenes, 10.0)
        assert len(selected) > 0


class TestVideoAnalyzer:
    def test_import(self):
        from python.processors.video_analyzer import VideoAnalyzer, VideoDownloader
        assert VideoAnalyzer is not None
