"""Main video processing pipeline orchestrator."""

from __future__ import annotations

import logging
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    platform: str = "tiktok"
    output_dir: str = "output"
    quality_preset: str = "high"
    enable_subtitles: bool = True
    enable_color_grading: bool = True
    enable_motion_effects: bool = True
    template: Optional[str] = None
    custom_settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    success: bool = False
    output_paths: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    report_path: str = ""


class VideoPipeline:
    """Main pipeline that orchestrates the entire video editing process."""

    def __init__(self, temp_dir: Optional[str] = None):
        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix="pipeline_")
        os.makedirs(self.temp_dir, exist_ok=True)
        self._start_time = 0.0

    def _step(self, name: str, func, *args, **kwargs):
        logger.info(f"[STEP] {name}")
        step_start = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - step_start
            logger.info(f"[DONE] {name} ({elapsed:.1f}s)")
            return result
        except Exception as e:
            elapsed = time.time() - step_start
            logger.error(f"[FAIL] {name} ({elapsed:.1f}s): {e}")
            raise

    def _load_template(self, template_path: str) -> Dict[str, Any]:
        import json
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template not found: {template_path}")
        with open(template_path) as f:
            return json.load(f)

    def process_video(self, input_path: str, config: Optional[PipelineConfig] = None) -> PipelineResult:
        self._start_time = time.time()
        errors: List[str] = []
        output_paths: List[str] = []

        if not os.path.exists(input_path):
            return PipelineResult(success=False, errors=[f"Input file not found: {input_path}"])

        if config is None:
            config = PipelineConfig()
        os.makedirs(config.output_dir, exist_ok=True)

        try:
            from python.ai.orchestrator import AIOrchestrator, ExportPlatform
            orchestrator = AIOrchestrator(temp_dir=self.temp_dir)

            # Step 1: Analyze
            plan = self._step("Analyze & Plan", orchestrator.generate_full_edit, input_path, ExportPlatform(config.platform))

            # Step 2: Execute
            result = self._step("Execute Edit", orchestrator.execute_edit, plan, config.output_dir)

            if result.success:
                output_paths.append(result.output_path)
            else:
                errors.append(result.error)

        except Exception as e:
            errors.append(str(e))
            logger.error(f"Pipeline failed: {e}")

        elapsed = time.time() - self._start_time
        return PipelineResult(
            success=len(output_paths) > 0,
            output_paths=output_paths,
            errors=errors,
            duration_seconds=elapsed,
        )

    def process_url(self, url: str, config: Optional[PipelineConfig] = None) -> PipelineResult:
        try:
            from python.processors.video_analyzer import VideoDownloader
            downloader = VideoDownloader()
            dl_dir = os.path.join(self.temp_dir, "downloads")
            os.makedirs(dl_dir, exist_ok=True)
            downloaded = self._step("Download", downloader.download_video, url, dl_dir)
            return self.process_video(downloaded, config)
        except Exception as e:
            return PipelineResult(success=False, errors=[str(e)])

    def process_batch(self, input_dir: str, config: Optional[PipelineConfig] = None) -> PipelineResult:
        if config is None:
            config = PipelineConfig()
        try:
            from python.processors.batch_processor import BatchProcessor, BatchConfig
            bp = BatchProcessor(temp_dir=self.temp_dir)
            batch_config = BatchConfig(max_workers=2)
            batch_result = self._step(
                "Batch Process", bp.process_from_directory,
                input_dir, config.output_dir, config.platform, batch_config
            )
            output_paths = [r.output_path for r in batch_result.results if r.success]
            errors = [r.error for r in batch_result.results if not r.success and r.error]

            from python.processors.reporter import ReportGenerator
            reporter = ReportGenerator(config.output_dir)
            report_path = self._step("Generate Report", reporter.generate_html_report, batch_result, config.output_dir)

            return PipelineResult(
                success=batch_result.success_count > 0,
                output_paths=output_paths,
                errors=errors,
                duration_seconds=time.time() - self._start_time,
                report_path=report_path,
            )
        except Exception as e:
            return PipelineResult(success=False, errors=[str(e)])
