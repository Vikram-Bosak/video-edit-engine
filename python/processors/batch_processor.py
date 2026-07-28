"""Batch processor for handling multiple video files."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    max_workers: int = 2
    retry_count: int = 2
    timeout_seconds: int = 3600
    memory_limit_mb: int = 4096


@dataclass
class ProcessResult:
    video_path: str = ""
    output_path: str = ""
    success: bool = False
    error: str = ""
    duration_seconds: float = 0.0
    file_size_mb: float = 0.0


@dataclass
class BatchResult:
    batch_id: str = ""
    results: List[ProcessResult] = field(default_factory=list)
    total: int = 0
    success_count: int = 0
    failure_count: int = 0
    start_time: str = ""
    end_time: str = ""


@dataclass
class QueueStatus:
    pending: int = 0
    in_progress: int = 0
    completed: int = 0
    failed: int = 0


class BatchProcessor:
    """Process multiple videos concurrently with retry and resume support."""

    def __init__(self, temp_dir: Optional[str] = None):
        self.temp_dir = temp_dir or os.path.join(os.getcwd(), "temp", "batch")
        os.makedirs(self.temp_dir, exist_ok=True)
        self._state_file = os.path.join(self.temp_dir, "batch_state.json")
        self._queue_status = QueueStatus()

    def _save_state(self, batch_id: str, results: List[ProcessResult], pending: List[str]):
        state = {
            "batch_id": batch_id,
            "results": [asdict(r) for r in results],
            "pending": pending,
            "timestamp": datetime.now().isoformat(),
        }
        path = os.path.join(self.temp_dir, f"state_{batch_id}.json")
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self, batch_id: str) -> Dict[str, Any]:
        path = os.path.join(self.temp_dir, f"state_{batch_id}.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return {"batch_id": batch_id, "results": [], "pending": []}

    def process_video(
        self, video_path: str, output_dir: str, platform: str = "tiktok",
        config: Optional[BatchConfig] = None,
    ) -> ProcessResult:
        start = time.time()
        try:
            from python.ai.orchestrator import AIOrchestrator, ExportPlatform
            orchestrator = AIOrchestrator(temp_dir=self.temp_dir)
            platform_enum = ExportPlatform(platform)
            plan = orchestrator.generate_full_edit(video_path, platform_enum)
            os.makedirs(output_dir, exist_ok=True)
            result = orchestrator.execute_edit(plan, output_dir)

            elapsed = time.time() - start
            out_size = 0.0
            if result.success and os.path.exists(result.output_path):
                out_size = os.path.getsize(result.output_path) / (1024 * 1024)

            return ProcessResult(
                video_path=video_path,
                output_path=result.output_path if result.success else "",
                success=result.success,
                error=result.error,
                duration_seconds=elapsed,
                file_size_mb=out_size,
            )
        except Exception as e:
            return ProcessResult(
                video_path=video_path, success=False,
                error=str(e), duration_seconds=time.time() - start,
            )

    def process_batch(
        self, video_paths: List[str], output_dir: str, platform: str = "tiktok",
        config: Optional[BatchConfig] = None,
    ) -> BatchResult:
        if config is None:
            config = BatchConfig()
        batch_id = uuid.uuid4().hex[:12]
        os.makedirs(output_dir, exist_ok=True)
        results: List[ProcessResult] = []
        start_time = datetime.now()

        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            futures = {
                executor.submit(
                    self._process_with_retry, vp, output_dir, platform, config.retry_count
                ): vp
                for vp in video_paths
            }
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=config.timeout_seconds)
                    results.append(result)
                    self._save_state(batch_id, results, [])
                except Exception as e:
                    vp = futures[future]
                    results.append(ProcessResult(video_path=vp, error=str(e)))

        end_time = datetime.now()
        success_count = sum(1 for r in results if r.success)
        return BatchResult(
            batch_id=batch_id, results=results, total=len(video_paths),
            success_count=success_count, failure_count=len(video_paths) - success_count,
            start_time=start_time.isoformat(), end_time=end_time.isoformat(),
        )

    def _process_with_retry(self, video_path: str, output_dir: str, platform: str, retries: int) -> ProcessResult:
        last_error = None
        for attempt in range(retries + 1):
            result = self.process_video(video_path, output_dir, platform)
            if result.success:
                return result
            last_error = result.error
            logger.warning(f"Attempt {attempt + 1} failed for {video_path}: {last_error}")
        return result

    def process_from_directory(
        self, input_dir: str, output_dir: str, platform: str = "tiktok",
        config: Optional[BatchConfig] = None,
    ) -> BatchResult:
        extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
        video_paths = [
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if os.path.splitext(f)[1].lower() in extensions
        ]
        if not video_paths:
            logger.warning(f"No video files found in {input_dir}")
            return BatchResult(batch_id="empty", total=0)
        return self.process_batch(video_paths, output_dir, platform, config)

    def get_queue_status(self) -> QueueStatus:
        return self._queue_status

    def retry_failed(self, batch_result: BatchResult, output_dir: str) -> BatchResult:
        failed_paths = [r.video_path for r in batch_result.results if not r.success]
        if not failed_paths:
            return batch_result
        new_result = self.process_batch(failed_paths, output_dir)
        return new_result
