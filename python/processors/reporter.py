"""Report generator for batch processing results."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ReportSummary:
    total_videos: int = 0
    successful: int = 0
    failed: int = 0
    total_duration: float = 0.0
    total_file_size_mb: float = 0.0
    avg_processing_time: float = 0.0
    errors: List[str] = field(default_factory=list)


class ReportGenerator:
    """Generate reports for video processing results."""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir or os.path.join(os.getcwd(), "reports")
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_batch_report(self, batch_result: Any, output_dir: Optional[str] = None) -> str:
        dir_path = output_dir or self.output_dir
        os.makedirs(dir_path, exist_ok=True)
        report = self._build_report_data(batch_result)
        path = os.path.join(dir_path, f"batch_report_{batch_result.batch_id}.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Batch report saved to {path}")
        return path

    def generate_video_report(self, process_result: Any, output_dir: Optional[str] = None) -> str:
        dir_path = output_dir or self.output_dir
        os.makedirs(dir_path, exist_ok=True)
        report = asdict(process_result) if hasattr(process_result, '__dataclass_fields__') else {"result": str(process_result)}
        path = os.path.join(dir_path, f"video_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        return path

    def generate_html_report(self, batch_result: Any, output_dir: Optional[str] = None) -> str:
        dir_path = output_dir or self.output_dir
        os.makedirs(dir_path, exist_ok=True)
        summary = self.create_summary([batch_result])
        results = getattr(batch_result, 'results', [])

        rows = ""
        for r in results:
            status = '<span style="color:green">Success</span>' if r.success else '<span style="color:red">Failed</span>'
            rows += f"""
            <tr>
                <td>{os.path.basename(r.video_path)}</td>
                <td>{status}</td>
                <td>{r.duration_seconds:.1f}s</td>
                <td>{r.file_size_mb:.2f} MB</td>
                <td>{r.error or '-'}</td>
            </tr>"""

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Video Processing Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1000px; margin: 0 auto; background: white; border-radius: 12px; padding: 40px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #1a1a1a; border-bottom: 2px solid #007AFF; padding-bottom: 10px; }}
        .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 30px 0; }}
        .stat {{ background: #f8f9fa; border-radius: 8px; padding: 20px; text-align: center; }}
        .stat .value {{ font-size: 32px; font-weight: bold; color: #007AFF; }}
        .stat .label {{ color: #666; margin-top: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 30px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        tr:hover {{ background: #f8f9fa; }}
        .footer {{ margin-top: 30px; color: #999; font-size: 12px; text-align: center; }}
    </style>
</head>
<body>
<div class="container">
    <h1>Video Processing Report</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <div class="stats">
        <div class="stat"><div class="value">{summary.total_videos}</div><div class="label">Total Videos</div></div>
        <div class="stat"><div class="value">{summary.successful}</div><div class="label">Successful</div></div>
        <div class="stat"><div class="value">{summary.failed}</div><div class="label">Failed</div></div>
        <div class="stat"><div class="value">{summary.avg_processing_time:.1f}s</div><div class="label">Avg Time</div></div>
    </div>
    <table>
        <thead><tr><th>Video</th><th>Status</th><th>Duration</th><th>Size</th><th>Error</th></tr></thead>
        <tbody>{rows}</tbody>
    </table>
    <div class="footer">AI Video Edit Engine - Report Generator</div>
</div>
</body>
</html>"""

        path = os.path.join(dir_path, f"report_{batch_result.batch_id}.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info(f"HTML report saved to {path}")
        return path

    def generate_json_report(self, batch_result: Any, output_dir: Optional[str] = None) -> str:
        return self.generate_batch_report(batch_result, output_dir)

    def create_summary(self, batch_results: List[Any]) -> ReportSummary:
        summary = ReportSummary()
        all_errors = []
        total_time = 0.0
        total_size = 0.0

        for batch in batch_results:
            results = getattr(batch, 'results', [])
            summary.total_videos += getattr(batch, 'total', len(results))
            for r in results:
                if r.success:
                    summary.successful += 1
                else:
                    summary.failed += 1
                    if r.error:
                        all_errors.append(r.error)
                total_time += r.duration_seconds
                total_file_size_mb = getattr(r, 'file_size_mb', 0.0)
                total_size += total_file_size_mb

        summary.total_duration = total_time
        summary.total_file_size_mb = total_size
        summary.avg_processing_time = total_time / max(summary.total_videos, 1)
        summary.errors = all_errors[:20]
        return summary

    def _build_report_data(self, batch_result: Any) -> Dict[str, Any]:
        summary = self.create_summary([batch_result])
        return {
            "batch_id": getattr(batch_result, 'batch_id', 'unknown'),
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_videos": summary.total_videos,
                "successful": summary.successful,
                "failed": summary.failed,
                "total_duration_seconds": summary.total_duration,
                "total_file_size_mb": summary.total_file_size_mb,
                "avg_processing_time": summary.avg_processing_time,
            },
            "results": [asdict(r) if hasattr(r, '__dataclass_fields__') else str(r) for r in getattr(batch_result, 'results', [])],
            "errors": summary.errors,
        }
