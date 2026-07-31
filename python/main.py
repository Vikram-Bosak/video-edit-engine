"""Main entry point for the video editing engine."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt, stream=sys.stderr)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def load_config(path: str, config: "PipelineConfig") -> "PipelineConfig":
    import json

    try:
        with open(path) as f:
            cfg_data = json.load(f)
        config.platform = cfg_data.get("platform", config.platform)
        config.enable_subtitles = cfg_data.get("enable_subtitles", True)
        config.enable_color_grading = cfg_data.get("enable_color_grading", True)
        config.enable_motion_effects = cfg_data.get("enable_motion_effects", True)
        config.quality_preset = cfg_data.get("quality", config.quality_preset)
    except Exception as e:
        print(f"  [WARN] Could not load config {path}: {e}")
    return config


def load_template(path: str, config: "PipelineConfig") -> "PipelineConfig":
    import json

    try:
        with open(path) as f:
            template = json.load(f)
        config.platform = template.get("platform", config.platform)
        config.quality_preset = template.get("quality", config.quality_preset)
        config.enable_subtitles = template.get("enable_subtitles", True)
        config.enable_color_grading = template.get("enable_color_grading", True)
        config.enable_motion_effects = template.get("enable_motion_effects", True)
    except Exception as e:
        print(f"  [WARN] Could not load template {path}: {e}")
    return config


def generate_report(output_dir: str, report_dir: str) -> bool:
    import glob

    from python.processors.batch_processor import ProcessResult, BatchResult
    from python.processors.reporter import ReportGenerator
    from datetime import datetime

    r = ReportGenerator(report_dir)
    videos = glob.glob(os.path.join(output_dir, "**", "*.mp4"), recursive=True)
    results = []
    for v in videos:
        results.append(ProcessResult(
            video_path=v,
            output_path=v,
            success=True,
            file_size_mb=os.path.getsize(v) / (1024 * 1024),
        ))

    batch = BatchResult(
        batch_id=f"run_{int(time.time())}",
        results=results,
        total=len(videos),
        success_count=len(videos),
        failure_count=0,
        start_time=datetime.now().isoformat(),
        end_time=datetime.now().isoformat(),
    )

    if results:
        html_path = r.generate_html_report(batch, report_dir)
        json_path = r.generate_batch_report(batch, report_dir)
        print(f"  Report: {html_path}")
        return True
    else:
        print("  No videos found in output directory")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="AI Video Edit Engine - Automated social media video production",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m python.main --input video.mp4 --platform tiktok
  python -m python.main --input assets/videos/ --batch --platform youtube_short
  python -m python.main --url "https://youtube.com/watch?v=xxx" --platform instagram_reel
        """,
    )
    parser.add_argument("--input", "-i", help="Input video file or directory")
    parser.add_argument("--output", "-o", default="output", help="Output directory")
    parser.add_argument("--platform", "-p", default="tiktok",
                        choices=["tiktok", "youtube_short", "facebook_reel", "instagram_reel", "twitter_video"],
                        help="Target platform")
    parser.add_argument("--quality", "-q", default="high",
                        choices=["low", "medium", "high", "ultra"],
                        help="Quality preset")
    parser.add_argument("--config", "-c", help="Configuration file path")
    parser.add_argument("--template", "-t", help="Template file path")
    parser.add_argument("--batch", "-b", action="store_true", help="Batch process directory")
    parser.add_argument("--url", "-u", help="Video URL to download and process")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--report-only", action="store_true",
                        help="Generate report from existing output")
    parser.add_argument("--report-dir", default="reports",
                        help="Directory for reports")

    args = parser.parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger("main")

    from python.processors.pipeline import VideoPipeline, PipelineConfig
    from python.utils.logger import setup_logger

    setup_logger("video_engine", verbose=args.verbose)

    print(f"\n{'='*60}")
    print(f"  AI Video Edit Engine v1.0")
    print(f"  Platform: {args.platform}")
    print(f"  Quality: {args.quality}")
    print(f"{'='*60}\n")

    if args.report_only:
        success = generate_report(args.output, args.report_dir)
        sys.exit(0 if success else 1)

    config = PipelineConfig(
        platform=args.platform,
        output_dir=args.output,
        quality_preset=args.quality,
    )

    if args.config:
        config = load_config(args.config, config)
        print(f"  Loaded config: {args.config}")
    if args.template:
        config = load_template(args.template, config)
        print(f"  Loaded template: {args.template}")

    pipeline = VideoPipeline()

    start = time.time()
    if args.url:
        logger.info(f"Processing URL: {args.url}")
        result = pipeline.process_url(args.url, config)
    elif args.batch and args.input:
        if not os.path.isdir(args.input):
            print(f"  [ERROR] Directory not found: {args.input}")
            sys.exit(1)
        videos = [
            f for f in os.listdir(args.input)
            if os.path.splitext(f)[1].lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
        ]
        if not videos:
            print("  [INFO] No video files found in input directory")
            print("  [INFO] Use --url to process from a URL, or add files to the input directory")
            sys.exit(0)
        print(f"  Found {len(videos)} videos to process")
        logger.info(f"Batch processing: {args.input}")
        result = pipeline.process_batch(args.input, config)
    elif args.input:
        if not os.path.exists(args.input):
            print(f"  [ERROR] Input file not found: {args.input}")
            sys.exit(1)
        logger.info(f"Processing: {args.input}")
        result = pipeline.process_video(args.input, config)
    else:
        parser.print_help()
        sys.exit(1)

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"  Results")
    print(f"{'='*60}")
    print(f"  Status: {'SUCCESS' if result.success else 'FAILED'}")
    print(f"  Duration: {elapsed:.1f}s")
    if result.output_paths:
        print(f"  Outputs:")
        for p in result.output_paths:
            print(f"    - {p}")
    if result.errors:
        print(f"  Errors:")
        for e in result.errors:
            print(f"    - {e}")
    print(f"{'='*60}\n")

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
