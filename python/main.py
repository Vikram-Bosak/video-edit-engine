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
    parser.add_argument("--config", "-c", help="Configuration file path")
    parser.add_argument("--template", "-t", help="Template file path")
    parser.add_argument("--batch", "-b", action="store_true", help="Batch process directory")
    parser.add_argument("--url", "-u", help="Video URL to download and process")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--report-only", action="store_true", help="Generate report only")

    args = parser.parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger("main")

    from python.processors.pipeline import VideoPipeline, PipelineConfig
    from python.utils.logger import setup_logger

    setup_logger("video_engine", verbose=args.verbose)

    config = PipelineConfig(
        platform=args.platform,
        output_dir=args.output,
    )

    if args.config:
        import json
        with open(args.config) as f:
            cfg_data = json.load(f)
        config.platform = cfg_data.get("platform", config.platform)
        config.enable_subtitles = cfg_data.get("enable_subtitles", True)
        config.enable_color_grading = cfg_data.get("enable_color_grading", True)
        config.enable_motion_effects = cfg_data.get("enable_motion_effects", True)

    pipeline = VideoPipeline()
    print(f"\n{'='*60}")
    print(f"  AI Video Edit Engine")
    print(f"  Platform: {args.platform}")
    print(f"{'='*60}\n")

    start = time.time()
    if args.url:
        logger.info(f"Processing URL: {args.url}")
        result = pipeline.process_url(args.url, config)
    elif args.batch and args.input:
        logger.info(f"Batch processing: {args.input}")
        result = pipeline.process_batch(args.input, config)
    elif args.input:
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
