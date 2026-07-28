# AI Video Edit Engine

A fully automated, professional video editing engine that runs on GitHub Actions - producing CapCut-quality social media videos entirely through code.

## Features

- **Smart Scene Detection** - AI-powered scene change detection, highlighting, and intelligent splitting
- **AI Smart Crop** - Auto-detect faces, humans, vehicles, animals, and keep subjects centered
- **Dynamic Zoom** - Ken Burns, zoom in/out, pan, camera shake, cinematic push effects
- **Transition Engine** - 15+ transitions: fade, flash, whip, blur, slide, zoom, glitch, light leak, film burn
- **Text Animation** - 14 animated text styles: pop, bounce, fade, scale, typewriter, elastic, neon glow
- **Graphics Overlay** - Logo, watermark, progress bar, lower thirds, subscribe buttons, intro/outro
- **Audio Engine** - Background music, voiceover, ducking, normalization, beat detection, silence removal
- **Subtitle Engine** - Word-highlighted captions, animated subtitles, ASS/SRT generation
- **Color Grading** - Cinematic presets, LUT support, auto color correction, temperature/tint
- **Motion Effects** - Ken Burns, parallax, speed ramp, stabilization, dynamic crop
- **Timeline Engine** - Multi-layer compositing with video, audio, overlay, and text tracks
- **Export Engine** - H264, H265, VP9, AV1, GIF with platform-specific presets
- **Batch Processing** - Process 100+ videos with queue, retry, and resume support
- **AI Orchestrator** - Automatic editing decisions based on video content analysis

## Architecture

```
                    ┌──────────────────┐
                    │   Input Source    │
                    │  (File/URL/Batch)│
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Video Analyzer   │
                    │  (Metadata/Quality)│
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  AI Orchestrator  │
                    │ (Decision Engine) │
                    └────────┬─────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
  ┌───────▼──────┐  ┌───────▼──────┐  ┌───────▼──────┐
  │ Scene Detect │  │  Smart Crop  │  │Color Grading │
  └───────┬──────┘  └───────┬──────┘  └───────┬──────┘
          │                  │                  │
  ┌───────▼──────┐  ┌───────▼──────┐  ┌───────▼──────┐
  │Transitions   │  │Text Animation│  │Motion Effects│
  └───────┬──────┘  └───────┬──────┘  └───────┬──────┘
          │                  │                  │
  ┌───────▼──────┐  ┌───────▼──────┐  ┌───────▼──────┐
  │Audio Process │  │  Subtitles   │  │   Graphics   │
  └───────┬──────┘  └───────┬──────┘  └───────┬──────┘
          │                  │                  │
          └──────────────────┼──────────────────┘
                             │
                    ┌────────▼─────────┐
                    │  Timeline Engine  │
                    │   (Compositing)   │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │   Export Engine   │
                    │ (Platform Export) │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Report Generator │
                    └──────────────────┘
```

## Prerequisites

- Python 3.10+
- FFmpeg 5.0+
- ImageMagick (for advanced text)
- libass (for ASS subtitles)
- Ubuntu 22.04+ (for GitHub Actions)

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/video-edit-engine.git
cd video-edit-engine

# Install dependencies
chmod +x install_dependencies.sh
./install_dependencies.sh

# Process a single video
python -m python.main --input video.mp4 --platform tiktok

# Process a directory of videos
python -m python.main --input assets/videos/ --batch --platform youtube_short

# Process from URL
python -m python.main --url "https://youtube.com/watch?v=xxx" --platform instagram_reel
```

## Installation

### System Dependencies (Ubuntu)
```bash
sudo apt-get update
sudo apt-get install -y ffmpeg imagemagick libass-dev libgl1-mesa-dev
```

### Python Dependencies
```bash
pip install -r requirements.txt
```

### GitHub Actions
The workflow installs everything automatically. Just push to `main` or trigger manually.

## Configuration

### Platform Presets

| Platform | Resolution | Max Duration | Bitrate | Codec |
|----------|-----------|-------------|---------|-------|
| TikTok | 1080x1920 | 180s | 8Mbps | H264 |
| YouTube Short | 1080x1920 | 60s | 10Mbps | H264 |
| Facebook Reel | 1080x1920 | 90s | 8Mbps | H264 |
| Instagram Reel | 1080x1920 | 90s | 8Mbps | H264 |
| Twitter Video | 1280x720 | 140s | 5Mbps | H264 |

### Templates

Pre-built templates in `assets/templates/`:
- `tiktok.json` - High energy, neon text, glitch transitions
- `youtube_short.json` - Subscribe CTA, end screen, cinematic
- `facebook_reel.json` - Vibrant, word-highlight subtitles
- `instagram_reel.json` - Warm vintage, progress bar
- `cinematic.json` - Film burn transitions, teal-orange grading
- `fast_paced.json` - Beat-synced, camera shake, fast cuts

### Custom Configuration
```bash
python -m python.main --input video.mp4 --config config.json
```

## CLI Usage

```
usage: main.py [-h] [--input INPUT] [--output OUTPUT]
               [--platform {tiktok,youtube_short,facebook_reel,instagram_reel,twitter_video}]
               [--config CONFIG] [--template TEMPLATE] [--batch] [--url URL]
               [--verbose] [--report-only]

Options:
  --input, -i       Input video file or directory
  --output, -o      Output directory (default: output)
  --platform, -p    Target platform
  --config, -c      Configuration file path
  --template, -t    Template file path
  --batch, -b       Batch process directory
  --url, -u         Video URL to download and process
  --verbose, -v     Verbose output
  --report-only     Generate report only
```

## GitHub Actions Setup

### Automatic (Push)
```yaml
# Videos in assets/videos/ are auto-processed on push to main
```

### Manual Trigger
1. Go to Actions > AI Video Edit Engine
2. Click "Run workflow"
3. Enter video URL (optional)
4. Select platform
5. Click "Run"

### With Custom Config
```yaml
env:
  CONFIG_PATH: assets/templates/cinematic.json
```

## API Reference

### VideoPipeline
```python
from python.processors.pipeline import VideoPipeline, PipelineConfig

pipeline = VideoPipeline()
config = PipelineConfig(platform="tiktok")
result = pipeline.process_video("input.mp4", config)
```

### AI Orchestrator
```python
from python.ai.orchestrator import AIOrchestrator, ExportPlatform

orch = AIOrchestrator()
plan = orch.generate_full_edit("input.mp4", ExportPlatform.TIKTOK)
result = orch.execute_edit(plan)
```

### Batch Processing
```python
from python.processors.batch_processor import BatchProcessor, BatchConfig

bp = BatchProcessor()
config = BatchConfig(max_workers=4, retry_count=3)
result = bp.process_batch(video_paths, "output/", "tiktok", config)
```

## Module Reference

| Module | Description |
|--------|-------------|
| `python.engines.scene_detection` | Scene change detection, highlighting, splitting |
| `python.engines.smart_crop` | AI-powered subject detection and cropping |
| `python.engines.transitions` | 15+ video transition effects |
| `python.engines.text_animation` | Animated text with ASS generation |
| `python.engines.graphics_overlay` | Logo, watermark, lower thirds, progress bars |
| `python.engines.audio_processing` | Music mixing, ducking, normalization, beats |
| `python.engines.subtitle_engine` | Word-highlighted captions, ASS/SRT |
| `python.engines.color_grading` | Cinematic looks, LUT, auto correction |
| `python.engines.motion_effects` | Zoom, Ken Burns, shake, parallax |
| `python.engines.timeline` | Multi-layer timeline compositing |
| `python.engines.export_engine` | Multi-format export with platform presets |
| `python.processors.video_analyzer` | Video metadata and quality analysis |
| `python.processors.batch_processor` | Concurrent batch processing |
| `python.processors.reporter` | JSON/HTML report generation |
| `python.processors.pipeline` | Main pipeline orchestrator |
| `python.ai.orchestrator` | AI-driven editing decisions |
| `python.core.config` | Configuration management |
| `python.utils.logger` | Structured logging |
| `python.utils.helpers` | Utility functions |

## Troubleshooting

### FFmpeg not found
```bash
sudo apt-get install ffmpeg
ffmpeg -version
```

### OpenCV errors
```bash
pip install opencv-python-headless
# NOT opencv-python (requires GUI)
```

### Memory issues in batch mode
```json
{
  "batch": {
    "max_workers": 1,
    "memory_limit_mb": 2048
  }
}
```

### Slow processing
- Reduce `max_workers` if CPU-bound
- Use `preset: ultrafast` in export config
- Disable motion effects for faster processing

## License

MIT License
