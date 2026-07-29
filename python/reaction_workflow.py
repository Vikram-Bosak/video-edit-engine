"""
Reaction Workflow Orchestrator.

Main entry point for generating automated reaction compilation videos
similar to Ryth-style YouTube Shorts.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import tempfile
import time
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.engines.character_overlay import CharacterOverlayEngine
from python.utils.google_drive_uploader import GoogleDriveUploader

logger = logging.getLogger("reaction_workflow")


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt, stream=sys.stderr)


def search_and_download_clips(query: str, download_dir: str, limit: int = 3) -> List[str]:
    """Simulates finding and downloading trending funny/news clips."""
    logger.info(f"Searching for clips with query: '{query}'")
    
    # We will look for existing sample videos in assets/videos first
    sample_dir = r"assets/videos"
    downloaded_paths = []

    if os.path.exists(sample_dir):
        files = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if f.endswith((".mp4", ".mov", ".avi"))]
        if files:
            logger.info(f"Found {len(files)} local videos to use as reaction source clips.")
            return files[:limit]

    # Try to download using yt-dlp search first
    os.makedirs(download_dir, exist_ok=True)
    out_template = os.path.join(download_dir, "yt_raw_%(autonumber)d.mp4")
    cmd_yt = [
        "yt-dlp",
        "--no-check-certificates",
        "-f", "mp4",
        "-o", out_template,
        "--max-downloads", str(limit),
        f"ytsearch{limit}:{query}"
    ]
    try:
        logger.info(f"Running yt-dlp search download for: {query}")
        subprocess.run(cmd_yt, check=True)
        downloaded_paths = [os.path.join(download_dir, f) for f in os.listdir(download_dir) if f.startswith("yt_raw_") and f.endswith(".mp4")]
    except Exception as e:
        logger.warning(f"yt-dlp search download failed: {e}")

    # Fallback to Nitter RSS Twitter downloader
    if len(downloaded_paths) < limit:
        logger.info("Trying Nitter RSS Twitter search fallback...")
        try:
            from python.nitter_video_downloader import fetch_nitter_videos
            nitter_paths = fetch_nitter_videos(f"{query} video", download_dir, limit=limit)
            downloaded_paths.extend(nitter_paths)
            downloaded_paths = list(set(downloaded_paths))
        except Exception as ne:
            logger.warning(f"Nitter RSS search fallback failed: {ne}")

    if len(downloaded_paths) >= limit:
        # Trim clips to exactly 5 seconds vertical format
        trimmed_paths = []
        for idx, raw_path in enumerate(downloaded_paths[:limit]):
            trimmed_path = os.path.join(download_dir, f"clip_{idx+1}.mp4")
            cmd_trim = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-ss", "0.0",
                "-i", raw_path,
                "-t", "5.0",
                "-vf", "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920",
                "-c:v", "libx264", "-preset", "ultrafast",
                "-c:a", "aac",
                trimmed_path
            ]
            try:
                subprocess.run(cmd_trim, check=True)
                trimmed_paths.append(trimmed_path)
            except Exception as tr_err:
                logger.error(f"Failed to format raw clip: {tr_err}")
        if len(trimmed_paths) >= limit:
            return trimmed_paths

    # Hard fallback to simulated color bar testsrc
    logger.info("Falling back to cloning simulated clips...")
    downloaded_paths = []
    for i in range(limit):
        clip_path = os.path.join(download_dir, f"clip_{i+1}.mp4")
        cmd_sim = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=duration=5:size=1080x1920:rate=30",
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-t", "5",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-c:a", "aac",
            clip_path
        ]
        try:
            subprocess.run(cmd_sim, check=True)
            downloaded_paths.append(clip_path)
        except Exception as e:
            logger.error(f"Failed to create simulated clip: {e}")
            
    return downloaded_paths


def create_meme_sfx(temp_dir: str) -> str:
    """Generates a synthetic meme boing sound effect."""
    sfx_path = os.path.join(temp_dir, "boing_sfx.wav")
    import subprocess
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "sine=frequency=400:duration=0.5",
        "-af", "tremolo=f=10:d=0.8",
        "-t", "0.5",
        sfx_path
    ]
    subprocess.run(cmd, check=True)
    return sfx_path


def add_ryth_elements_with_opencv(
    input_path: str,
    output_path: str,
    moment_num: int,
    rank_titles: List[str]
):
    """
    Renders advanced Ryth-style visual overlays:
    1. Dynamic Zoom-in (Slow Ken Burns effect).
    2. Left-side Persistent Ranking Overlay.
    3. Header Title "MOMENT #X".
    4. Saturation Drop (Black & White conversion) during the joke punchline (from 3.0s to 5.0s).
    """
    import cv2
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 150
    punchline_start_frame = int(3.0 * fps)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. Apply Dynamic Zoom-in (up to 1.15x scale at the end of the clip)
        zoom_factor = 1.0 + (frame_idx / total_frames) * 0.15
        zoom_factor = min(zoom_factor, 1.15)
        
        new_w = int(width / zoom_factor)
        new_h = int(height / zoom_factor)
        x_offset = (width - new_w) // 2
        y_offset = (height - new_h) // 2
        cropped = frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w]
        frame = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
            
        # 2. Apply Saturation Drop (Black & White / Grayscale conversion) during punchline (3.0s - 5.0s)
        if frame_idx >= punchline_start_frame:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.merge([gray, gray, gray])
            
        # 3. Draw Moment Header
        header_text = f"MOMENT #{moment_num}"
        text_size = cv2.getTextSize(header_text, font, 2.0, 5)[0]
        text_x = (width - text_size[0]) // 2
        cv2.putText(frame, header_text, (text_x, 120), font, 2.0, (0, 0, 0), 9, cv2.LINE_AA)
        cv2.putText(frame, header_text, (text_x, 120), font, 2.0, (0, 255, 255), 5, cv2.LINE_AA)
        
        # 4. Draw Left-side Persistent Ranking List Overlay
        for idx, title in enumerate(rank_titles):
            rank_y = 350 + idx * 80
            rank_text = f"{idx + 1}. {title}"
            is_active = (moment_num == (3 - idx))
            
            text_color = (0, 255, 255) if is_active else (220, 220, 220)
            font_scale = 1.2 if is_active else 0.9
            thickness = 3 if is_active else 2
            
            cv2.putText(frame, rank_text, (50, rank_y), font, font_scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
            cv2.putText(frame, rank_text, (50, rank_y), font, font_scale, text_color, thickness, cv2.LINE_AA)
            
        out.write(frame)
        frame_idx += 1
        
    cap.release()
    out.release()


def edit_clip_ryth_style(
    input_path: str,
    output_path: str,
    moment_num: int,
    sfx_path: str,
    temp_dir: str
) -> bool:
    """
    Applies Ryth-style editing to a single video clip:
    1. Zoom-in dynamic pan (OpenCV).
    2. Numbering title card and persistent left ranking overlay using OpenCV.
    3. Black & White conversion on punchlines (OpenCV).
    4. Meme sound effect mixed at the fail moment.
    """
    logger.info(f"Applying Ryth-style edits to {input_path} (Moment #{moment_num})")
    
    # Step 1: Run SFX audio mixing with FFmpeg (no video filter, extremely fast!)
    sfx_mixed_temp = os.path.join(temp_dir, f"sfx_mixed_{moment_num}.mp4")
    import subprocess
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", input_path,
        "-i", sfx_path,
        "-filter_complex",
        f"[1:a]adelay=3000|3000[delayed_sfx];"
        f"[0:a][delayed_sfx]amix=inputs=2:duration=first[outa]",
        "-map", "0:v",
        "-map", "[outa]",
        "-c:v", "copy",  # Direct copy, no transcoding!
        "-c:a", "aac",
        sfx_mixed_temp
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to mix SFX: {e}")
        return False

    # Step 2: Apply Zoom, Draw text overlays & Grayscale dynamically with OpenCV
    rank_titles = ["KABOOM", "SPIDEY", "LEBRON"]  # Map to index 3, 2, 1
    try:
        temp_opencv_out = os.path.join(temp_dir, f"opencv_out_{moment_num}.mp4")
        add_ryth_elements_with_opencv(sfx_mixed_temp, temp_opencv_out, moment_num, rank_titles)
        
        # Merge the silent OpenCV output video with the original audio from sfx_mixed_temp,
        # encoding to highly compatible H.264 format
        cmd_merge = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", temp_opencv_out,
            "-i", sfx_mixed_temp,
            "-map", "0:v",
            "-map", "1:a?",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            output_path
        ]
        subprocess.run(cmd_merge, check=True)
        return True
    except Exception as e:
        logger.error(f"Failed to apply OpenCV Ryth effects or transcode: {e}")
        return False


def generate_voiceover_and_srt(script_lines: List[str], temp_dir: str) -> tuple[str, str]:
    """Generates audio voiceover file and matching subtitle SRT file."""
    logger.info("Generating AI voiceover audio and subtitles...")
    audio_path = os.path.join(temp_dir, "voiceover.wav")
    srt_path = os.path.join(temp_dir, "subtitles.srt")
    
    try:
        from gtts import gTTS
        tts_text = " . ".join(script_lines)
        tts = gTTS(text=tts_text, lang='en')
        tts.save(audio_path)
    except ImportError:
        logger.warning("gTTS not installed. Creating placeholder audio file.")
        import subprocess
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "sine=frequency=1000:duration=15",
            "-t", "15",
            "-c:a", "pcm_s16le",
            audio_path
        ]
        subprocess.run(cmd, check=True)

    with open(srt_path, "w") as f:
        for i, line in enumerate(script_lines):
            start_sec = i * 5
            end_sec = start_sec + 4.5
            start_str = f"00:00:{int(start_sec):02d},000"
            end_str = f"00:00:{int(end_sec):02d},000"
            f.write(f"{i+1}\n")
            f.write(f"{start_str} --> {end_str}\n")
            f.write(f"{line}\n\n")
            
    return audio_path, srt_path


def get_script_for_topic(query: str) -> List[str]:
    q = query.lower()
    if "swimming" in q or "pool" in q:
        return [
            "Moment Number Three: This guy thought he was doing a majestic dive, but it ended up as a massive belly flop.",
            "Moment Number Two: When you try to impress everyone at the pool but slip on the edge.",
            "Moment Number One: That moment when the water slide decides to launch you into another dimension."
        ]
    elif "football" in q or "soccer" in q:
        return [
            "Moment Number Three: Bro tried to perform a bicycle kick but only kicked the air and his own dignity.",
            "Moment Number Two: The goalkeeper was so busy celebrating that the ball rolled right past him.",
            "Moment Number One: Bro celebrated a goal that was actually ruled offside. Pure embarrassment."
        ]
    elif "gym" in q or "workout" in q:
        return [
            "Moment Number Three: Trying to show off on the treadmill never ends well. Watch him fly.",
            "Moment Number Two: The weights were definitely winning this round. Respect the gravity, bro.",
            "Moment Number One: When you try to lift 300 pounds but your knees decide to leave the chat."
        ]
    elif "animal" in q or "pet" in q or "dog" in q or "cat" in q:
        return [
            "Moment Number Three: This cat thought it could make the jump, but physics had other plans.",
            "Moment Number Two: The dog stole the show, and by show, I mean the entire family's barbecue lunch.",
            "Moment Number One: This parrot is literally laughing at the owner. Smartest bird alive."
        ]
    elif "basketball" in q:
        return [
            "Moment Number Three: Bro thought he was LeBron, but the rim said absolutely not.",
            "Moment Number Two: Watch this guy drop the ball and his self-esteem at the same time.",
            "Moment Number One: Absolute legendary fail, my man playing in Yeezys."
        ]
    else:
        return [
            "Moment Number Three: That moment when self-confidence meets absolute reality.",
            "Moment Number Two: The execution was poor, but the entertainment value is absolute gold.",
            "Moment Number One: A certified classic fail that will be remembered for generations."
        ]


def main():
    parser = argparse.ArgumentParser(description="AI Reaction Video Workflow Orchestrator")
    parser.add_argument("--query", "-q", default="funny basketball fails", help="Search query for trending clips")
    parser.add_argument("--output", "-o", default="output/final_reaction.mp4", help="Output video file path")
    parser.add_argument("--drive-folder", default=None, help="Google Drive folder ID to upload to")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    start_time = time.time()
    logger.info("=== STARTING ADVANCED REACTION WORKFLOW GENERATION ===")
    
    temp_dir = tempfile.mkdtemp(prefix="reaction_")
    
    # Step 1 & 2: Search and download clips
    clips = search_and_download_clips(args.query, temp_dir, limit=3)
    if not clips:
        logger.error("No source clips available. Exiting.")
        sys.exit(1)
        
    sfx_path = create_meme_sfx(temp_dir)
    
    # Step 3: Sequence and Apply Ryth-style edits individually
    edited_clips = []
    for i, clip in enumerate(clips):
        # Moment 3, 2, 1 descending
        moment_num = 3 - i
        out_clip_path = os.path.join(temp_dir, f"edited_moment_{moment_num}.mp4")
        if edit_clip_ryth_style(clip, out_clip_path, moment_num, sfx_path, temp_dir):
            edited_clips.append(out_clip_path)
            
    if len(edited_clips) < 3:
        logger.error("Failed to edit all clips. Exiting.")
        sys.exit(1)
        
    # Step 4: Generate script & voiceover
    script_lines = get_script_for_topic(args.query)
    audio_vo, srt_subs = generate_voiceover_and_srt(script_lines, temp_dir)
    
    # Step 5: Concat the edited clips together
    concat_list_path = os.path.join(temp_dir, "clips.txt")
    with open(concat_list_path, "w") as f:
        for clip in edited_clips:
            f.write(f"file '{clip}'\n")
            
    raw_concat_path = os.path.join(temp_dir, "raw_concat.mp4")
    import subprocess
    cmd_concat = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", concat_list_path,
        "-c:v", "libx264", "-preset", "ultrafast",
        "-an", raw_concat_path
    ]
    subprocess.run(cmd_concat, check=True)
    
    # Step 6: Add Voiceover and Subtitles
    synced_video_path = os.path.join(temp_dir, "synced_video.mp4")
    cmd_audio = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", raw_concat_path,
        "-i", audio_vo,
        "-c:v", "copy",
        "-c:a", "aac", "-map", "0:v", "-map", "1:a",
        "-shortest", synced_video_path
    ]
    subprocess.run(cmd_audio, check=True)
    
    # Step 7: Apply Character Reaction Overlay
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    overlay_engine = CharacterOverlayEngine()
    success = overlay_engine.overlay_avatar(
        video_path=synced_video_path,
        output_path=args.output,
        position="bottom_right",
        scale=0.3
    )
    
    if not success:
        logger.error("Failed to apply character overlay. Exiting.")
        sys.exit(1)
        
    logger.info(f"Video compiled successfully: {args.output}")
    
    # Step 8: Upload to Google Drive
    uploader = GoogleDriveUploader()
    drive_link = uploader.upload_file(args.output, folder_id=args.drive_folder)
    
    elapsed = time.time() - start_time
    logger.info(f"=== WORKFLOW COMPLETE IN {elapsed:.2f}s ===")
    if drive_link:
        logger.info(f"Drive Sharing URL: {drive_link}")
        
    sys.exit(0)


if __name__ == "__main__":
    main()
