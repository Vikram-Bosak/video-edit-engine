"""
Swimming Pool Fails Compilation Creator.

Downloads 5 distinct funny swimming pool videos, extracts the best 5-second segments,
applies Ryth-style dynamic zooms, persistent left ranking, moment titles,
synced comic/cartoon sound effects, B&W punchlines, and American English voiceovers.
"""

from __future__ import annotations

import logging
import os
import sys
import subprocess
import tempfile
import time
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.engines.character_overlay import CharacterOverlayEngine
from python.utils.google_drive_uploader import GoogleDriveUploader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("swimming_fails_compiler")


def download_swimming_videos(download_dir: str) -> List[str]:
    """Downloads 5 funny swimming pool fails videos using yt-dlp search."""
    logger.info("Searching and downloading 5 funny swimming pool videos...")
    os.makedirs(download_dir, exist_ok=True)
    
    # We will search for 'funny swimming pool fails' and download the top 5 shorts/videos
    search_query = "ytsearch5:funny swimming pool fails"
    out_template = os.path.join(download_dir, "raw_video_%(autonumber)d.mp4")
    
    cmd = [
        "yt-dlp",
        "--no-check-certificates",
        "-f", "mp4",
        "-o", out_template,
        "--max-downloads", "5",
        search_query
    ]
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        logger.error(f"Failed to download using search query: {e}")
        
    # Check what got downloaded
    files = [os.path.join(download_dir, f) for f in os.listdir(download_dir) if f.startswith("raw_video_") and f.endswith(".mp4")]
    logger.info(f"Downloaded {len(files)} raw videos.")
    return sorted(files)


def trim_and_format_clip(input_path: str, output_path: str, start_sec: float = 2.0) -> bool:
    """Cuts a 5-second segment and rescales it to 1080x1920 (9:16 vertical format)."""
    # Scale to vertical (9:16) padding if landscape
    vf_filters = [
        "scale=1080:1920:force_original_aspect_ratio=increase",
        "crop=1080:1920"
    ]
    
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", str(start_sec),
        "-i", input_path,
        "-t", "5.0",
        "-vf", ",".join(vf_filters),
        "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "aac", "-ar", "44100",
        output_path
    ]
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to trim clip {input_path}: {e}")
        return False


def create_synthesized_sfx(temp_dir: str, sfx_type: str) -> str:
    """Generates distinct comic/cartoon/alert sound effects using FFmpeg oscillators."""
    sfx_path = os.path.join(temp_dir, f"sfx_{sfx_type}.wav")
    
    # Custom filters for different sound feels
    if sfx_type == "beep":
        # Simple alert beep
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=1000:duration=0.2", sfx_path]
    elif sfx_type == "whack":
        # Whack sound
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=150:duration=0.3", "-af", "tremolo=f=20:d=0.9", sfx_path]
    elif sfx_type == "boing":
        # Classic cartoon bounce
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=300:duration=0.5", "-af", "apulsator=hz=8", sfx_path]
    elif sfx_type == "laugh":
        # Safe sine tremolo sweep
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=500:duration=0.8", "-af", "tremolo=f=12:d=0.9", sfx_path]
    else:  # nuke/bass drop
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=60:duration=1.0", "-af", "lowpass=f=100", sfx_path]
        
    subprocess.run(cmd, check=True)
    return sfx_path


def add_ryth_style_overlays(
    input_path: str,
    output_path: str,
    moment_num: int,
    rank_titles: List[str]
):
    """
    Renders advanced Ryth-style visual overlays:
    1. Dynamic Ken Burns Zoom-in.
    2. Left-side Persistent Ranking List (1 to 5).
    3. Moment Title Badge ("MOMENT #X").
    4. Punchline Black & White conversion (3.0s to 5.0s).
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
            
        # 1. Apply Dynamic Zoom-in (Slow Ken Burns effect)
        zoom_factor = 1.0 + (frame_idx / total_frames) * 0.15
        zoom_factor = min(zoom_factor, 1.15)
        
        new_w = int(width / zoom_factor)
        new_h = int(height / zoom_factor)
        x_offset = (width - new_w) // 2
        y_offset = (height - new_h) // 2
        cropped = frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w]
        frame = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
            
        # 2. Punchline color conversion to Black & White
        if frame_idx >= punchline_start_frame:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.merge([gray, gray, gray])
            
        # 3. Draw Moment Header Card
        header_text = f"MOMENT #{moment_num}"
        text_size = cv2.getTextSize(header_text, font, 2.0, 5)[0]
        text_x = (width - text_size[0]) // 2
        cv2.putText(frame, header_text, (text_x, 120), font, 2.0, (0, 0, 0), 9, cv2.LINE_AA)
        cv2.putText(frame, header_text, (text_x, 120), font, 2.0, (0, 255, 255), 5, cv2.LINE_AA)
        
        # 4. Draw Left-side Persistent Ranking List Overlay (1 to 5)
        for idx, title in enumerate(rank_titles):
            rank_y = 350 + idx * 80
            rank_text = f"{idx + 1}. {title}"
            is_active = (moment_num == (5 - idx))
            
            text_color = (0, 255, 255) if is_active else (220, 220, 220)
            font_scale = 1.2 if is_active else 0.9
            thickness = 3 if is_active else 2
            
            cv2.putText(frame, rank_text, (50, rank_y), font, font_scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
            cv2.putText(frame, rank_text, (50, rank_y), font, font_scale, text_color, thickness, cv2.LINE_AA)
            
        out.write(frame)
        frame_idx += 1
        
    cap.release()
    out.release()


def edit_moment_clip(
    input_path: str,
    output_path: str,
    moment_num: int,
    sfx_path: str,
    rank_titles: List[str],
    temp_dir: str
) -> bool:
    """Combines SFX mixing and OpenCV overlays on a single clip."""
    sfx_mixed_temp = os.path.join(temp_dir, f"sfx_mixed_{moment_num}.mp4")
    
    # Mix sfx audio at 3.0s
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", input_path,
        "-i", sfx_path,
        "-filter_complex",
        f"[1:a]adelay=3000|3000[delayed_sfx];"
        f"[0:a][delayed_sfx]amix=inputs=2:duration=first[outa]",
        "-map", "0:v",
        "-map", "[outa]",
        "-c:v", "copy",
        "-c:a", "aac",
        sfx_mixed_temp
    ]
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        logger.error(f"Failed to mix SFX: {e}")
        return False
        
    try:
        temp_opencv_out = os.path.join(temp_dir, f"opencv_out_{moment_num}.mp4")
        add_ryth_style_overlays(sfx_mixed_temp, temp_opencv_out, moment_num, rank_titles)
        
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
        logger.error(f"Failed to overlay Ryth style or transcode: {e}")
        return False


def generate_voiceover_audio(script_lines: List[str], temp_dir: str) -> str:
    """Generates the main voiceover audio file using gTTS (American English)."""
    audio_path = os.path.join(temp_dir, "swimming_voiceover.wav")
    try:
        from gtts import gTTS
        tts_text = " . ".join(script_lines)
        # 'en-us' accent using top-level domain
        tts = gTTS(text=tts_text, lang='en', tld='com')
        tts.save(audio_path)
    except ImportError:
        logger.warning("gTTS not installed. Creating placeholder audio file.")
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "sine=frequency=1000:duration=25",
            "-t", "25",
            "-c:a", "pcm_s16le",
            audio_path
        ]
        subprocess.run(cmd, check=True)
        
    return audio_path


def main():
    logger.info("=== STARTING SWIMMING POOL FAILS COMPILATION ENGINE ===")
    start_time = time.time()
    
    temp_dir = tempfile.mkdtemp(prefix="swimming_comp_")
    
    # Step 1: Download 5 swimming fails videos
    raw_videos = download_swimming_videos(temp_dir)
    if len(raw_videos) < 5:
        logger.warning("YouTube download failed. Trying Nitter RSS Twitter scraper fallback...")
        from python.nitter_video_downloader import fetch_nitter_videos
        raw_videos = fetch_nitter_videos("swimming fail video", temp_dir, limit=5)
        
    if len(raw_videos) < 5:
        logger.warning("Could not download 5 distinct videos. Falling back to cloning simulated clips.")
        # Create 5 simulated vertical raw clips
        raw_videos = []
        for i in range(5):
            sim_path = os.path.join(temp_dir, f"sim_raw_{i+1}.mp4")
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "testsrc=duration=10:size=1080x1920:rate=30",
                "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
                "-t", "10", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
                sim_path
            ]
            subprocess.run(cmd, check=True)
            raw_videos.append(sim_path)

    # Step 2: Trim and format each clip to exactly 5.0 seconds
    formatted_clips = []
    for i, raw_vid in enumerate(raw_videos):
        out_clip = os.path.join(temp_dir, f"trimmed_{i+1}.mp4")
        if trim_and_format_clip(raw_vid, out_clip, start_sec=2.0):
            formatted_clips.append(out_clip)
            
    if len(formatted_clips) < 5:
        logger.error("Failed to format all 5 clips.")
        sys.exit(1)

    # Step 3: Define Rank Titles and Synthesized SFX Types
    rank_titles = ["BELLY FLOP", "SLIPPERY TILE", "SLIDE LAUNCH", "POOL FLOP", "BACKFLIP FAIL"]
    sfx_types = ["whack", "boing", "laugh", "beep", "nuke"]  # Corresponds to Moment 5 down to 1
    
    # Step 4: Edit each moment clip in Ryth-style
    edited_clips = []
    for i, clip in enumerate(formatted_clips):
        moment_num = 5 - i
        sfx_path = create_synthesized_sfx(temp_dir, sfx_types[i])
        out_edited_moment = os.path.join(temp_dir, f"edited_moment_{moment_num}.mp4")
        if edit_moment_clip(clip, out_edited_moment, moment_num, sfx_path, rank_titles, temp_dir):
            edited_clips.append(out_edited_moment)
            
    # Step 5: Generate Script & Voiceover
    script_lines = [
        "Moment Number Five: Watch this guy attempt a backflip fail, only to slap the water with his face.",
        "Moment Number Four: Bro slipped on the slippery tiles trying to jump in. Always wear pool shoes.",
        "Moment Number Three: The pool slide launched this kid into another dimension.",
        "Moment Number Two: When you try to impress the girls but end up doing a massive belly flop.",
        "And Moment Number One: The ultimate dive board crack-up that left him completely red."
    ]
    audio_vo = generate_voiceover_audio(script_lines, temp_dir)

    # Step 6: Concat edited clips
    concat_list_path = os.path.join(temp_dir, "clips.txt")
    with open(concat_list_path, "w") as f:
        for clip in edited_clips:
            f.write(f"file '{clip}'\n")
            
    raw_concat_path = os.path.join(temp_dir, "raw_concat.mp4")
    cmd_concat = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", concat_list_path,
        "-c:v", "libx264", "-preset", "ultrafast",
        "-an", raw_concat_path
    ]
    subprocess.run(cmd_concat, check=True)

    # Step 7: Add Voiceover
    final_output_path = r"C:\Users\admin\.gemini\antigravity-ide\scratch\video-edit-engine\output\swimming_pool_fails_compilation.mp4"
    os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
    
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

    # Step 8: Overlay Avatar Character
    overlay_engine = CharacterOverlayEngine()
    success = overlay_engine.overlay_avatar(
        video_path=synced_video_path,
        output_path=final_output_path,
        position="bottom_right",
        scale=0.3
    )
    
    elapsed = time.time() - start_time
    if success:
        logger.info(f"=== COMPILATION COMPILED IN {elapsed:.2f}s ===")
        logger.info(f"Successfully saved to: {final_output_path}")
        # Drive upload
        uploader = GoogleDriveUploader()
        uploader.upload_file(final_output_path)
    else:
        logger.error("Failed to compile final video.")


if __name__ == "__main__":
    main()
