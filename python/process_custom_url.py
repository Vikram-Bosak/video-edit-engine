"""
Process Custom URL.

Downloads a specified YouTube Shorts URL, cuts it into 3 segments,
and runs the Ryth-style reaction editing workflow on it.
"""

from __future__ import annotations

import os
import sys
import subprocess
import tempfile
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.reaction_workflow import edit_clip_ryth_style, generate_voiceover_and_srt, get_script_for_topic
from python.engines.character_overlay import CharacterOverlayEngine
from python.utils.google_drive_uploader import GoogleDriveUploader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("custom_url_processor")


def download_youtube_video(url: str, download_dir: str) -> str:
    """Downloads a YouTube video using yt-dlp."""
    logger.info(f"Downloading video from: {url}")
    os.makedirs(download_dir, exist_ok=True)
    out_path_tpl = os.path.join(download_dir, "downloaded_video.mp4")
    
    cmd = [
        "yt-dlp",
        "--no-check-certificates",
        "-f", "mp4",
        "-o", out_path_tpl,
        url
    ]
    subprocess.run(cmd, check=True)
    return out_path_tpl


def split_video_into_clips(input_video: str, output_dir: str) -> List[str]:
    """Splits input video into three 5-second segments."""
    logger.info("Splitting source video into clips...")
    clips = []
    
    # Clip 1: 0s to 5s
    # Clip 2: 5s to 10s
    # Clip 3: 10s to 15s
    for i in range(3):
        start_time = i * 5
        clip_path = os.path.join(output_dir, f"split_clip_{i+1}.mp4")
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", str(start_time),
            "-i", input_video,
            "-t", "5",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-c:a", "aac",
            clip_path
        ]
        subprocess.run(cmd, check=True)
        clips.append(clip_path)
        
    return clips


def main():
    url = "https://www.youtube.com/shorts/GwcoSst5lL0?si=vVw0fzXwAzxnimm7"
    output_path = r"C:\Users\admin\.gemini\antigravity-ide\scratch\video-edit-engine\output\youtube_short_reaction_final.mp4"
    temp_dir = tempfile.mkdtemp(prefix="custom_reaction_")
    
    logger.info("=== STARTING YOUTUBE REACTION GENERATOR ===")
    
    # 1. Download the real YouTube Shorts video
    downloaded_video = download_youtube_video(url, temp_dir)
    
    # 2. Split it into segments
    clips = split_video_into_clips(downloaded_video, temp_dir)
    
    # 3. Create sound effect
    # We will generate a synthetic sound effect for fail moment sync
    sfx_path = os.path.join(temp_dir, "boing_sfx.wav")
    cmd_sfx = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "sine=frequency=400:duration=0.5",
        "-af", "tremolo=f=10:d=0.8",
        "-t", "0.5",
        sfx_path
    ]
    subprocess.run(cmd_sfx, check=True)
    
    # 4. Apply Ryth-style editing to each segment
    edited_clips = []
    for i, clip in enumerate(clips):
        moment_num = 3 - i
        out_clip_path = os.path.join(temp_dir, f"edited_moment_{moment_num}.mp4")
        if edit_clip_ryth_style(clip, out_clip_path, moment_num, sfx_path, temp_dir):
            edited_clips.append(out_clip_path)
            
    # 5. Generate commentary and voiceover
    script_lines = get_script_for_topic("basketball")
    audio_vo, srt_subs = generate_voiceover_and_srt(script_lines, temp_dir)
    
    # 6. Concat the edited clips together
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
    
    # 7. Add Voiceover audio track
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
    
    # 8. Apply Character Reaction Overlay
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    overlay_engine = CharacterOverlayEngine()
    success = overlay_engine.overlay_avatar(
        video_path=synced_video_path,
        output_path=output_path,
        position="bottom_right",
        scale=0.3
    )
    
    if success:
        logger.info(f"Successfully compiled custom YouTube reaction video: {output_path}")
        # Upload
        uploader = GoogleDriveUploader()
        uploader.upload_file(output_path)
    else:
        logger.error("Failed to compile reaction video.")


if __name__ == "__main__":
    main()
