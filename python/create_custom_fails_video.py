"""
Interactive Custom Fails compiler.

Allows the user to easily input direct YouTube video URLs or local file paths,
downloads and trims them, and edits them in Ryth-style with the custom fox character.
"""

from __future__ import annotations

import os
import sys
import subprocess
import tempfile
import logging
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.reaction_workflow import edit_clip_ryth_style, generate_voiceover_and_srt
from python.engines.character_overlay import CharacterOverlayEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("custom_compiler")


def download_url(url: str, temp_dir: str, idx: int) -> str | None:
    """Downloads a single video URL using yt-dlp."""
    logger.info(f"Downloading video {idx}: {url}")
    out_path = os.path.join(temp_dir, f"raw_video_{idx}.mp4")
    cmd = [
        "yt-dlp",
        "--no-check-certificates",
        "-f", "mp4",
        "-o", out_path,
        url
    ]
    try:
        subprocess.run(cmd, check=True)
        if os.path.exists(out_path):
            return out_path
    except Exception as e:
        logger.error(f"Failed to download URL {url}: {e}")
    return None


def main():
    print("====================================================")
    print("    RYTH-STYLE CUSTOM COMPILATION GENERATOR         ")
    print("====================================================")
    
    # Prompt the user for URLs (for automation or interactive shell runs)
    url_input = input("Enter direct YouTube URLs (comma-separated) or press Enter to use local samples:\n").strip()
    
    urls = [u.strip() for u in url_input.split(",") if u.strip()] if url_input else []
    
    temp_dir = tempfile.mkdtemp(prefix="interactive_comp_")
    raw_files = []
    
    if urls:
        print(f"Starting downloads for {len(urls)} URLs...")
        for i, url in enumerate(urls):
            path = download_url(url, temp_dir, i+1)
            if path:
                raw_files.append(path)
    else:
        # Fallback to local default project video if no URLs are supplied
        default_video = r"C:\Users\admin\Documents\Default Project\project\assets\videos\input.mp4"
        if os.path.exists(default_video):
            print(f"Using default local video: {default_video}")
            raw_files.append(default_video)
        else:
            print("No source files available.")
            sys.exit(1)
            
    if not raw_files:
        print("Error: No videos downloaded successfully.")
        sys.exit(1)
        
    print(f"Processing {len(raw_files)} clips...")
    
    # Trim and format clips to exactly 5.0 seconds
    formatted_clips = []
    for i, raw_path in enumerate(raw_files):
        out_clip = os.path.join(temp_dir, f"trimmed_{i+1}.mp4")
        # Split a 5-second slice (e.g. from 2.0s to 7.0s)
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", "2.0" if not urls else "0.0",
            "-i", raw_path,
            "-t", "5.0",
            "-vf", "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-c:a", "aac",
            out_clip
        ]
        try:
            subprocess.run(cmd, check=True)
            formatted_clips.append(out_clip)
        except Exception as e:
            logger.error(f"Failed to trim clip {i+1}: {e}")
            
    if not formatted_clips:
        print("Error: Trimming failed.")
        sys.exit(1)
        
    # Set dynamic ranking overlays based on number of clips
    rank_titles = [f"MOMENT SPECIAL {idx+1}" for idx in range(len(formatted_clips))]
    sfx_path = os.path.join(temp_dir, "boing_sfx.wav")
    cmd_sfx = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "sine=frequency=400:duration=0.5",
        "-af", "tremolo=f=10:d=0.8",
        "-t", "0.5",
        sfx_path
    ]
    subprocess.run(cmd_sfx, check=True)
    
    edited_clips = []
    for i, clip in enumerate(formatted_clips):
        moment_num = len(formatted_clips) - i
        out_edited = os.path.join(temp_dir, f"edited_moment_{moment_num}.mp4")
        # Call Ryth-style editing
        if edit_clip_ryth_style(clip, out_edited, moment_num, sfx_path, temp_dir):
            edited_clips.append(out_edited)
            
    # Generate Voiceover Script
    script_lines = [f"This is moment number {len(edited_clips) - idx} of our fail compilation." for idx in range(len(edited_clips))]
    audio_vo, _ = generate_voiceover_and_srt(script_lines, temp_dir)
    
    # Concat
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
    
    # Add Voiceover
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
    
    # Apply Fox Character Overlay
    final_output_path = r"C:\Users\admin\.gemini\antigravity-ide\scratch\video-edit-engine\output\custom_fails_compilation.mp4"
    os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
    
    overlay_engine = CharacterOverlayEngine()
    success = overlay_engine.overlay_avatar(
        video_path=synced_video_path,
        output_path=final_output_path,
        avatar_path=r"C:\Users\admin\.gemini\antigravity-ide\scratch\video-edit-engine\assets\logos\fox_observer.png",
        position="bottom_right",
        scale=0.3
    )
    
    if success:
        print("\n====================================================")
        print(f"Compilation Compiled Successfully!")
        print(f"Saved at: {final_output_path}")
        print("====================================================")
    else:
        print("Error compiling final video.")


if __name__ == "__main__":
    main()
