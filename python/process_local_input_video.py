"""
Process Local Input Video.

Uses the local pre-existing input.mp4 video file to split, compile,
and edit into a full Ryth-style reaction compilation with real gTTS voiceover.
"""

from __future__ import annotations

import os
import sys
import subprocess
import tempfile
import logging
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.reaction_workflow import edit_clip_ryth_style, generate_voiceover_and_srt, get_script_for_topic
from python.engines.character_overlay import CharacterOverlayEngine
from python.utils.google_drive_uploader import GoogleDriveUploader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("local_input_processor")


def main():
    source_video = r"C:\Users\admin\Documents\Default Project\project\assets\videos\input.mp4"
    output_path = r"C:\Users\admin\.gemini\antigravity-ide\scratch\video-edit-engine\output\youtube_short_reaction_final.mp4"
    temp_dir = tempfile.mkdtemp(prefix="local_input_reaction_")
    
    logger.info("=== STARTING LOCAL INPUT VIDEO REACTION GENERATOR ===")
    
    # 1. Split local input.mp4 into 3 segments of 5 seconds each
    clips = []
    for i in range(3):
        start_time = i * 5
        clip_path = os.path.join(temp_dir, f"split_clip_{i+1}.mp4")
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", str(start_time),
            "-i", source_video,
            "-t", "5.0",
            "-vf", "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-c:a", "aac",
            clip_path
        ]
        subprocess.run(cmd, check=True)
        clips.append(clip_path)

    # 2. Create sound effect
    sfx_path = os.path.join(temp_dir, "boing_sfx.wav")
    cmd_sfx = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "sine=frequency=400:duration=0.5",
        "-af", "tremolo=f=10:d=0.8",
        "-t", "0.5",
        sfx_path
    ]
    subprocess.run(cmd_sfx, check=True)
    
    # 3. Apply Ryth-style editing to each segment
    edited_clips = []
    for i, clip in enumerate(clips):
        moment_num = 3 - i
        out_clip_path = os.path.join(temp_dir, f"edited_moment_{moment_num}.mp4")
        if edit_clip_ryth_style(clip, out_clip_path, moment_num, sfx_path, temp_dir):
            edited_clips.append(out_clip_path)
            
    # 4. Generate commentary and voiceover (using now-installed gTTS)
    script_lines = [
        "Moment Number Three: That was an absolute classic fail, check out his reaction.",
        "Moment Number Two: Slipped on the floor and dropped the weights. Be careful next time.",
        "And Moment Number One: This is the ultimate workout fail of the year."
    ]
    audio_vo, srt_subs = generate_voiceover_and_srt(script_lines, temp_dir)
    
    # 5. Concat the edited clips together
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
    
    # 6. Add Voiceover audio track
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
    
    # 7. Apply Character Reaction Overlay
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    overlay_engine = CharacterOverlayEngine()
    success = overlay_engine.overlay_avatar(
        video_path=synced_video_path,
        output_path=output_path,
        avatar_path=r"C:\Users\admin\.gemini\antigravity-ide\scratch\video-edit-engine\assets\logos\fox_observer.png",
        position="bottom_right",
        scale=0.3
    )
    
    if success:
        logger.info(f"Successfully compiled custom reaction video: {output_path}")
        # Upload
        uploader = GoogleDriveUploader()
        uploader.upload_file(output_path)
    else:
        logger.error("Failed to compile reaction video.")


if __name__ == "__main__":
    main()
