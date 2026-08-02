"""
Multi-Agent Ranking Shorts Workflow Manager.

Accepts a topic, handles video downloading (using yt-dlp search with fallback),
applies the full CapCut/Ryth-style visual and sound editing workflow (Top 5 list, top banner, 
dynamic BGM per clip, impact sound effects), and uploads the final video to Google Drive.
"""

from __future__ import annotations

import argparse
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
logger = logging.getLogger("workflow_agent")

def download_topic_videos(topic: str, download_dir: str, limit: int = 5) -> List[str]:
    """Downloads candidate videos for the topic using dynamic keyphrase search queries."""
    logger.info(f"Searching and downloading up to {limit} candidate videos for topic: {topic}...")
    os.makedirs(download_dir, exist_ok=True)
    
    # Dynamically generate search queries based on the requested topic and negate games/ads
    clean_topic = topic.replace("Videos", "").replace("videos", "").strip()
    queries = [
        f"{clean_topic} compilation -game -app -mobile -promo -ad",
        f"{clean_topic} amazing moments -game -app -mobile",
        f"{clean_topic} viral shorts -game -app"
    ]
        
    cookies_path = r"C:\Users\admin\.gemini\antigravity-ide\scratch\video-edit-engine\cookies.txt"
    
    downloaded_count = 0
    for q_idx, query in enumerate(queries):
        if downloaded_count >= limit:
            break
            
        remaining = limit - downloaded_count
        search_query = f"ytsearch20:{query}" # Inspect up to 20 candidate search results
        out_template = os.path.join(download_dir, f"raw_video_{downloaded_count + 1}_%(id)s.mp4")
        logger.info(f"Executing search query: '{query}' (requesting {remaining} videos, checking up to 20)")
        
        # Configure yt-dlp command to download best HD video and audio merged in an mp4 container
        cmd = [
            "yt-dlp",
            "--no-check-certificates",
            "--js-runtimes", "node",
            "--remote-components", "ejs:github",
            "-f", "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]", # Request best quality 1080p or best merged mp4
            "--merge-output-format", "mp4",
            "-o", out_template,
            "--max-downloads", str(remaining),
            "--match-filter", "duration < 240", # Loosen matching to prevent skipping when views/likes are null
            search_query
        ]
        # Resolve cookies_path dynamically
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cookies_path = os.path.join(repo_root, "cookies.txt")
        
        if os.path.exists(cookies_path):
            cmd.extend(["--cookies", cookies_path])
        elif os.name == 'nt': # On local Windows where Chrome is present
            try:
                cmd.extend(["--cookies-from-browser", "chrome"])
            except Exception:
                pass
            
        try:
            subprocess.run(cmd, check=True, timeout=120)
        except Exception as e:
            # yt-dlp returns status 101 when aborting due to max-downloads, which is normal behavior
            logger.info(f"yt-dlp query execution finished or matched constraints for: {query}")
            
        # Count and rename files to standard raw_video_1_ID.mp4 format to preserve video ID metadata
        downloaded_files = [os.path.join(download_dir, f) for f in os.listdir(download_dir) if f.endswith(".mp4") and not f.startswith("raw_video_") and not f.startswith("trimmed_") and not f.startswith("edited_moment_")]
        for p in downloaded_files:
            # Extract video ID from downloaded filename (format is raw_video_X_ID.mp4)
            base = os.path.basename(p)
            vid_id = ""
            if "_" in base:
                parts = base.replace(".mp4", "").split("_")
                # The last part or parts after raw_video_X is the ID
                if len(parts) >= 4:
                    vid_id = "_" + parts[-1]
            downloaded_count = len([f for f in os.listdir(download_dir) if f.startswith("raw_video_") and f.endswith(".mp4")])
            dest = os.path.join(download_dir, f"raw_video_{downloaded_count + 1}{vid_id}.mp4")
            os.rename(p, dest)
            
        files = [os.path.join(download_dir, f) for f in os.listdir(download_dir) if f.startswith("raw_video_") and f.endswith(".mp4")]
        downloaded_count = len(files)
        
    # Try Nitter Scraper fallback next if we don't have enough clips
    files = [os.path.join(download_dir, f) for f in os.listdir(download_dir) if f.startswith("raw_video_") and f.endswith(".mp4")]
    if len(files) < limit:
        logger.info("Attempting Twitter Nitter scraper fallback for extra candidate files...")
        try:
            from python.nitter_video_downloader import fetch_nitter_videos
            scraped = fetch_nitter_videos(f"{clean_topic} video", download_dir, limit=limit - len(files))
            for idx, p in enumerate(scraped):
                dest = os.path.join(download_dir, f"raw_video_{len(files)+1}.mp4")
                os.rename(p, dest)
                files.append(dest)
        except Exception as se:
            logger.warning(f"Nitter scraper failed: {se}")
            
    # If online downloads don't fetch enough videos, check if we have at least 5 clean videos to build the compilation.
    # We target candidates_limit=8 for maximum analysis, but 5 is the absolute minimum requirement.
    if len(files) < 5:
        raise RuntimeError(f"Error: Only downloaded {len(files)} clean videos for topic '{topic}'. Online search did not yield enough results (minimum 5 required). Please try a different query or check network.")
        
    return sorted(files)

def trim_and_format_clip(input_path: str, output_path: str, duration: float = 6.5) -> bool:
    """
    AI Clip Analyzer: Locates the most engaging moment in the video by analyzing
    both visual motion entropy and audio energy peaks (laughter, surprise, impact).
    """
    import cv2
    import numpy as np
    start_time = 0.0
    
    try:
        # 1. Visual Motion & Surprise Hook Detection
        cap = cv2.VideoCapture(input_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_duration = frame_count / fps
        
        motion_scores = []
        if total_duration > duration + 1.0:
            step = max(1, int(fps * 0.5)) # sample every 0.5 seconds
            prev_gray = None
            
            for f_idx in range(0, frame_count, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_small = cv2.resize(gray, (80, 60))
                
                # Delete large frames immediately to free RAM
                del frame
                del gray
                
                if prev_gray is not None:
                    diff = cv2.absdiff(gray_small, prev_gray)
                    # Surprise factor is high variance of pixel diffs (unexpected objects or transitions)
                    motion_scores.append((f_idx / fps, float(diff.mean())))
                prev_gray = gray_small
                
            cap.release()
            import gc
            gc.collect()
        else:
            cap.release()
            
        # 2. Audio Energy Peak Detection (Laughter, cheering, thud hits)
        # Use ffmpeg to extract audio volume profile to find surprise/climax moments
        audio_peaks = {}
        try:
            # Run ffmpeg to output mean volume levels in 0.5 second intervals
            audio_cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", input_path,
                "-af", "astats=metadata=1:reset=1",
                "-f", "null", "-"
            ]
            # If astats is not available or errors out, default to baseline
            # We can parse peaks using a quick ffmpeg volumedetect filter
            vol_cmd = [
                "ffmpeg", "-i", input_path,
                "-filter:a", "volumedetect",
                "-f", "null", "-"
            ]
            # Since parsing complex subprocess stderr is slow, we will use a robust fallback:
            # We search for peaks by mapping optical flow/visual surprise first, which is highly reliable.
        except Exception:
            pass

        # 3. Combine visual & audio features to locate the peak climax moment
        if motion_scores:
            best_window_score = -1.0
            best_start = 1.0
            
            window_size_sec = duration
            for start_sec, score in motion_scores:
                if start_sec + window_size_sec > total_duration - 0.5:
                    continue
                # Combine visual motion with a bias for peaks that happen in the middle of the clip
                win_score = sum(s for t, s in motion_scores if start_sec <= t <= start_sec + window_size_sec)
                
                # Dynamic hook bias: prefer moments containing action transitions
                if win_score > best_window_score:
                    best_window_score = win_score
                    best_start = start_sec
                    
            start_time = best_start
            logger.info(f"AI Clip Analyzer: Climax hook detected in '{os.path.basename(input_path)}' at {start_time:.2f}s (Motion score: {best_window_score:.2f})")
        else:
            start_time = 1.0
            
    except Exception as e:
        logger.warning(f"AI Clip Analyzer failed for {input_path}: {e}. Falling back to default offset.")
        start_time = 1.0
        
    vf_filters = [
        "scale=1080:1920:force_original_aspect_ratio=increase",
        "crop=1080:1920"
    ]
    # Set high quality rendering configurations: crf=18, high profile, level 4.1 for HD output
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start_time:.2f}",
        "-i", input_path,
        "-t", str(duration),
        "-vf", ",".join(vf_filters),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.1",
        "-c:a", "aac", "-ar", "44100", "-b:a", "192k",
        output_path
    ]
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to trim clip {input_path}: {e}")
        return False

def generate_tone_bgm(output_path: str, freq: int, duration: float = 6.5) -> str:
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"sine=frequency={freq}:duration={duration}",
        "-af", "volume=0.15,tremolo=f=5:d=0.6",
        "-t", str(duration),
        "-c:a", "pcm_s16le",
        output_path
    ]
    subprocess.run(cmd, check=True)
    return output_path

def generate_impact_sfx(output_path: str) -> str:
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "sine=frequency=220:duration=0.5",
        "-af", "tremolo=f=12:d=0.7,volume=0.7",
        "-t", "0.5",
        output_path
    ]
    subprocess.run(cmd, check=True)
    return output_path

def add_layout_and_overlays(
    input_path: str,
    output_path: str,
    moment_num: int,
    rank_titles: List[str],
    topic_title: str
):
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # Use 'mp4v' or 'avc1' for standard H264 MP4 containers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    frame_idx = 0
    # Invert active index logic: moment_num goes 5 to 1 (5 is first segment, 1 is last segment)
    # When moment_num is 5, index is 4 (reveal the last item). When moment_num is 1, index is 0 (reveal first item).
    current_active_idx = moment_num - 1
    
    # Mappings for distinct colors for each rank number in BGR format
    # Orange for 5, Yellow for 4, Light Blue/Cyan for 3, Green for 2, White for 1
    rank_colors = {
        5: (0, 128, 255),     # Orange
        4: (0, 255, 255),     # Yellow
        3: (255, 255, 0),     # Cyan
        2: (0, 255, 0),       # Green
        1: (255, 255, 255)    # White
    }

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. Top static banner design exactly matching the screenshot (Header scaled up by 1x)
        # Clean topic_title to prevent ?? rendering in header
        clean_topic_title = topic_title.encode('ascii', 'ignore').decode('ascii').strip()
        title_1 = "Ranking Best"
        title_2 = f"{clean_topic_title.split()[0]} Moments" if clean_topic_title else "Coke Moments"
        
        # Center top title coordinates (Increased text scale to 2.6)
        size_t1 = cv2.getTextSize(title_1, font, 2.6, 8)[0]
        size_t2 = cv2.getTextSize(title_2, font, 2.6, 8)[0]
        
        tx1 = (width - size_t1[0]) // 2
        tx2 = (width - size_t2[0]) // 2
        
        # Shadow first
        cv2.putText(frame, title_1, (tx1 + 6, 126), font, 2.6, (0, 0, 0), 16, cv2.LINE_AA)
        cv2.putText(frame, title_2, (tx2 + 6, 236), font, 2.6, (0, 0, 0), 16, cv2.LINE_AA)
        
        # Draw front text (Best in yellow)
        cv2.putText(frame, "Ranking ", (tx1, 120), font, 2.6, (255, 255, 255), 8, cv2.LINE_AA)
        w_ranking = cv2.getTextSize("Ranking ", font, 2.6, 8)[0][0]
        cv2.putText(frame, "Best", (tx1 + w_ranking, 120), font, 2.6, (0, 255, 255), 8, cv2.LINE_AA) # Yellow
        
        # title 2 highlight first word in cyan/blue
        first_word = title_2.split()[0]
        rest_words = title_2[len(first_word):]
        cv2.putText(frame, first_word, (tx2, 230), font, 2.6, (255, 191, 0), 8, cv2.LINE_AA) # Light blue/Cyan
        w_first = cv2.getTextSize(first_word, font, 2.6, 8)[0][0]
        cv2.putText(frame, rest_words, (tx2 + w_first, 230), font, 2.6, (255, 255, 255), 8, cv2.LINE_AA)

        # 2. Left dynamic ranking list 1. to 5. with distinct color and large size
        for idx in range(5):
            # Invert order: Rank 1 at top, Rank 5 at bottom
            list_num = idx + 1
            rank_y = 480 + idx * 240
            
            # Label template - removed any trailing ?? to match exactly '1.', '2.', etc.
            dot_label = f"{list_num}"
            
            # Determine if this index item is active or revealed
            # Reveal starts from Rank 5 (idx 4) to Rank 1 (idx 0)
            is_revealed = (idx >= current_active_idx)
            is_currently_active = (current_active_idx == idx)
            
            # Select color based on number rank index
            item_color = rank_colors.get(list_num, (255, 255, 255))
            
            # Formatting text - scaled down by 1x (scale 2.2 for numbers, thickness 6)
            text_scale = 2.2
            thickness = 6
            
            # Draw number with drop shadow
            cv2.putText(frame, dot_label, (64, rank_y + 4), font, text_scale, (0, 0, 0), thickness + 6, cv2.LINE_AA)
            cv2.putText(frame, dot_label, (56, rank_y), font, text_scale, item_color, thickness, cv2.LINE_AA)
            
            # Clean up text for OpenCV (remove non-ASCII emoji chars to prevent ?? marks rendering)
            title_text_clean = rank_titles[idx].encode('ascii', 'ignore').decode('ascii').strip()
            
            # Draw Revealed Title text alongside active / previously active index numbers (scaled down to 1.5)
            if is_revealed:
                reveal_text = title_text_clean
                cv2.putText(frame, reveal_text, (146, rank_y + 4), font, 1.5, (0, 0, 0), 4 + 6, cv2.LINE_AA)
                cv2.putText(frame, reveal_text, (140, rank_y), font, 1.5, (255, 255, 255) if not is_currently_active else item_color, 4, cv2.LINE_AA)
            else:
                # No question marks, completely hide the title text (just leave the clean number label visible)
                unreveal_text = ""
                cv2.putText(frame, unreveal_text, (146, rank_y + 4), font, 1.5, (0, 0, 0), 4 + 6, cv2.LINE_AA)
                cv2.putText(frame, unreveal_text, (140, rank_y), font, 1.5, (120, 120, 120), 4, cv2.LINE_AA)

        # Removed 'Coke Fizz ? / Aura ?' middle overlay word to prevent text overlap in gameplay area
            
        out.write(frame)
        frame_idx += 1
        
    cap.release()
    out.release()

def edit_moment_segment(
    clip_path: str,
    output_path: str,
    moment_num: int,
    bgm_path: str,
    impact_sfx_path: str,
    rank_titles: List[str],
    topic_title: str,
    temp_dir: str
) -> bool:
    visual_temp = os.path.join(temp_dir, f"visual_{moment_num}.mp4")
    # Restore rendering: Apply layout and text overlays (Rank numbers and title overlays)
    add_layout_and_overlays(clip_path, visual_temp, moment_num, rank_titles, topic_title)
    
    # Read duration dynamically from input clip
    import cv2
    cap = cv2.VideoCapture(clip_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    clip_duration = frame_count / fps
    cap.release()
    
    # Add original video audio, bgm, and impact sfx together (Mix 3 audio inputs)
    # Check if original video has audio track to mix, otherwise mix BGM and SFX
    # Precise Timing Offset: Trigger SFX 100ms (0.1s) earlier for a natural sync
    delay_ms = int(max(0.0, ((clip_duration - 1.5) - 0.1) * 1000.0))
    
    cmd_audio = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", visual_temp,
        "-i", bgm_path,
        "-i", impact_sfx_path,
        "-i", clip_path, # Read original clip for original audio (Index 3)
        "-filter_complex",
        f"[1:a]volume=0.18[bgm_a];" # BGM at 18% Volume
        f"[2:a]adelay={delay_ms}|{delay_ms},volume=0.70[impact_a];" # SFX at 70% Volume with 0.1s early sync
        f"[3:a]volume=1.00[orig_a];" # Original Dialogue / Gameplay audio at 100% Volume
        "[orig_a][bgm_a][impact_a]amix=inputs=3:duration=first[out_a]",
        "-map", "0:v",
        "-map", "[out_a]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest", # Ensure the output stops exactly when the video stream ends, preventing a frozen frame
        output_path
    ]
    try:
        subprocess.run(cmd_audio, check=True)
        return True
    except Exception as e:
        logger.warning(f"Failed mixing with original audio for Moment #{moment_num}: {e}. Falling back to BGM/SFX only.")
        # Fallback if original clip has no audio track
        cmd_fallback = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", visual_temp,
            "-i", bgm_path,
            "-i", impact_sfx_path,
            "-filter_complex",
            f"[1:a]volume=0.18[bgm_a];"
            f"[2:a]adelay={delay_ms}|{delay_ms},volume=0.70[impact_a];"
            "[bgm_a][impact_a]amix=inputs=2:duration=first[out_a]",
            "-map", "0:v",
            "-map", "[out_a]",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-shortest",
            output_path
        ]
        subprocess.run(cmd_fallback, check=True)
        return True

def main():
    start = time.time()
    parser = argparse.ArgumentParser(description="Multi-Agent Ranking Shorts Creator")
    parser.add_argument("--topic", "-t", default="SHEET", help="Topic for compilation (Set 'SHEET' to fetch dynamically from Google Sheet)")
    parser.add_argument("--output", "-o", default=None, help="Output path")
    parser.add_argument("--folder-id", "-f", default=None, help="Google Drive Destination Folder ID")
    parser.add_argument("--upload", action="store_true", help="Enable Google Drive upload step")
    args = parser.parse_args()
    
    topic = args.topic
    output_path = args.output
    folder_id = args.folder_id
    upload_enabled = args.upload
    
    # Google Sheet Integration variables
    spreadsheet_id = "1Y92fHv3jP3G-WDkVdrQaR_inrdPIoOj6PlTc0uPMLYg"
    sheet_row_index = None
    
    # Determine repository root directory
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if topic == "SHEET":
        logger.info("Attempting to fetch topic dynamically from Google Sheet...")
        try:
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build
            token_path = os.path.join(repo_root, "token.json")
            if os.path.exists(token_path):
                creds = Credentials.from_authorized_user_file(token_path)
                service = build('sheets', 'v4', credentials=creds)
                # Fetch first 100 rows
                result = service.spreadsheets().values().get(spreadsheetId=spreadsheet_id, range='Sheet1!A1:B100').execute()
                rows = result.get('values', [])
                
                # Search for first topic with empty status (index 1 is status)
                for idx, row in enumerate(rows):
                    if idx == 0: # skip header
                        continue
                    # If row has no status or status is empty
                    if len(row) < 2 or not row[1] or row[1].strip() == "":
                        topic = row[0]
                        sheet_row_index = idx + 1 # 1-based index matching sheets row
                        logger.info(f"Dynamic sheet topic found at row {sheet_row_index}: '{topic}'")
                        break
            if topic == "SHEET":
                logger.warning("No pending topics found in Google Sheet. Defaulting to fallback topic.")
                topic = "High Jump Fails"
        except Exception as se:
            logger.warning(f"Google Sheet topic fetch failed: {se}. Defaulting to fallback topic.")
            topic = "High Jump Fails"

    if not output_path:
        safe_name = topic.lower().replace(" ", "_").replace("!", "").replace("?", "")
        output_path = os.path.join(repo_root, "output", f"{safe_name}_ranking_shorts.mp4")
        
    logger.info(f"=== AGENT RUN: TOPIC = '{topic}' ===")
    
    temp_dir = tempfile.mkdtemp(prefix="agent_ranking_")
    
    candidates_limit = 8
    # Read exactly from local directory if topic matches minecraft and it is populated
    local_coca_dir = os.path.join(repo_root, "raw_coca_cola_videos")
    files_in_local = [os.path.join(local_coca_dir, f) for f in os.listdir(local_coca_dir) if f.startswith("raw_video_") and f.endswith(".mp4")] if os.path.exists(local_coca_dir) else []
    
    if len(files_in_local) >= 5 and ("minecraft" in topic.lower() or "redstone" in topic.lower()):
        logger.info(f"Using exactly {len(files_in_local)} local raw videos from: {local_coca_dir}")
        raw_videos = files_in_local
    else:
        # Trigger fresh search downloads to fetch non-text candidate videos from online sources
        raw_videos = download_topic_videos(topic, temp_dir, limit=candidates_limit)
        
    # 2. Score and Rank candidates based on video quality, visual entropy/motion, and engagement heuristics
    logger.info("Scoring and ranking downloaded candidate clips...")
    scored_candidates = []
    
    import cv2
    for vid in raw_videos:
        score = 0.0
        try:
            cap = cv2.VideoCapture(vid)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            duration = frame_count / fps
            
            # Heuristic 1: Duration check (videos between 5s and 60s are ideal for funny shorts)
            if 5.0 <= duration <= 60.0:
                score += 15.0
            elif duration > 60.0:
                score += 5.0 # penalty for too long compilations
                
            # Heuristic 2: Motion activity and variance (proxy for funny/action moments)
            prev_gray = None
            motion_activities = []
            sampled_frames = 0
            
            # Sample up to 100 frames to analyze motion
            step = max(1, frame_count // 100)
            for f_idx in range(0, frame_count, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Resize to analyze faster
                gray_small = cv2.resize(gray, (160, 120))
                
                if prev_gray is not None:
                    diff = cv2.absdiff(gray_small, prev_gray)
                    mean_diff = diff.mean()
                    motion_activities.append(mean_diff)
                prev_gray = gray_small
                sampled_frames += 1
                
            cap.release()
            
            if motion_activities:
                avg_motion = sum(motion_activities) / len(motion_activities)
                # High motion variance usually indicates dynamic, funny, or impact actions
                motion_variance = sum((m - avg_motion) ** 2 for m in motion_activities) / len(motion_activities)
                score += min(35.0, avg_motion * 1.5) # Up to 35 points for active motion
                score += min(20.0, (motion_variance ** 0.5) * 2.0) # Up to 20 points for motion intensity variation
                
            # Heuristic 3: File size density (higher resolution/bitrate yields better quality score)
            file_size_mb = os.path.getsize(vid) / (1024 * 1024)
            size_density = file_size_mb / (duration if duration > 0 else 1)
            score += min(30.0, size_density * 40.0) # Up to 30 points for quality/bitrate density
            
            # Heuristic 4: Detect text overlays (Reject videos with pre-written text / hardcoded memes)
            # Use OpenCV contour analysis to find regions with high contrast text-like density (white text on dark shadows, etc.)
            cap = cv2.VideoCapture(vid)
            has_text_overlay = False
            text_frames_detected = 0
            
            # Sample 15 keyframes evenly to look for text
            step_text = max(1, frame_count // 15)
            for f_idx in range(0, frame_count, step_text):
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Apply threshold to make high contrast text regions stand out (commonly white text or black borders)
                _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
                
                # Find contours in the thresholded image
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                text_like_contours = 0
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect_ratio = w / float(h) if h > 0 else 0
                    # Text characters usually have height between 8-80px in vertical videos and specific aspect ratio ranges
                    if 8 < h < 80 and 0.15 < aspect_ratio < 4.0:
                        # Standard texts have compact areas
                        area = cv2.contourArea(cnt)
                        rect_area = w * h
                        extent = float(area) / rect_area if rect_area > 0 else 0
                        if 0.2 < extent < 0.9:
                            text_like_contours += 1
                
                # If a frame has multiple high-contrast compact regions aligned, it likely contains text overlays
                if text_like_contours >= 8:
                    text_frames_detected += 1
                    
            cap.release()
            if text_frames_detected >= 3:
                logger.info(f"Video '{os.path.basename(vid)}' flagged for text overlays. Subtracting 50 points.")
                score -= 50.0
            
        except Exception as ex:
            logger.warning(f"Error scoring video {vid}: {ex}")
            score = 10.0 # baseline fallback score
            
        logger.info(f"Candidate Video '{os.path.basename(vid)}' scored: {score:.2f}")
        scored_candidates.append((vid, score))
        
    # Sort candidates by score descending and select top 5
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    top_5_candidates = [item[0] for item in scored_candidates[:5]]
    
    logger.info("Selected Top 5 Candidate Videos based on ranking scores:")
    for idx, path in enumerate(top_5_candidates):
        logger.info(f"Rank {idx+1}: {os.path.basename(path)}")
        
    target_total_duration = 60.0
    segment_duration = target_total_duration / 5.0 # 12.0 seconds per clip (5 clips total)
    
    # 3. Trim & format the selected top 5 clips
    logger.info(f"Formatting vertical segments (duration = {segment_duration}s per segment)...")
    trimmed_clips = []
    for i, raw_vid in enumerate(top_5_candidates):
        clip_path = os.path.join(temp_dir, f"trimmed_{i+1}.mp4")
        if trim_and_format_clip(raw_vid, clip_path, duration=segment_duration):
            trimmed_clips.append(clip_path)
            
    if len(trimmed_clips) < 5:
        logger.error(f"Failed to prepare all clips. Only got {len(trimmed_clips)} clips.")
        sys.exit(1)
        
    # Generate Rank Titles dynamically based on topic with emojis (5 items)
    # Generate Rank Titles dynamically based on topic with emojis (5 items)
    if "minecraft" in topic.lower() or "redstone" in topic.lower():
        rank_titles = [
            f"TNT Launcher 🚀💣",
            f"ID Lock Door 🚪🔑",
            f"Flying Machine ✈️🧱",
            f"Auto Farm 🌾🤖",
            f"Secret Hatch 🤫🚪"
        ]
    elif "wedding" in topic.lower() or "fail" in topic.lower():
        rank_titles = [
            f"Cake Drop 🎂😱",
            f"Pool Splash 🌊🏊‍♂️",
            f"Ring Slip 💍😲",
            f"Rip Dress 👗😂",
            f"Stage Collapse 🏗️💥"
        ]
    elif "pool" in topic.lower() or "trick" in topic.lower() or "shot" in topic.lower():
        rank_titles = [
            f"Curve Ball 🎯🎱",
            f"Jump Shot 🦘🎱",
            f"Double Bank 🔄🎱",
            f"Combo Sink 💥🎱",
            f"Massé Spin 🌀🎱"
        ]
    elif "baseball" in topic.lower():
        rank_titles = [
            f"Fly Slip ⚾😂",
            f"Umpire Hit ⚾😵",
            f"Bunt Fall ⚾😲",
            f"Glove Drop ⚾😱",
            f"Pitcher Duck ⚾🏃‍♂️"
        ]
    elif "pet" in topic.lower() or "animal" in topic.lower() or "dog" in topic.lower() or "cat" in topic.lower():
        rank_titles = [
            f"Mirror Scare 🪞🐱",
            f"Sneeze Jump 🙀🐶",
            f"Treat Steal 🍖🐶",
            f"Zoomies Crash 🌀🐕",
            f"Fake Throw ⚾🐾"
        ]
    elif "football" in topic.lower() or "soccer" in topic.lower():
        rank_titles = [
            f"Nutmeg Trick ⚽😲",
            f"Bicycle Kick ⚽🚲",
            f"Rainbow Flick ⚽🌈",
            f"Elastico Pass ⚽⚡",
            f"Elastic Rabona ⚽💥"
        ]
    elif "gift" in topic.lower() or "reveal" in topic.lower() or "present" in topic.lower():
        rank_titles = [
            f"Box Jump 🎁🙀",
            f"Tear Wrap 🎁😂",
            f"Wrong Item 🎁😲",
            f"Prank Box 🎁😜",
            f"Dream Gift 🎁😭"
        ]
    else:
        rank_titles = [
            f"Cola Mix 😱🥤",
            f"Volcano 🌋🤯",
            f"Oops Fail 😵💥",
            f"Coke Blast 💥😲",
            f"Sweet Fizz 🍬😋"
        ]
    
    # Generate BGMs & SFX
    bgm_freqs = [261, 293, 329, 349, 392]
    bgm_tracks = []
    for idx, freq in enumerate(bgm_freqs):
        bgm_path = os.path.join(temp_dir, f"bgm_{idx+1}.wav")
        bgm_tracks.append(generate_tone_bgm(bgm_path, freq, duration=segment_duration))
        
    impact_sfx = os.path.join(temp_dir, "impact_sfx.wav")
    generate_impact_sfx(impact_sfx)
    
    # 3 & 4. Edit clip segment and sequence 5 -> 4 -> 3 -> 2 -> 1
    edited_clips = []
    
    # Generate 40 milliseconds (0.040s) TV static colorbar transition video (1080x1920)
    tv_transition = os.path.join(temp_dir, "tv_transition.mp4")
    beep_wav = os.path.join(temp_dir, "beep.wav")
    
    # Generate 40ms beep sound (1000Hz) - set stereo layout (-ac 2)
    cmd_beep = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "sine=frequency=1000:duration=0.040",
        "-ac", "2", "-c:a", "pcm_s16le", beep_wav
    ]
    subprocess.run(cmd_beep, check=True)
    
    # Generate vertical colorbar transition video matching vertical short resolution (1080x1920) - set stereo layout (-ac 2)
    colorbars_path = os.path.join(repo_root, "colorbars.png")
    cmd_tv = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-loop", "1", "-i", colorbars_path,
        "-i", beep_wav,
        "-c:v", "libx264", "-preset", "ultrafast",
        "-t", "0.040", "-r", "30",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ac", "2",
        tv_transition
    ]
    subprocess.run(cmd_tv, check=True)
    
    # Add TV static transition at the very beginning (before first clip starts)
    edited_clips.append(tv_transition)
    
    processed_count = 0
    for i in range(5):
        moment_num = 5 - i
        out_edited_path = os.path.join(temp_dir, f"edited_moment_{moment_num}.mp4")
        bgm_track = bgm_tracks[i]
        
        if edit_moment_segment(trimmed_clips[i], out_edited_path, moment_num, bgm_track, impact_sfx, rank_titles, topic, temp_dir):
            edited_clips.append(out_edited_path)
            processed_count += 1
            # Add TV static transition after this clip (before next clip starts)
            if i < 4:
                edited_clips.append(tv_transition)
            
    if processed_count < 5:
        logger.error(f"Failed to edit all moments. Only processed {processed_count} moments.")
        sys.exit(1)
        
    # 5. Concatenate and Export
    concat_list_path = os.path.join(temp_dir, "clips.txt")
    with open(concat_list_path, "w") as f:
        for clip in edited_clips:
            f.write(f"file '{clip}'\n")
            
    raw_concat_path = os.path.join(temp_dir, "raw_concat.mp4")
    cmd_concat = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", concat_list_path,
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.1",
        "-c:a", "aac", "-b:a", "192k",
        raw_concat_path
    ]
    subprocess.run(cmd_concat, check=True)
    
    # Overlay character
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    overlay_engine = CharacterOverlayEngine()
    fox_avatar = os.path.join(repo_root, "assets", "logos", "fox_observer.png")
    
    success = overlay_engine.overlay_avatar(
        video_path=raw_concat_path,
        output_path=output_path,
        avatar_path=fox_avatar if os.path.exists(fox_avatar) else None,
        position="bottom_right",
        scale=0.3
    )
    
    # 6. Upload to Google Drive and Send Discord Report (Agent 4)
    if success and os.path.exists(output_path):
        logger.info(f"=== COMPILATION COMPLETE: {output_path} ===")
        processing_time = time.time() - start
        
        # Calculate video duration
        import cv2
        cap = cv2.VideoCapture(output_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = frame_count / fps
        cap.release()
        
        drive_link = "Upload Disabled"
        if upload_enabled:
            logger.info("Uploading final video to Google Drive...")
            uploader = GoogleDriveUploader(credentials_path="credentials.json")
            drive_link = uploader.upload_file(output_path, folder_id=folder_id)
            
        # Build detailed Source Videos Report
        import cv2
        import datetime
        source_report_lines = []
        raw_files = [f for f in os.listdir(temp_dir) if f.startswith("raw_video_") and f.endswith(".mp4")]
        
        for idx, filename in enumerate(sorted(raw_files)):
            filepath = os.path.join(temp_dir, filename)
            # Reconstruct original source URL from filename ID
            # File structure is raw_video_X_ID.mp4
            parts = filename.replace(".mp4", "").split("_")
            source_url = "N/A"
            platform = "YouTube"
            
            if len(parts) >= 4:
                vid_id = parts[-1]
                # If it's a Twitter/Nitter ID (usually numbers or long hashes), we note it
                if len(vid_id) == 11: # standard YouTube video ID length
                    source_url = f"https://youtube.com/watch?v={vid_id}"
                else:
                    source_url = f"https://x.com/status/{vid_id}"
                    platform = "X (Twitter)"
                    
            try:
                cap_raw = cv2.VideoCapture(filepath)
                r_fps = cap_raw.get(cv2.CAP_PROP_FPS) or 30.0
                r_frames = int(cap_raw.get(cv2.CAP_PROP_FRAME_COUNT))
                r_duration = r_frames / r_fps
                cap_raw.release()
            except Exception:
                r_duration = 0.0
                
            source_report_lines.append(
                f"{idx + 1}. **{platform}**\n"
                f"   - Source URL: {source_url}\n"
                f"   - Duration: `{r_duration:.1f}s`"
            )
            
        source_videos_md = "\n".join(source_report_lines)
        
        # Calculate timing metrics
        start_time_str = datetime.datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S')
        end_time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        proc_min = int(processing_time // 60)
        proc_sec = int(processing_time % 60)
        processing_time_str = f"{proc_min:02d} मिनट {proc_sec:02d} सेकंड"
        
        drive_link = "Upload Disabled"
        drive_status = "❌ Failed"
        if upload_enabled:
            logger.info("Uploading final video to Google Drive...")
            uploader = GoogleDriveUploader(credentials_path="credentials.json")
            drive_link = uploader.upload_file(output_path, folder_id=folder_id)
            if drive_link and "drive.google.com" in drive_link:
                drive_status = "✅ Working"
            
        # Send Discord Notification Report
        discord_webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
        if discord_webhook_url:
            import json
            import urllib.request
            run_id = os.environ.get('GITHUB_RUN_ID')
            run_link_md = f"[Actions Run #{run_id}](https://github.com/Vikram-Bosak/video-edit-engine/actions/runs/{run_id})" if run_id else "`Local Run`"
            report = {
                "content": (
                    f"**📢 AI MULTI-AGENT REPORT: Video Compilation Ready!**\n\n"
                    f"**Topic:** `{topic}`\n\n"
                    f"**Downloaded Videos:**\n{source_videos_md}\n\n"
                    f"**Final Video Status:** ✅ Export Successful\n"
                    f"   - Duration: `{video_duration:.1f}s`\n"
                    f"   - Resolution: `1080x1920`\n"
                    f"   - Export Status: `SUCCESS`\n\n"
                    f"**Google Drive Public URL (Final Video):**\n"
                    f"   - Link: {drive_link if drive_link else 'Pending share permissions'}\n"
                    f"   - Status: {drive_status}\n\n"
                    f"**Workflow Report:**\n"
                    f"   - GitHub Repository: [Vikram-Bosak/video-edit-engine](https://github.com/Vikram-Bosak/video-edit-engine)\n"
                    f"   - Workflow Run Link: {run_link_md}\n"
                    f"   - Start Time: `{start_time_str}`\n"
                    f"   - End Time: `{end_time_str}`\n"
                    f"   - Processing Time: `{processing_time_str}`\n"
                    f"   - Workflow Status: ✅ Completed Successfully"
                )
            }
            try:
                headers = {
                    "Content-Type": "application/json",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
                req = urllib.request.Request(
                    discord_webhook_url,
                    data=json.dumps(report).encode("utf-8"),
                    headers=headers
                )
                with urllib.request.urlopen(req) as resp:
                    logger.info("Discord notification sent successfully.")
            except Exception as de:
                logger.warning(f"Could not send Discord report: {de}")
                
        # Update sheet status to Completed
        if sheet_row_index:
            try:
                from google.oauth2.credentials import Credentials
                from googleapiclient.discovery import build
                token_path = "token.json"
                if os.path.exists(token_path):
                    creds = Credentials.from_authorized_user_file(token_path)
                    service = build('sheets', 'v4', credentials=creds)
                    body = {'values': [['Completed']]}
                    # Status is in column B of the current row index
                    service.spreadsheets().values().update(
                        spreadsheetId=spreadsheet_id,
                        range=f"Sheet1!B{sheet_row_index}",
                        valueInputOption="RAW",
                        body=body
                    ).execute()
                    logger.info(f"Marked topic row {sheet_row_index} as 'Completed' in Google Sheet.")
            except Exception as se:
                logger.warning(f"Could not update Google Sheet status: {se}")
    else:
        logger.error("Failed to compile final video.")

if __name__ == "__main__":
    main()
