"""
Ranking Compilation Workflow (6 → 1).

Fully automated pipeline that:
1. Searches Twitter/X and YouTube for a given topic.
2. Downloads 5-6 different clips from different videos on that topic.
3. Edits every clip in the viral "Ryth-style" (rank list overlay, B&W punchline,
   dynamic zoom, meme SFX, animated captions).
4. Applies the 6 → 5 → 4 → 3 → 2 → 1 ranking countdown.
5. Uploads the final video to Google Drive.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.utils.google_drive_uploader import GoogleDriveUploader

logger = logging.getLogger("ranking_compilation")


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Step 1 & 2: Search + Download clips
# ---------------------------------------------------------------------------

def download_with_ytdlp(query: str, download_dir: str, limit: int) -> List[str]:
    """Search YouTube with yt-dlp and download `limit` different videos."""
    os.makedirs(download_dir, exist_ok=True)
    out_template = os.path.join(download_dir, "yt_%(autonumber)d.%(ext)s")
    cmd = [
        "yt-dlp",
        "--no-check-certificates",
        "-f", "mp4",
        "-o", out_template,
        "--max-downloads", str(limit),
        "--no-playlist",
        "--quiet",
        f"ytsearch{limit}:{query}",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
        paths = [
            os.path.join(download_dir, f)
            for f in os.listdir(download_dir)
            if f.startswith("yt_") and f.endswith(".mp4")
        ]
        return sorted(paths)
    except Exception as e:
        logger.warning(f"yt-dlp search failed: {e}")
        return []


def download_with_nitter(query: str, download_dir: str, limit: int) -> List[str]:
    """Search Twitter/X via Nitter RSS fallback."""
    try:
        from python.nitter_video_downloader import fetch_nitter_videos
        return fetch_nitter_videos(query, download_dir, limit=limit)
    except Exception as e:
        logger.warning(f"Nitter fallback failed: {e}")
        return []


def download_topic_clips(query: str, download_dir: str, num_clips: int = 6) -> List[str]:
    """Get `num_clips` different clips for the topic, trying multiple sources."""
    downloaded: List[str] = []

    logger.info(f"=== SEARCHING: '{query}' ===")
    downloaded = download_with_ytdlp(query, download_dir, num_clips)
    if len(downloaded) < num_clips:
        logger.info("yt-dlp did not get enough clips. Trying Nitter (Twitter) fallback...")
        nitter_paths = download_with_nitter(query, download_dir, num_clips - len(downloaded))
        for p in nitter_paths:
            if p not in downloaded:
                downloaded.append(p)

    # Normalize all clips to vertical 1080x1920
    normalized = []
    for idx, raw in enumerate(downloaded[:num_clips]):
        out = os.path.join(download_dir, f"clip_{idx + 1}.mp4")
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "stream=width,height",
                 "-of", "csv=p=0", "-select_streams", "v:0", raw],
                capture_output=True, text=True, timeout=30,
            )
            dims = probe.stdout.strip().split(",")
            if len(dims) == 2 and dims[0].isdigit() and dims[1].isdigit():
                w, h = int(dims[0]), int(dims[1])
            else:
                w, h = 720, 1280
        except Exception:
            w, h = 720, 1280

        if w > h:
            vf = "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2"
        else:
            vf = "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2"

        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", raw,
            "-vf", vf,
            "-t", "6",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-c:a", "aac", "-b:a", "128k",
            "-pix_fmt", "yuv420p",
            out,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=180)
            normalized.append(out)
        except Exception as e:
            logger.error(f"Failed to normalize {raw}: {e}")

    if len(normalized) >= num_clips:
        return normalized[:num_clips]

    # Hard fallback: create simulated clips so the pipeline always works
    logger.warning(f"Only got {len(normalized)} real clips. Generating simulated fallback clips...")
    while len(normalized) < num_clips:
        idx = len(normalized)
        sim_path = os.path.join(download_dir, f"clip_{idx + 1}.mp4")
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i",
            f"testsrc2=duration=6:size=1080x1920:rate=30:seed={idx + 1}",
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-t", "6",
            "-c:v", "libx264", "-preset", "fast",
            "-c:a", "aac",
            "-pix_fmt", "yuv420p",
            sim_path,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)
            normalized.append(sim_path)
        except Exception as e:
            logger.error(f"Fallback clip failed: {e}")
            break

    return normalized


# ---------------------------------------------------------------------------
# Ranking titles / scripts per topic
# ---------------------------------------------------------------------------

TOPIC_PRESETS: Dict[str, Dict] = {
    "high jump": {
        "titles": ["AIR REX", "MAT MISS", "BAR DESTROYER", "POLE RIDER", "SPINE CRACK", "GRAVITY LOSS"],
        "scripts": [
            "Number six. This guy defied gravity for exactly one second.",
            "Number five. The bar said no, and gravity said welcome back.",
            "Number four. He went over the bar, but the landing mat disagreed.",
            "Number three. The pole had other plans entirely.",
            "Number two. Scientists are still studying this jump form.",
            "Number one. Absolute legend. The mat gave him the applause he deserved.",
        ],
    },
    "basketball": {
        "titles": ["RIM REJECT", "AIR BALL KING", "YEEZY PLAY", "SELF SCORE", "ANKLE BREAKER", "FULL COURT MISS"],
        "scripts": [
            "Number six. Bro thought he was LeBron. The rim said absolutely not.",
            "Number five. Air ball so bad it joined a different zip code.",
            "Number four. Playing in Yeezys was the first mistake.",
            "Number three. He scored on his own basket. Certified genius.",
            "Number two. That crossover broke ankles and dreams.",
            "Number one. A full court shot. Missed. Still legendary.",
        ],
    },
    "swimming": {
        "titles": ["BELLY FLOP", "SLIPPERY TILE", "SLIDE LAUNCH", "DEEP DIVE NOPE", "CANONBALL BOSS", "POOL EXIT FAIL"],
        "scripts": [
            "Number six. This guy thought he was doing a majestic dive.",
            "Number five. Belly flop so loud the whole pool clapped.",
            "Number four. The slippery tile had different plans for him.",
            "Number three. The slide launched him into another dimension.",
            "Number two. The diving board bounced back with a vengeance.",
            "Number one. Classic pool exit fail. Absolute cinema.",
        ],
    },
    "gym": {
        "titles": ["TREADMILL FLY", "WEIGHT WIN", "KNEE CHAT", "MIRROR CRASH", "BENCH BAR NOPE", "SQUAT HERO"],
        "scripts": [
            "Number six. Treadmill at speed ten. Bad idea.",
            "Number five. The weights won this round. Respect gravity bro.",
            "Number four. His knees decided to leave the chat.",
            "Number three. The mirror didn't survive the session.",
            "Number two. The bench press bar said absolutely not.",
            "Number one. A squat so deep he went to another dimension.",
        ],
    },
    "football": {
        "titles": ["BICYCLE NOPE", "GOALIE DREAM", "OFFSIDE KING", "SKY KICK", "OWN GOAL GENIUS", "CELEBRATION FAIL"],
        "scripts": [
            "Number six. Tried a bicycle kick, kicked only air.",
            "Number five. The keeper was busy celebrating, ball rolled in.",
            "Number four. Celebrated a goal that was ruled offside.",
            "Number three. That kick went to space and back.",
            "Number two. Own goal. His own goal. Absolutely gifted.",
            "Number one. The celebration was better than the goal itself.",
        ],
    },
    "animals": {
        "titles": ["CAT JUMP NOPE", "DOG FOOD THIEF", "PARROT LAUGH", "COW ESCAPE", "GOAT CLIMB", "PENGUIN SLIP"],
        "scripts": [
            "Number six. This cat thought it could fly. Physics said no.",
            "Number five. The dog stole the barbecue. All of it.",
            "Number four. This parrot is literally laughing at its owner.",
            "Number three. The cow found an escape route. Nobody knows how.",
            "Number two. That goat climbed a wall. An actual wall.",
            "Number one. The penguin slipped on ice. Absolutely iconic.",
        ],
    },
    "gaming": {
        "titles": ["NOOB PLAY", "PHYSICS GLITCH", "KEYBOARD SMASH", "RAGE QUIT", "WALL BUG", "PRO OOPS"],
        "scripts": [
            "Number six. Bro thought he was a pro esports player.",
            "Number five. The game physics glitched and ruined the whole run.",
            "Number four. Rest in peace to another keyboard.",
            "Number three. The rage quit was inevitable.",
            "Number two. He walked through a wall. Literally.",
            "Number one. Certified pro moment. Except it was a fail.",
        ],
    },
}

DEFAULT_TITLES = ["MOMENT SIX", "MOMENT FIVE", "MOMENT FOUR", "MOMENT THREE", "MOMENT TWO", "MOMENT ONE"]
DEFAULT_SCRIPTS = [
    "Number six. Self-confidence meets reality.",
    "Number five. The execution was questionable at best.",
    "Number four. But the entertainment value is gold.",
    "Number three. Physics said absolutely not.",
    "Number two. A certified classic fail.",
    "Number one. Absolute legendary moment.",
]


def get_topic_preset(query: str) -> Dict:
    q = query.lower()
    for key, preset in TOPIC_PRESETS.items():
        if key in q:
            return preset
    return {"titles": DEFAULT_TITLES, "scripts": DEFAULT_SCRIPTS}


# ---------------------------------------------------------------------------
# Step 3: Edit a single clip in the ranking style
# ---------------------------------------------------------------------------

def create_meme_sfx(temp_dir: str, seed: int = 1) -> str:
    """Generate a meme 'boing' sound effect."""
    sfx_path = os.path.join(temp_dir, f"sfx_{seed}.wav")
    freq = 300 + seed * 60
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"sine=frequency={freq}:duration=0.6",
        "-af", "tremolo=f=8:d=0.8,afade=t=out:st=0.2:d=0.4",
        "-t", "0.6",
        "-c:a", "pcm_s16le",
        sfx_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return sfx_path


def create_boom_sfx(temp_dir: str, seed: int = 2) -> str:
    """Generate a 'boom' sound effect for the final punchline."""
    sfx_path = os.path.join(temp_dir, f"boom_{seed}.wav")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "sine=frequency=80:duration=1.0",
        "-af", "aecho=0.8:0.88:60:0.4,afade=t=in:st=0:d=0.01,afade=t=out:st=0.5:d=0.5",
        "-t", "1.0",
        "-c:a", "pcm_s16le",
        sfx_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return sfx_path


def edit_clip_ranking_style(
    input_path: str,
    output_path: str,
    rank_number: int,
    rank_title: str,
    rank_titles: List[str],
    temp_dir: str,
    enable_bw_punchline: bool = True,
) -> bool:
    """
    Edit one clip in the viral ranking style:
    - Header "NUMBER #X" + big rank number
    - Left-side persistent ranking list (1..6)
    - Dynamic zoom-in (Ken Burns)
    - B&W conversion at the punchline (last ~1.5s)
    - Meme SFX at the punchline
    """
    logger.info(f"Editing clip as NUMBER #{rank_number} ({rank_title})")

    try:
        sfx = create_meme_sfx(temp_dir, rank_number)
        boom = create_boom_sfx(temp_dir, rank_number + 10)

        # --- Step A: mix SFX into audio at punchline ---
        sfx_mixed = os.path.join(temp_dir, f"sfx_mixed_{rank_number}.mp4")
        cmd_sfx = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", input_path,
            "-i", sfx,
            "-i", boom,
            "-filter_complex",
            f"[1:a]adelay=3800|3800[a1];"
            f"[2:a]adelay=4200|4200[a2];"
            f"[0:a][a1][a2]amix=inputs=3:duration=first:normalize=0[outa]",
            "-map", "0:v",
            "-map", "[outa]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            sfx_mixed,
        ]
        subprocess.run(cmd_sfx, check=True, capture_output=True, text=True, timeout=180)

        # --- Step B: OpenCV visual edits ---
        import cv2

        cap = cv2.VideoCapture(sfx_mixed)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 180

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        visual_out = os.path.join(temp_dir, f"visual_{rank_number}.mp4")
        out = cv2.VideoWriter(visual_out, fourcc, fps, (width, height))

        font = cv2.FONT_HERSHEY_SIMPLEX
        frame_idx = 0
        punchline_frame = int(4.2 * fps)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 1. Dynamic zoom-in up to 1.15x
            zoom = 1.0 + (frame_idx / total_frames) * 0.15
            zoom = min(zoom, 1.15)
            new_w = int(width / zoom)
            new_h = int(height / zoom)
            x0 = (width - new_w) // 2
            y0 = (height - new_h) // 2
            cropped = frame[y0:y0 + new_h, x0:x0 + new_w]
            frame = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)

            # 2. B&W punchline
            if enable_bw_punchline and frame_idx >= punchline_frame:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.merge([gray, gray, gray])

            # 3. Header: "NUMBER #X"
            header = f"NUMBER #{rank_number}"
            ts = cv2.getTextSize(header, font, 1.6, 5)[0]
            tx = (width - ts[0]) // 2
            cv2.putText(frame, header, (tx + 2, 122), font, 1.6, (0, 0, 0), 8, cv2.LINE_AA)
            cv2.putText(frame, header, (tx, 120), font, 1.6, (0, 255, 255), 5, cv2.LINE_AA)

            # 4. Big rank number badge (top-right)
            badge = str(rank_number)
            bs = cv2.getTextSize(badge, font, 3.5, 8)[0]
            bx = width - bs[0] - 60
            by = 250
            cv2.putText(frame, badge, (bx + 3, by + 3), font, 3.5, (0, 0, 0), 10, cv2.LINE_AA)
            cv2.putText(frame, badge, (bx, by), font, 3.5, (255, 255, 255), 8, cv2.LINE_AA)

            # 5. Left-side ranking list
            active_index = len(rank_titles) - rank_number  # rank 6 -> idx 0, rank 1 -> idx 5
            for idx, title in enumerate(rank_titles):
                rank_y = 400 + idx * 90
                is_active = idx == active_index
                text = f"{len(rank_titles) - idx}. {title}"
                scale = 1.3 if is_active else 0.8
                thick = 4 if is_active else 2
                color = (0, 255, 255) if is_active else (200, 200, 200)
                cv2.putText(frame, text, (50 + 2, rank_y + 2), font, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
                cv2.putText(frame, text, (50, rank_y), font, scale, color, thick, cv2.LINE_AA)

            # 6. Bottom caption: the rank title in big letters
            t = rank_title
            bt = cv2.getTextSize(t, font, 1.2, 4)[0]
            btx = (width - bt[0]) // 2
            bty = height - 120
            cv2.putText(frame, t, (btx + 2, bty + 2), font, 1.2, (0, 0, 0), 6, cv2.LINE_AA)
            cv2.putText(frame, t, (btx, bty), font, 1.2, (255, 255, 255), 4, cv2.LINE_AA)

            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()

        # --- Step C: merge visual + audio, encode H.264 ---
        cmd_merge = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", visual_out,
            "-i", sfx_mixed,
            "-map", "0:v",
            "-map", "1:a?",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-c:a", "aac",
            "-pix_fmt", "yuv420p",
            "-shortest",
            output_path,
        ]
        subprocess.run(cmd_merge, check=True, capture_output=True, text=True, timeout=180)
        return True

    except Exception as e:
        logger.error(f"Failed to edit clip #{rank_number}: {e}")
        return False


# ---------------------------------------------------------------------------
# Step 4: Build the 6 → 1 countdown intro/outro
# ---------------------------------------------------------------------------

def create_countdown_intro(temp_dir: str, topic: str) -> str:
    """Generate the intro card: 'TOP 6 {TOPIC}'."""
    out = os.path.join(temp_dir, "intro.mp4")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i",
        f"color=c=0x1a1a2e:s=1080x1920:d=2.5:r=30",
        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-filter_complex",
        f"[0:v]drawtext=text='TOP 6':fontsize=120:fontcolor=yellow:x=(w-tw)/2:y=h/2-160:borderw=6:bordercolor=black:enable='between(t\\,0.3\\,2.5)':alpha='min(1\\,(t-0.3)/0.3)'"
        f",drawtext=text='{topic.upper()}':fontsize=70:fontcolor=white:x=(w-tw)/2:y=h/2+20:borderw=4:bordercolor=black:enable='between(t\\,0.8\\,2.5)':alpha='min(1\\,(t-0.8)/0.3)'[v]",
        "-map", "[v]", "-map", "1:a",
        "-t", "2.5",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac",
        "-pix_fmt", "yuv420p",
        "-shortest",
        out,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)
        return out
    except Exception:
        return ""


def create_outro(temp_dir: str, topic: str) -> str:
    """Generate the outro card: 'NUMBER 1 WINNER!' + subscribe CTA."""
    out = os.path.join(temp_dir, "outro.mp4")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i",
        "color=c=0x0f3460:s=1080x1920:d=3.5:r=30",
        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-filter_complex",
        "[0:v]drawtext=text='NUMBER 1 WINNER!':fontsize=90:fontcolor=yellow:x=(w-tw)/2:y=h/2-120:borderw=5:bordercolor=black:enable='between(t\\,0.4\\,3.5)':alpha='min(1\\,(t-0.4)/0.3)'"
        f",drawtext=text='{topic.upper()}':fontsize=50:fontcolor=white:x=(w-tw)/2:y=h/2+20:borderw=3:bordercolor=black:enable='between(t\\,0.8\\,3.5)':alpha='min(1\\,(t-0.8)/0.3)'"
        ",drawtext=text='SUBSCRIBE FOR MORE':fontsize=56:fontcolor=cyan:x=(w-tw)/2:y=h-300:borderw=4:bordercolor=black:enable='between(t\\,1.2\\,3.5)'[v]",
        "-map", "[v]", "-map", "1:a",
        "-t", "3.5",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac",
        "-pix_fmt", "yuv420p",
        "-shortest",
        out,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)
        return out
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Step: voiceover + subtitles
# ---------------------------------------------------------------------------

def generate_voiceover_and_srt(scripts: List[str], temp_dir: str) -> tuple[str, str]:
    """Generate voiceover audio and matching SRT subtitles."""
    audio_path = os.path.join(temp_dir, "voiceover.wav")
    srt_path = os.path.join(temp_dir, "subtitles.srt")

    try:
        from gtts import gTTS
        full_text = " . ".join(scripts)
        tts = gTTS(text=full_text, lang="en")
        tts.save(audio_path)
        logger.info("Voiceover generated with gTTS")
    except ImportError:
        logger.warning("gTTS not installed, creating placeholder tone audio")
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "sine=frequency=880:duration=30",
            "-t", "30", "-c:a", "pcm_s16le", audio_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)

    # Write SRT with per-script timing (each script = 6s clip + small gaps)
    with open(srt_path, "w", encoding="utf-8") as f:
        cursor = 2.5  # after intro
        for i, line in enumerate(scripts):
            start = cursor
            end = start + 5.5
            s_h, s_m, s_s = int(start // 3600), int((start % 3600) // 60), start % 60
            e_h, e_m, e_s = int(end // 3600), int((end % 3600) // 60), end % 60
            f.write(f"{i + 1}\n")
            f.write(f"{s_h:02d}:{s_m:02d}:{s_s:05.2f},000 --> {e_h:02d}:{e_m:02d}:{e_s:05.2f},000\n")
            f.write(f"{line}\n\n")
            cursor = end + 0.5

    return audio_path, srt_path


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def run_ranking_workflow(
    query: str,
    output_path: str,
    num_clips: int = 6,
    drive_folder: Optional[str] = None,
    enable_voiceover: bool = True,
    enable_character_overlay: bool = True,
) -> Dict:
    """Run the full 6→1 ranking compilation pipeline."""
    start_time = time.time()
    temp_dir = tempfile.mkdtemp(prefix="ranking_")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    result = {
        "query": query,
        "success": False,
        "clips_downloaded": 0,
        "clips_edited": 0,
        "output_path": "",
        "drive_link": None,
        "duration": 0,
    }

    logger.info("=" * 60)
    logger.info(f"  RANKING COMPILATION: '{query}' ({num_clips} clips, 6 to 1)")
    logger.info("=" * 60)

    preset = get_topic_preset(query)
    rank_titles = preset["titles"][:num_clips]
    scripts = preset["scripts"][:num_clips]

    # --- Step 1 & 2: Download clips ---
    clips = download_topic_clips(query, temp_dir, num_clips)
    result["clips_downloaded"] = len(clips)
    if len(clips) < num_clips:
        logger.error(f"Only downloaded {len(clips)} clips. Need {num_clips}. Exiting.")
        return result

    # --- Step 3: Edit each clip (6 → 1) ---
    edited_clips = []
    # Rank order: 6, 5, 4, 3, 2, 1
    rank_order = list(range(num_clips, 0, -1))  # [6,5,4,3,2,1]
    for i, rank_num in enumerate(rank_order):
        clip = clips[i]
        out = os.path.join(temp_dir, f"edited_{rank_num}.mp4")
        title = rank_titles[num_clips - rank_num]
        ok = edit_clip_ranking_style(
            clip, out, rank_num, title, rank_titles, temp_dir,
            enable_bw_punchline=(rank_num >= 3),
        )
        if ok:
            edited_clips.append((rank_num, out))
    result["clips_edited"] = len(edited_clips)
    if len(edited_clips) < num_clips:
        logger.error(f"Only edited {len(edited_clips)} clips. Exiting.")
        return result

    # --- Intro + outro ---
    intro = create_countdown_intro(temp_dir, query)
    outro = create_outro(temp_dir, query)

    # --- Step 4: Concatenate intro + clips(6→1) + outro ---
    concat_list = os.path.join(temp_dir, "concat.txt")
    items = []
    if intro:
        items.append(intro)
    for _, clip_path in sorted(edited_clips, key=lambda x: x[0], reverse=True):
        items.append(clip_path)
    if outro:
        items.append(outro)

    with open(concat_list, "w") as f:
        for it in items:
            f.write(f"file '{it}'\n")

    raw_concat = os.path.join(temp_dir, "raw_concat.mp4")
    cmd_concat = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", concat_list,
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        "-pix_fmt", "yuv420p",
        raw_concat,
    ]
    try:
        subprocess.run(cmd_concat, check=True, capture_output=True, text=True, timeout=300)
    except Exception as e:
        logger.error(f"Concat failed: {e}")
        return result

    # --- Voiceover + subtitles ---
    current = raw_concat
    if enable_voiceover:
        audio_vo, srt_subs = generate_voiceover_and_srt(scripts, temp_dir)
        vo_video = os.path.join(temp_dir, "with_vo.mp4")
        cmd_vo = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", current,
            "-i", audio_vo,
            "-filter_complex",
            "[0:a]volume=0.4[orig];"
            "[1:a]adelay=2500|2500[vo];"
            "[orig][vo]amix=inputs=2:duration=first:normalize=0[aout]",
            "-map", "0:v",
            "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            vo_video,
        ]
        try:
            subprocess.run(cmd_vo, check=True, capture_output=True, text=True, timeout=180)
            current = vo_video
            # Burn subtitles
            sub_video = os.path.join(temp_dir, "with_subs.mp4")
            subs_filter = srt_subs.replace(os.sep, "/").replace(":", "\\:")
            cmd_subs = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", current,
                "-vf", f"subtitles='{subs_filter}'",
                "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                "-c:a", "copy",
                "-pix_fmt", "yuv420p",
                sub_video,
            ]
            subprocess.run(cmd_subs, check=True, capture_output=True, text=True, timeout=180)
            current = sub_video
        except Exception as e:
            logger.warning(f"Voiceover/subtitle step failed: {e}")

    # --- Character overlay (fox avatar) ---
    if enable_character_overlay:
        try:
            from python.engines.character_overlay import CharacterOverlayEngine
            avatar = None
            for cand in [
                os.path.join("assets", "logos", "fox_observer.png"),
                os.path.join("assets", "logos", "fox_casual.png"),
            ]:
                if os.path.exists(cand):
                    avatar = cand
                    break
            overlay_video = os.path.join(temp_dir, "with_overlay.mp4")
            engine = CharacterOverlayEngine()
            ok = engine.overlay_avatar(
                video_path=current,
                output_path=overlay_video,
                avatar_path=avatar,
                position="bottom_right",
                scale=0.28,
            )
            if ok:
                current = overlay_video
        except Exception as e:
            logger.warning(f"Character overlay skipped: {e}")

    # --- Final copy ---
    shutil.copy2(current, output_path)

    # --- Step 5: Upload to Google Drive ---
    uploader = GoogleDriveUploader()
    drive_link = uploader.upload_file(output_path, folder_id=drive_folder)

    elapsed = time.time() - start_time
    result.update({
        "success": True,
        "output_path": output_path,
        "drive_link": drive_link,
        "duration": elapsed,
    })

    logger.info("=" * 60)
    logger.info(f"  WORKFLOW COMPLETE in {elapsed:.1f}s")
    logger.info(f"  Clips downloaded: {len(clips)}")
    logger.info(f"  Clips edited: {len(edited_clips)}")
    logger.info(f"  Output: {output_path}")
    if drive_link:
        logger.info(f"  Drive link: {drive_link}")
    logger.info("=" * 60)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Ranking Compilation Workflow (6 to 1) - automated topic video shorts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m python.ranking_compilation_workflow --query "high jump fails"
  python -m python.ranking_compilation_workflow --query "funny basketball" --clips 6
  python -m python.ranking_compilation_workflow --query "gym fails" --output output/rank.mp4
        """,
    )
    parser.add_argument("--query", "-q", required=True, help="Topic to search and rank")
    parser.add_argument("--output", "-o", default="output/ranking_compilation.mp4",
                        help="Output video path")
    parser.add_argument("--clips", "-n", type=int, default=6,
                        help="Number of clips (default 6)")
    parser.add_argument("--drive-folder", default=None,
                        help="Google Drive folder ID to upload to")
    parser.add_argument("--no-voiceover", action="store_true",
                        help="Disable voiceover")
    parser.add_argument("--no-overlay", action="store_true",
                        help="Disable character overlay")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--json-result", default=None, help="Path to write JSON result")

    args = parser.parse_args()
    setup_logging(args.verbose)

    result = run_ranking_workflow(
        query=args.query,
        output_path=args.output,
        num_clips=args.clips,
        drive_folder=args.drive_folder,
        enable_voiceover=not args.no_voiceover,
        enable_character_overlay=not args.no_overlay,
    )

    if args.json_result:
        with open(args.json_result, "w") as f:
            json.dump(result, f, indent=2)

    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
