"""Generate Office Life Roast video with text overlays and effects."""
import os
import subprocess

OUTPUT_DIR = r"C:\Users\admin\Documents\Default Project\project\output"
TEMP_DIR = r"C:\Users\admin\Documents\Default Project\project\temp"
FONT = r"C\:/Windows/Fonts/AGENCYB.TTF"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)


def run_ffmpeg(args, desc="ffmpeg"):
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + args
    print(f"  [RUN] {desc}...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"  [ERROR] {result.stderr[:300]}")
        return False
    print(f"  [DONE] {desc}")
    return True


print("=" * 60)
print("  GENERATING OFFICE LIFE ROAST VIDEO")
print("=" * 60)

scenes = [
    ("0x1a1a2e", "OFFICE LIFE", "A ROAST", 3.0, "none"),
    ("0x16213e", "WHEN BOSS SAYS", "'WE ARE A FAMILY'", 3.0, "bw"),
    ("0x0f3460", "BUT SALARY SAYS", "'WE ARE STRANGERS'", 3.0, "none"),
    ("0x1a1a2e", "9 AM MEETING", "THAT COULD BE AN EMAIL", 3.5, "shake"),
    ("0x16213e", "YOUR BOSS", "ANY QUESTIONS?", 2.5, "none"),
    ("0x0f3460", "EVERYONE", "SILENCE", 2.0, "bw"),
    ("0x1a1a2e", "THE INTERN", "DOING EVERYONES WORK", 3.0, "none"),
    ("0x16213e", "THE MANAGER", "TAKING ALL THE CREDIT", 3.0, "bw"),
    ("0x0f3460", "EXCEL SHEET", "47 TABS OPEN", 2.5, "none"),
    ("0x1a1a2e", "TEAM BUILDING", "NOBODY WANTED", 3.0, "shake"),
    ("0x16213e", "THE PRINTER", "ALWAYS ON BREAK", 2.5, "bw"),
    ("0x0f3460", "FRIDAY DEADLINE", "MONDAY PANIC", 3.0, "none"),
    ("0x1a1a2e", "WFH SCHEDULE", "COME TO OFFICE", 3.0, "bw"),
    ("0x16213e", "LUNCH BREAK", "5 MINUTES MAX", 2.5, "none"),
    ("0x0f3460", "THE OFFICE CHAIR", "TRUE COMFORT", 2.5, "shake"),
    ("0x1a1a2e", "THE COFFEE MACHINE", "TRUE MVP", 3.0, "none"),
    ("0x16213e", "YOUR BOSS", "LETS DISCUSS", 2.5, "bw"),
    ("0x0f3460", "AT 5:59 PM", "ON FRIDAY", 2.5, "none"),
    ("0x1a1a2e", "OFFICE DRAMA", "BETTER THAN NETFLIX", 3.0, "shake"),
    ("0x16213e", "THE END", "OR IS IT?", 2.0, "bw"),
    ("0x0f3460", "SUBSCRIBE", "FOR MORE ROASTS", 3.0, "none"),
]

# Step 1: Generate scene clips
print("\n[1/4] Generating scenes...")
scene_files = []

for i, (bg, line1, line2, dur, effect) in enumerate(scenes):
    scene_path = os.path.join(TEMP_DIR, f"scene_{i:03d}.mp4")
    t1 = line1.replace("'", "'\\''")
    t2 = line2.replace("'", "'\\''")

    vf_parts = [
        f"color=c=0x{bg}:s=720x1280:d={dur}:r=30",
        f"drawtext=fontfile='{FONT}':text='{t1}':fontsize=56:fontcolor=yellow:x=(w-tw)/2:y=(h/2)-60:borderw=3:bordercolor=black:enable='between(t\\,0.3\\,{dur})':alpha='min(1\\,(t-0.3)/0.3)'",
        f"drawtext=fontfile='{FONT}':text='{t2}':fontsize=44:fontcolor=white:x=(w-tw)/2:y=(h/2)+20:borderw=2:bordercolor=black:enable='between(t\\,0.6\\,{dur})':alpha='min(1\\,(t-0.6)/0.3)'",
    ]

    if effect == "bw":
        vf_parts.append(f"hue=s=0:enable='between(t\\,0.5\\,{dur-0.5})'")
    elif effect == "shake":
        vf_parts.append(f"crop=iw*0.95:ih*0.95:(iw-iw*0.95)/2+sin(t*20)*5:(ih-ih*0.95)/2+cos(t*15)*3")

    vf = ",".join(vf_parts)
    args = [
        "-f", "lavfi", "-i", vf,
        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-t", str(dur),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac", "-b:a", "128k",
        "-pix_fmt", "yuv420p", "-shortest", scene_path
    ]
    if run_ffmpeg(args, f"Scene {i+1}/{len(scenes)} - {line1[:20]}"):
        scene_files.append(scene_path)

print(f"  Generated {len(scene_files)} scenes")

# Step 2: Concatenate
print("\n[2/4] Concatenating...")
concat_file = os.path.join(TEMP_DIR, "roast_concat.txt")
with open(concat_file, "w") as f:
    for sf in scene_files:
        f.write(f"file '{sf}'\n")

concat_video = os.path.join(TEMP_DIR, "roast_concat.mp4")
args = ["-f", "concat", "-safe", "0", "-i", concat_file,
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac", "-b:a", "128k", "-pix_fmt", "yuv420p",
        concat_video]
run_ffmpeg(args, "Concat")

# Step 3: Audio enhance
print("\n[3/4] Audio enhance...")
audio_video = os.path.join(TEMP_DIR, "roast_audio.mp4")
args = ["-i", concat_video,
        "-af", "loudnorm=I=-14:TP=-1:LRA=11",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
        audio_video]
run_ffmpeg(args, "Audio")

# Step 4: Final
print("\n[4/4] Final export...")
final = os.path.join(OUTPUT_DIR, "office_life_roast.mp4")
args = ["-i", audio_video,
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-c:a", "aac", "-b:a", "128k", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", final]
run_ffmpeg(args, "Final")

if os.path.exists(final):
    size = os.path.getsize(final) / (1024 * 1024)
    cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", final]
    dur = float(subprocess.run(cmd, capture_output=True, text=True).stdout.strip())
    print("\n" + "=" * 60)
    print("  OFFICE LIFE ROAST VIDEO READY!")
    print("=" * 60)
    print(f"  Output: {final}")
    print(f"  Size: {size:.2f} MB")
    print(f"  Duration: {dur:.1f}s")
    print(f"  Resolution: 720x1280 (9:16)")
    print(f"  Scenes: {len(scenes)}")
    print("=" * 60)
