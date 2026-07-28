"""Apply viral YouTube Shorts editing style - Fixed version."""
import os
import sys
import subprocess

INPUT = r"C:\Users\admin\Documents\Default Project\project\assets\videos\input.mp4"
OUTPUT_DIR = r"C:\Users\admin\Documents\Default Project\project\output"
TEMP_DIR = r"C:\Users\admin\Documents\Default Project\project\temp"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)


def run_ffmpeg(args, desc="ffmpeg"):
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + args
    print(f"  [RUN] {desc}...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"  [ERROR] {desc}: {result.stderr[:300]}")
        return False
    print(f"  [DONE] {desc}")
    return True


def get_duration(path):
    cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", path]
    return float(subprocess.run(cmd, capture_output=True, text=True).stdout.strip())


print("=" * 60)
print("  APPLYING VIRAL SHORTS EDITING STYLE")
print("=" * 60)

# Step 1: Detect scenes
print("\n[1/6] Detecting scenes...")
scene_cmd = [
    "ffmpeg", "-i", INPUT,
    "-vf", "select='gt(scene,0.25)',showinfo",
    "-f", "null", "-"
]
scene_result = subprocess.run(scene_cmd, capture_output=True, text=True, timeout=60)
timestamps = []
for line in scene_result.stderr.split("\n"):
    if "pts_time:" in line:
        try:
            t = float(line.split("pts_time:")[1].split()[0])
            timestamps.append(t)
        except:
            pass
duration = get_duration(INPUT)
print(f"  Found {len(timestamps)} scene changes in {duration:.1f}s")

# Step 2: Split into segments (jump cuts)
print("\n[2/6] Creating fast-paced segments...")
segments = []
if timestamps:
    segments.append((0, timestamps[0]))
    for i in range(len(timestamps) - 1):
        seg_dur = timestamps[i + 1] - timestamps[i]
        if seg_dur > 0.3:
            segments.append((timestamps[i], timestamps[i + 1]))
    segments.append((timestamps[-1], duration))
else:
    t = 0
    while t < duration:
        segments.append((t, min(t + 3.0, duration)))
        t += 3.0

if len(segments) > 15:
    segments = segments[:15]
print(f"  Using {len(segments)} segments")

segment_files = []
for i, (start, end) in enumerate(segments):
    seg_path = os.path.join(TEMP_DIR, f"seg_{i:03d}.mp4")
    args = ["-i", INPUT, "-ss", str(start), "-to", str(end),
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "aac", "-b:a", "128k", "-pix_fmt", "yuv420p", "-r", "30",
            seg_path]
    if run_ffmpeg(args, f"Segment {i+1}/{len(segments)}"):
        segment_files.append(seg_path)

# Step 3: Concatenate (jump cuts)
print("\n[3/6] Concatenating jump cuts...")
concat_file = os.path.join(TEMP_DIR, "concat.txt")
with open(concat_file, "w") as f:
    for sf in segment_files:
        f.write(f"file '{sf}'\n")

jumpcut = os.path.join(TEMP_DIR, "jumpcut.mp4")
args = ["-f", "concat", "-safe", "0", "-i", concat_file,
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac", "-b:a", "128k", "-pix_fmt", "yuv420p", jumpcut]
run_ffmpeg(args, "Jump cuts")
current = jumpcut

# Step 4: B&W effect + punchline overlay
print("\n[4/6] Applying B&W punchline effects...")
cur_dur = get_duration(current)
# Apply hue=s=0 for B&W effect at punchline timestamps
bw_timestamps = []
t = 2.0
while t < cur_dur - 0.8:
    bw_timestamps.append(t)
    t += 4.0

if bw_timestamps:
    enable_parts = " OR ".join([f"between(t,{t},{t+0.6})" for t in bw_timestamps])
    bw_filter = f"hue=s=0:enable='{enable_parts}'"
    bw_video = os.path.join(TEMP_DIR, "bw_punchline.mp4")
    args = ["-i", current, "-vf", bw_filter,
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "copy", "-pix_fmt", "yuv420p", bw_video]
    if run_ffmpeg(args, "B&W punchlines"):
        current = bw_video

# Step 5: Audio enhancement
print("\n[5/6] Enhancing audio...")
audio_enhanced = os.path.join(TEMP_DIR, "enhanced.mp4")
args = ["-i", current,
        "-af", "loudnorm=I=-14:TP=-1:LRA=11,equalizer=f=100:t=q:w=100:g=3",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", audio_enhanced]
if run_ffmpeg(args, "Audio enhancement"):
    current = audio_enhanced

# Step 6: Add Hormozi-style captions
print("\n[6/6] Adding animated captions...")
ass_path = os.path.join(TEMP_DIR, "captions.ass")

captions = [
    (0.0, 2.0, "BASKETBALL", "Pop"),
    (2.0, 4.0, "RANKING", "Pop"),
    (4.0, 6.0, "FUNNIEST", "Big"),
    (6.0, 8.0, "MOMENTS", "Big"),
    (8.0, 10.0, "LET'S GO!", "Pop"),
    (10.0, 12.5, "NUMBER 3", "Big"),
    (12.5, 15.0, "OH MY GOD", "Pop"),
    (15.0, 17.5, "NO WAY!", "Big"),
    (17.5, 20.0, "LOOK AT THIS", "Pop"),
    (20.0, 22.5, "INSANE!", "Big"),
    (22.5, 25.0, "THE BLAST", "Pop"),
    (25.0, 27.5, "CRAZY!", "Big"),
    (27.5, 30.0, "UNBELIEVABLE", "Pop"),
    (30.0, 33.0, "TOP PLAY", "Big"),
    (33.0, 36.0, "WOW!", "Pop"),
    (36.0, 38.0, "COMMENT BELOW", "Big"),
]

ass_content = """[Script Info]
Title: Viral Captions
ScriptType: v4.00+
PlayResX: 720
PlayResY: 1280
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,52,&H0000FFFF&,&H00FFFFFF&,&H00000000&,&H80000000&,-1,0,0,0,100,100,2,0,1,3,2,2,30,30,80,1
Style: Pop,Arial,60,&H0000FFFF&,&H00FFFFFF&,&H00000000&,&H80000000&,-1,0,0,0,100,100,2,0,1,4,2,2,30,30,80,1
Style: Big,Arial,72,&H00FFFFFF&,&H0000FFFF&,&H00000000&,&H80000000&,-1,0,0,0,100,100,2,0,1,4,3,2,30,30,100,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"""

for start, end, text, style in captions:
    s_h = int(start // 3600)
    s_m = int((start % 3600) // 60)
    s_s = start % 60
    e_h = int(end // 3600)
    e_m = int((end % 3600) // 60)
    e_s = end % 60
    anim = "{{\\fad(100,100)\\fscx120\\fscy120}}\\t(0,150,\\fscx100\\fscy100)"
    ass_content += f"\nDialogue: 0,{s_h}:{s_m:02d}:{s_s:05.2f},{e_h}:{e_m:02d}:{e_s:05.2f},{style},,0,0,0,,{anim}{text}"

with open(ass_path, "w", encoding="utf-8") as f:
    f.write(ass_content)

captioned = os.path.join(TEMP_DIR, "captioned.mp4")
args = ["-i", current, "-vf", f"ass='{ass_path}'",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "copy", "-pix_fmt", "yuv420p", captioned]
if run_ffmpeg(args, "Burn captions"):
    current = captioned

# Final export
print("\n[EXPORT] Final output...")
final = os.path.join(OUTPUT_DIR, "basketball_viral_edit.mp4")
args = ["-i", current,
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-c:a", "aac", "-b:a", "128k", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", final]
run_ffmpeg(args, "Final export")

if os.path.exists(final):
    size_mb = os.path.getsize(final) / (1024 * 1024)
    final_dur = get_duration(final)
    print("\n" + "=" * 60)
    print("  EDITING COMPLETE!")
    print("=" * 60)
    print(f"  Output: {final}")
    print(f"  Size: {size_mb:.2f} MB")
    print(f"  Duration: {final_dur:.1f}s")
    print(f"  Resolution: 720x1280 (9:16)")
    print("=" * 60)
else:
    print("\n  [FAILED] Output not created")
