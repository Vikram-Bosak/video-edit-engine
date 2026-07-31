"""
Crop Fox Character Panels.

Crops the 5 character panels from the user's uploaded image and saves them
as assets/logos/fox_X.png files.
"""

from __future__ import annotations

import os
from PIL import Image

def main():
    img_path = r"C:\Users\admin\.gemini\antigravity-ide\brain\62f92936-2e06-4fa8-b612-51608b008f32\media__1785318161772.jpg"
    output_dir = r"C:\Users\admin\.gemini\antigravity-ide\scratch\video-edit-engine\assets\logos"
    os.makedirs(output_dir, exist_ok=True)
    
    img = Image.open(img_path)
    w, h = img.size
    print(f"Loaded image size: {w}x{h}")
    
    # 2x3 Grid definition:
    # Row heights: 1/3 of H each.
    # Column widths: 1/2 of W each.
    
    # Crop definitions (left, upper, right, lower)
    crops = {
        "fox_observer.png": (0, 0, int(w/2), int(h/3)),
        "fox_thinker.png": (int(w/2), 0, w, int(h/3)),
        "fox_casual.png": (0, int(h/3), int(w/2), h),
        "fox_inquisitive.png": (int(w/2), int(h/3), w, int(2*h/3)),
        "fox_tense.png": (int(w/2), int(2*h/3), w, h)
    }
    
    for name, bbox in crops.items():
        cropped = img.crop(bbox)
        dest_path = os.path.join(output_dir, name)
        cropped.save(dest_path)
        print(f"Saved: {dest_path} {cropped.size}")

if __name__ == "__main__":
    main()
