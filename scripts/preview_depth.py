"""Convert random .npy depth images to viewable PNG files.

Usage:
    python scripts/preview_depth.py
    python scripts/preview_depth.py --data_dir data/depth_rooms --count 50
"""

import os
import glob
import random
import argparse
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

parser = argparse.ArgumentParser(description="Convert .npy depth images to PNG.")
parser.add_argument("--data_dir", type=str,
                    default=os.path.join(PROJECT_ROOT, "data", "depth_rooms"),
                    help="Directory with .npy files (searched recursively).")
parser.add_argument("--count", type=int, default=100,
                    help="Number of random images to convert.")
parser.add_argument("--output_dir", type=str, default=None,
                    help="Output directory. Defaults to <data_dir>_preview.")
args = parser.parse_args()

# Find all .npy files
all_files = sorted(glob.glob(os.path.join(args.data_dir, "**", "*.npy"), recursive=True))
if not all_files:
    print(f"[ERROR] No .npy files found in {args.data_dir}")
    exit(1)

print(f"[INFO] Found {len(all_files)} .npy files in {args.data_dir}")

# Pick random subset
count = min(args.count, len(all_files))
selected = random.sample(all_files, count)

# Output directory
output_dir = args.output_dir or (args.data_dir.rstrip("/\\") + "_preview")
os.makedirs(output_dir, exist_ok=True)

for i, fpath in enumerate(selected):
    depth = np.load(fpath)  # (72, 128), float32, values in [0, 1]
    # Convert to 8-bit grayscale (0=close/black, 255=far/white)
    img = (depth * 255).clip(0, 255).astype(np.uint8)
    # Scale up 4x for easier viewing
    img_pil = Image.fromarray(img, mode="L").resize(
        (img.shape[1] * 4, img.shape[0] * 4), Image.NEAREST
    )
    basename = os.path.splitext(os.path.basename(fpath))[0]
    out_path = os.path.join(output_dir, f"{basename}.png")
    img_pil.save(out_path)

print(f"[DONE] Saved {count} PNG images to: {output_dir}")
