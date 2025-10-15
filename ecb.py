#!/usr/bin/env python3
# bmp_from_ecb_ciphertext.py
import os, math, hashlib
from collections import defaultdict, Counter
from pathlib import Path
from PIL import Image, ImageOps, ImageEnhance 
import numpy as np 

IN = Path("aes.bmp.enc")
OUT_DIR = Path("recon_images")
OUT_DIR.mkdir(exist_ok=True)

data = IN.read_bytes()
N = len(data)
BLOCK = 16

# --- 1) split into 16-byte blocks and detect repeated-block distances
blocks = [data[i:i+BLOCK] for i in range(0, N, BLOCK)]
pos = defaultdict(list)
for i,b in enumerate(blocks):
    pos[b].append(i)

diffs = []
for b,poses in pos.items():
    if len(poses) > 1:
        # limit to first 50 occurrences to cap combinatorics
        for i in range(min(50,len(poses))):
            for j in range(i+1, min(50,len(poses))):
                diffs.append(poses[j]-poses[i])
cnt = Counter(diffs)
most_common = cnt.most_common(40)
g = 0
if most_common:
    top_diffs = [d for d,_ in most_common[:10]]
    g = 0
    for d in top_diffs:
        g = math.gcd(g,d)
print("File bytes:", N, "blocks:", len(blocks))
print("Most common block-distance diffs (top 10):", most_common[:10])
print("Estimated blocks_per_row (GCD):", g if g>0 else "unknown")

# convert to suggested bytes-per-row and candidate widths for bpp 24/32
candidates = []

if g and g>0:
    bytes_per_row = g * BLOCK
    # try typical bit depths 24 and 32
    if bytes_per_row % 3 == 0:
        candidates.append(("from_gcd_24bpp", bytes_per_row//3, 24))
    if bytes_per_row % 4 == 0:
        candidates.append(("from_gcd_32bpp", bytes_per_row//4, 32))
    candidates.append(("from_gcd_bytes", bytes_per_row, None))

# also add common widths to try (24/32 bpp)
common_widths = [320, 360, 480, 512, 640, 800, 1024, 1280]
for w in common_widths:
    candidates.append((f"common_{w}_24", w, 24))
    candidates.append((f"common_{w}_32", w, 32))

# helper to create image from raw bytes using BMP rules
def try_assemble(width, bpp):
    bpp = int(bpp)
    bytes_per_pixel = bpp // 8
    row_bytes = width * bytes_per_pixel
    row_bytes_padded = ((row_bytes + 3) // 4) * 4
    total_pixels_area_bytes = (N // row_bytes_padded) * row_bytes_padded  # floor
    # if we can fit at least one row
    if row_bytes_padded == 0 or total_pixels_area_bytes == 0:
        return None
    height = total_pixels_area_bytes // row_bytes_padded
    # if height too small, skip
    if height < 10:
        return None

    # prepare bytearray sized exactly height * row_bytes_padded
    needed = height * row_bytes_padded
    ba = bytearray(data[:needed])
    # If not enough bytes, pad (shouldn't happen due to floor above)
    if len(ba) < needed:
        ba += b'\x00' * (needed - len(ba))

    # Compose image pixel rows (remember BMP stores bottom-to-top)
    rows = []
    for r in range(height):
        off = r * row_bytes_padded
        rowdata = ba[off:off+row_bytes]  # drop padding bytes at end of row
        rows.append(rowdata)

    # rows currently top-to-bottom in our slicing (because ciphertext laid out linearly),
    # but BMP's pixel array is bottom-to-top. We'll reverse to show correctly in normal image viewers:
    rows = rows[::-1]

    # build RGB array (PIL expects top-to-bottom)
    # For 24bpp: each pixel is B,G,R
    img_arr = np.zeros((height, width, 3), dtype=np.uint8)
    for y, row in enumerate(rows):
        if bytes_per_pixel == 3:
            # BGR -> convert to RGB
            row_pixels = np.frombuffer(row, dtype=np.uint8).reshape((width,3))
            # swap B and R
            img_arr[y, :, :] = row_pixels[:, ::-1]
        elif bytes_per_pixel == 4:
            # BGRA -> take B,G,R ignore A or place alpha as needed
            row_pixels = np.frombuffer(row, dtype=np.uint8).reshape((width,4))
            img_arr[y, :, :] = row_pixels[:, :3][:, ::-1]  # BGR->RGB (drop A)
        else:
            return None

    img = Image.fromarray(img_arr, mode="RGB")
    return img

# try candidates
saved = []
seen = set()
for tag, val, bpp in candidates:
    if bpp is None:
        continue
    width = int(val)
    key = (width,bpp)
    if key in seen:
        continue
    seen.add(key)
    img = try_assemble(width, bpp)
    if img:
        fn = OUT_DIR / f"recon_{tag}_w{width}_bpp{bpp}.png"
        # also save a rotated variant and an enhanced variant
        img.save(fn)
        ImageOps.exif_transpose(img).rotate(90, expand=True).save(OUT_DIR / f"rot_{fn.name}")
        enh = ImageEnhance.Contrast(img).enhance(1.8)
        enh.save(OUT_DIR / f"enh_{fn.name}")
        saved.append(fn)
        print("Saved:", fn, "width", width, "bpp", bpp, "height", img.size[1])

# Also make a block-level color map (one pixel per AES block)
unique_map = {}
cols = []
for b in blocks:
    if b in unique_map:
        cols.append(unique_map[b])
    else:
        h = hashlib.sha1(b).digest()
        col = (h[0], h[1], h[2])
        unique_map[b] = col
        cols.append(col)

blocks_per_row_guess = g if g>0 else 120  # fallback guess
bpr = blocks_per_row_guess
bp_rows = math.ceil(len(cols) / bpr)
canvas = Image.new("RGB", (bpr, bp_rows))
px = canvas.load()
for i,col in enumerate(cols):
    r = i // bpr
    c = i % bpr
    px[c,r] = col
canvas = canvas.resize((bpr*6, bp_rows*6), Image.NEAREST)
canvas.save(OUT_DIR / f"block_colormap_bpr{bpr}.png")
print("Block colormap saved; try vis in", OUT_DIR)

print("Done. Inspect images in the recon_images folder.")
