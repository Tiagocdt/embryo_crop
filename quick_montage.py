#!/usr/bin/env python3
"""One timepoint, every well, one plate montage — fast.

For the "which wells look right?" question you need answered in minutes, not the
full time course. Detects each embryo at ONE timepoint, crops, and lays the
wells out on the plate grid with labels.

    python3 quick_montage.py RAW_DIR OUT.png [--tp middle] [--channel CO6]

Deliberately separate from process.py: it writes no crops, no metadata and no
database, so it cannot disturb a real run or be mistaken for one.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from acquifer import find_image_dirs, index_folder, index_folders   # noqa: E402
from process import EGG_MM, crop_at, detect_center, resize          # noqa: E402


def stretch(a, lo_pct=1.0, hi_pct=99.0):
    a = a.astype(np.float32)
    lo, hi = np.percentile(a, lo_pct), np.percentile(a, hi_pct)
    return np.clip((a - lo) * 255.0 / max(1.0, hi - lo), 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_dir")
    ap.add_argument("out_png")
    ap.add_argument("--tp", default="middle", help="timepoint number, or 'middle'")
    ap.add_argument("--channel", default=None, help="default: the detect channel")
    ap.add_argument("--thumb", type=int, default=200)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    dirs = find_image_dirs(a.raw_dir)
    idx = index_folders(dirs) if len(dirs) > 1 else index_folder(a.raw_dir)
    tp = (idx.timepoints[len(idx.timepoints) // 2] if a.tp == "middle"
          else int(a.tp))
    ch = a.channel or idx.detect_channel
    sl = idx.detect_slice
    crop_px = idx.crop_px() or 576
    print(f"{len(idx.positions)} wells | {len(idx.timepoints)} timepoints "
          f"| LO{tp:03d} {sl} | showing {ch}, detecting on {idx.detect_channel} "
          f"| crop {crop_px}px @ {idx.um_per_px} um/px")

    from concurrent.futures import ThreadPoolExecutor

    def one(pos):
        # ALWAYS detect on the brightfield/detect channel -- fluorescence shows
        # the injection, not the embryo outline, so detecting on it would centre
        # on the bolus (or on nothing, in an uninjected well). Then crop the
        # requested channel at that centre.
        kd = (pos, tp, idx.detect_channel, sl)
        ks = (pos, tp, ch, sl)
        if kd not in idx.frames or ks not in idx.frames:
            return pos, None
        try:
            det_img = tifffile.imread(idx.path(kd))
            show_img = det_img if ch == idx.detect_channel else tifffile.imread(idx.path(ks))
        except Exception:                                   # noqa: BLE001
            return pos, None
        cy, cx, conf = detect_center(det_img, um_per_px=idx.um_per_px,
                                     egg_mm=EGG_MM, with_confidence=True)
        c, _f, _v = crop_at(show_img, cy, cx, crop_px)
        return pos, (resize(c, a.thumb), conf)

    tiles = {}
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for pos, r in ex.map(one, idx.positions):
            if r is not None:
                tiles[pos] = r

    rows = sorted({p[0] for p in tiles})
    cols = sorted({int(p[1:]) for p in tiles})
    T = a.thumb
    pad, top = 2, 18
    W = len(cols) * (T + pad) + 40
    H = len(rows) * (T + pad) + top + 10

    from PIL import Image, ImageDraw
    sheet = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(sheet)
    for j, c in enumerate(cols):
        d.text((40 + j * (T + pad) + T // 2 - 6, 4), f"{c:02d}", fill=0)
    doubtful = []
    for i, r in enumerate(rows):
        d.text((6, top + i * (T + pad) + T // 2), r, fill=0)
        for j, c in enumerate(cols):
            pos = f"{r}{c:02d}"
            if pos not in tiles:
                continue
            img, conf = tiles[pos]
            sheet.paste(Image.fromarray(stretch(img)),
                        (40 + j * (T + pad), top + i * (T + pad)))
            if conf["prominence"] < 2.0 or conf["on_kernel_edge"]:
                doubtful.append(pos)
                # ring the tile so a doubtful detection is visible at a glance
                x0, y0 = 40 + j * (T + pad), top + i * (T + pad)
                d.rectangle([x0, y0, x0 + T - 1, y0 + T - 1], outline=0, width=3)
    sheet.save(a.out_png)
    print(f"wrote {a.out_png}  ({len(tiles)} wells)")
    if doubtful:
        print(f"detection doubtful (ringed): {', '.join(doubtful)}")


if __name__ == "__main__":
    main()
