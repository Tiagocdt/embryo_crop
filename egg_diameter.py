#!/usr/bin/env python3
"""Measure the chorion diameter of a cropped embryo.

    python3 egg_diameter.py CROP.tif --um-per-px 3.25
    python3 egg_diameter.py --validate            # against annotated ground truth

METHOD
------
The chorion is a single strong circular edge, and it is the only structure in
the well at that scale. So: for every candidate radius, average the RADIAL
component of the image gradient around the circle of that radius, and take the
radius where that is largest. A circular edge lights up its own radius and
nothing else does — a radial accumulator, not a threshold.

Two things make it robust rather than clever:

* **The search is bounded by biology.** Only radii corresponding to a plausible
  egg (default 1.1-1.9 mm diameter) are considered, so yolk boundaries, the
  well wall and debris cannot win by being bigger or smaller than an egg.
* **The gradient is projected onto the radial direction.** A chorion edge is
  perpendicular to the radius everywhere; texture inside the egg is not, so it
  contributes little even where it is strong.

The centre is refined before measuring, because a centre a few pixels off
smears the accumulator and biases the radius low.
"""
from __future__ import annotations

import argparse
import os

import numpy as np


def _radial_accumulator(img, cy, cx, r_lo, r_hi):
    """Mean inward-projected gradient on each circle from r_lo to r_hi."""
    a = img.astype(np.float32)
    gy, gx = np.gradient(a)
    h, w = a.shape
    yy, xx = np.mgrid[0:h, 0:w]
    dy, dx = yy - cy, xx - cx
    r = np.sqrt(dy * dy + dx * dx)
    with np.errstate(invalid="ignore", divide="ignore"):
        uy, ux = dy / r, dx / r                    # outward unit vector
    uy, ux = np.nan_to_num(uy), np.nan_to_num(ux)
    radial = np.abs(gy * uy + gx * ux)             # gradient along the radius
    rb = r.astype(int)
    n = min(r_hi + 2, rb.max() + 1)
    tot = np.bincount(rb.ravel(), radial.ravel(), minlength=n)[:n]
    cnt = np.bincount(rb.ravel(), minlength=n)[:n]
    prof = np.where(cnt > 0, tot / np.maximum(cnt, 1), 0.0)
    return prof


def _refine_center(img, cy, cx, r_lo, r_hi, span=8, step=2):
    """Nudge the centre to maximise the sharpest ring; an off centre smears it."""
    best, bc = -1.0, (cy, cx)
    for ddy in range(-span, span + 1, step):
        for ddx in range(-span, span + 1, step):
            p = _radial_accumulator(img, cy + ddy, cx + ddx, r_lo, r_hi)
            if len(p) <= r_lo:
                continue
            v = float(p[r_lo:r_hi].max()) if r_hi > r_lo else 0.0
            if v > best:
                best, bc = v, (cy + ddy, cx + ddx)
    return bc


def egg_diameter(img, um_per_px, min_mm=1.1, max_mm=1.9, refine=True,
                 outer_frac=0.5):
    """(diameter_um, info). Returns (None, info) if nothing plausible is found."""
    if img.ndim == 3:
        img = img[..., 0]
    h, w = img.shape
    cy, cx = h // 2, w // 2
    r_lo = int(min_mm * 1000 / um_per_px / 2)
    r_hi = int(max_mm * 1000 / um_per_px / 2)
    r_hi = min(r_hi, min(h, w) // 2 - 1)
    if r_hi <= r_lo:
        return None, {"why": "search range does not fit in the crop"}
    if refine:
        cy, cx = _refine_center(img, cy, cx, r_lo, r_hi)
    prof = _radial_accumulator(img, cy, cx, r_lo, r_hi)
    if len(prof) <= r_lo:
        return None, {"why": "profile shorter than the search range"}
    band = prof[r_lo:r_hi]
    if band.size == 0 or band.max() <= 0:
        return None, {"why": "no gradient ring in the plausible range"}
    # The chorion is the OUTERMOST strong ring, not the strongest one. Taking
    # the global maximum picks whichever edge happens to be sharpest, which is
    # often an interior boundary (yolk, blastoderm) -- that is why the first
    # version underestimated by up to 450 um on exactly the wells where the
    # inside is high-contrast. Walk in from the outside and take the last ring
    # that is still a real edge.
    thr = band.max() * outer_frac
    above = np.nonzero(band >= thr)[0]
    k = int(above[-1]) if above.size else int(np.argmax(band))
    # slide to the local peak so the sub-pixel fit is centred on the ring
    while 0 < k < band.size - 1 and band[k + 1] > band[k]:
        k += 1
    while 0 < k < band.size - 1 and band[k - 1] > band[k]:
        k -= 1
    r = r_lo + k
    # sub-pixel: parabola through the peak and its neighbours
    if 0 < k < band.size - 1:
        y0, y1, y2 = band[k - 1], band[k], band[k + 1]
        denom = (y0 - 2 * y1 + y2)
        if denom != 0:
            r = r_lo + k - 0.5 * (y2 - y0) / denom
    med = float(np.median(band))
    return 2 * r * um_per_px, {
        "radius_px": round(float(r), 2),
        "center": (int(cy), int(cx)),
        "prominence": round(float(band.max() / med), 2) if med > 0 else None,
    }


# ---------------------------------------------------------------- validation
def validate(db, root, um_per_px, limit, plates=None):
    import sqlite3, collections, glob
    import tifffile
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    q = ("SELECT plate_id, well, timepoint, length_um FROM measurement "
         "WHERE name='egg_diameter' AND length_um IS NOT NULL")
    rows = list(con.execute(q))
    folder = {}
    for d in sorted(os.listdir(root)):
        if os.path.isdir(os.path.join(root, d)):
            folder[d.split("_", 1)[1] if d[:8].isdigit() else d] = d
    seen, out = set(), []
    for plate, well, tp, truth in rows:
        if plates and not any(p in plate for p in plates):
            continue
        if (plate, well, tp) in seen:
            continue
        seen.add((plate, well, tp))
        d = folder.get(plate)
        if not d:
            continue
        hits = glob.glob(os.path.join(root, d, "CO*", well, "SL*",
                                      f"*LO{int(tp):03d}_*.tif"))
        if not hits:
            continue
        try:
            img = tifffile.imread(sorted(hits)[len(hits) // 2])
        except Exception:                                     # noqa: BLE001
            continue
        est, info = egg_diameter(img, um_per_px)
        if est is None:
            continue
        out.append((plate, well, tp, truth, est, est - truth))
        if len(out) >= limit:
            break
    if not out:
        print("no comparable annotations found")
        return
    diffs = np.array([o[5] for o in out])
    truths = np.array([o[3] for o in out])
    print(f"compared {len(out)} annotated wells")
    print(f"  annotation : mean {truths.mean():.0f} um   SD {truths.std():.0f}   "
          f"range {truths.min():.0f}..{truths.max():.0f}")
    print(f"  difference : mean {diffs.mean():+.1f} um (bias)   SD {diffs.std():.1f}")
    print(f"               median |diff| {np.median(np.abs(diffs)):.1f} um   "
          f"90th pct {np.percentile(np.abs(diffs),90):.1f} um")
    within = lambda t: 100 * (np.abs(diffs) <= t).mean()
    for t in (25, 50, 75, 100):
        print(f"    within {t:>3} um: {within(t):5.1f}%")
    worst = sorted(out, key=lambda o: -abs(o[5]))[:5]
    print("  worst:")
    for p, w, tp, t, e, d in worst:
        print(f"    {p[:26]:<26} {w} tp{tp}  you {t:.0f}  auto {e:.0f}  ({d:+.0f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("crop", nargs="?")
    ap.add_argument("--um-per-px", type=float, default=3.25)
    ap.add_argument("--min-mm", type=float, default=1.1)
    ap.add_argument("--max-mm", type=float, default=1.9)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--db", default="/Users/tiago/metameda/imaging/data/medaka.db")
    ap.add_argument("--root", default="/Volumes/aulehla/Tiago/AQ-EMBL/PROCESSED_v2")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--plates", default="")
    a = ap.parse_args()
    if a.validate:
        validate(a.db, a.root, a.um_per_px, a.limit,
                 [p for p in a.plates.split(",") if p] or None)
    else:
        import tifffile
        d, info = egg_diameter(tifffile.imread(a.crop), a.um_per_px,
                               a.min_mm, a.max_mm)
        print(f"{d:.0f} um" if d else "not found", info)
