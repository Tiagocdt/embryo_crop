#!/usr/bin/env python3
"""Everything that happens to an image. There are seven steps and no more.

    1. parse filename      -> well, timepoint, channel, z-slice     (acquifer.py)
    2. pick detect slice   -> middle of the z-stack                 (acquifer.py)
    3. detect the embryo   -> (cy, cx), a few times per well
    4. crop                -> a fixed PHYSICAL field of view
    5. resize              -> always the same output pixel size
    6. rescale intensity   -> 16-bit counts to 8-bit, by a chosen map
    7. write               -> your folder structure, your filename

Steps 3-7 are this file. Nothing else happens to a pixel. In particular:
no sharpening, no denoising, no per-frame auto-contrast unless you explicitly
ask for scaling mode "image", no gamma.

INTENSITY SCALING -- the only real choice in here
-------------------------------------------------
    raw16   no rescale at all, uint16 counts preserved. The only fully
            quantitative option. Comparable across everything, forever.
    fixed   one constant LO:HI you supply. Comparable across plates.
    plate   percentiles over the WHOLE plate, one map per channel.  <- default
            Comparable across wells and timepoints within the plate.
    well    percentiles over one well's whole movie ("per trajectory").
            Comparable across timepoints, NOT across wells.
    image   percentiles of that single frame. Auto-contrast. Comparable with
            NOTHING. Every frame is stretched to look good individually.

Wider scope = more comparable, but flatter individual images, because one map
must cover the brightest frame on the plate. Percentiles (not min-max) are the
default because a single hot pixel sets a min-max range.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tifffile

from acquifer import DEFAULT_FOV_MM, DEFAULT_OUTPUT_PX, index_folder

SCALING_MODES = ("plate", "well", "image", "fixed", "raw16")
HIST_BINS = 65536

# An external drive that drops off the bus throws OSError mid-run. Retry a few
# times with a backoff so a blip costs seconds, not the whole plate. If it
# stays gone the frame is counted as failed and the run carries on -- rerunning
# skips everything already written, so a plate resumes where it stopped.
IO_ATTEMPTS = 4
IO_BACKOFF = 1.5

# A frame can be unreadable for two different reasons and they need different
# treatment. A vanished mount raises OSError and IS worth retrying. A file the
# microscope failed to write is zero bytes forever -- tifffile raises
# TiffFileError("not a TIFF file: header=b''"), retrying is pointless, and
# crucially TiffFileError is NOT an OSError, so it escapes the retry guard and
# would abort the whole plate. Roughly 1% of frames on some ACQUIFER runs are
# zero-byte, so a plate MUST survive them.
UNREADABLE = (tifffile.TiffFileError, ValueError)


def read_frame(path):
    """Read one raw frame. Transient I/O is retried; a corrupt/empty file is
    raised straight away because no amount of retrying will fix it."""
    try:
        return _retry(tifffile.imread, path)
    except UNREADABLE:
        raise
    except OSError:
        raise


def _retry(fn, *a, **kw):
    last = None
    for i in range(IO_ATTEMPTS):
        try:
            return fn(*a, **kw)
        except (OSError, ValueError) as e:      # includes a vanished mount
            last = e
            if i < IO_ATTEMPTS - 1:
                time.sleep(IO_BACKOFF * (i + 1))
    raise last


# ---------------------------------------------------------------------------
# 3. detection -- no ML, no model file, no GPU
# ---------------------------------------------------------------------------
def _box(a, r):
    """Mean filter of radius r via an integral image. Zero-padded on purpose:
    a window hanging off the edge sums less, which stops the well wall and the
    vignette (both high-gradient) from winning the argmax."""
    if r < 1:
        return a
    n, m = a.shape
    k = 2 * r + 1
    p = np.pad(a, r, mode="constant", constant_values=0.0)
    ii = np.zeros((p.shape[0] + 1, p.shape[1] + 1), dtype=np.float64)
    ii[1:, 1:] = p.cumsum(0).cumsum(1)
    return (ii[k:k + n, k:k + m] - ii[0:n, k:k + m]
            - ii[k:k + n, 0:m] + ii[0:n, 0:m]) / (k * k)


# How big the specimen is, in millimetres. This is a PHYSICAL size, so the
# smoothing radius is derived from it and um/px -- see detect_center().
#
# MEASURED, not guessed: the chorion of a medaka egg came out at 1.69 mm on
# 1024px binned EMBL frames and 1.61 mm on 2048px unbinned Wittbrodt frames
# (radial gradient profile taken about the reference centres in both).
#
# It does not need to be exact. Sweeping it against 40 reference centres, every
# value from 1.5 to 1.8 mm placed all wells within half an egg-radius; the
# default sits in the middle of that plateau rather than at its edge:
#
#     egg_mm   1.00  1.25  1.50  1.65  1.80  2.00  2.50
#     median    130    69    30    26    28    46    68   px error
#     >57px      40    26     0     0     0    10    23   wells
#
# For a different species, set --egg-mm instead of editing this.
EGG_MM = 1.65


def detect_center(img, um_per_px=None, downsample=4, egg_mm=EGG_MM,
                  egg_frac=None):
    """(cy, cx) of the embryo, in full-resolution pixels.

    The egg is the only textured object in an otherwise flat well, so its
    centre is the peak of gradient energy smoothed over about an egg radius.

    THE RADIUS MUST COME FROM um/px, NOT FROM A FRACTION OF THE FRAME. An egg
    is ~1.5 mm however you image it; at 3.25 um/px that is 462 px, which is 23%
    of a 2048 px frame but 45% of a 1024 px one. A fixed fraction therefore
    smooths over half the correct radius on the smaller frame, and the argmax
    settles on a sub-feature instead of the whole egg -- centres drift toward
    one edge and the crop runs off the frame. `egg_frac` is kept only as a
    manual override for data whose pixel size is unknown.

    Validated against EmbryoNet on all 96 wells of Wittbrodt V01: median error
    9 px, worst 28 px on a 2048 px frame -- see README.md.
    """
    small = img[::downsample, ::downsample].astype(np.float32)
    if egg_frac is not None:
        r = max(1, int(egg_frac * small.shape[0] / 2))
    elif um_per_px:
        r = max(1, int((egg_mm * 1000.0 / um_per_px) / 2 / downsample))
    else:
        r = max(1, int(0.23 * small.shape[0] / 2))      # last-resort guess
    gy, gx = np.gradient(small)
    smoothed = _box(np.hypot(gy, gx), r)
    cy, cx = np.unravel_index(np.argmax(smoothed), smoothed.shape)
    return int(cy * downsample), int(cx * downsample)


# ---------------------------------------------------------------------------
# 4. crop  /  5. resize
# ---------------------------------------------------------------------------
def crop_at(img, cy, cx, size, clamp=True):
    """Cut `size` x `size` centred on (cy, cx). Returns (crop, fill_fraction).

    If the window would run off the frame it is SHIFTED back inside rather than
    padded with black (`clamp`). The specimen is smaller than the crop -- an egg
    is ~519 px inside a 576 px window -- so sliding the window a few dozen
    pixels keeps the whole specimen visible, whereas padding throws away real
    image and then poisons any per-image contrast stretch with a slab of zeros.
    The field of view is never shrunk; only its position moves.

    Black padding still happens when the frame itself is smaller than the crop,
    which is the one case shifting cannot fix.
    """
    h, w = img.shape[:2]
    half = size // 2
    if clamp:
        if size <= h:
            cy = min(max(cy, half), h - (size - half))
        if size <= w:
            cx = min(max(cx, half), w - (size - half))
    y0, y1 = cy - half, cy - half + size
    x0, x1 = cx - half, cx - half + size
    out = np.zeros((size, size), dtype=img.dtype)
    sy0, sy1 = max(0, y0), min(h, y1)
    sx0, sx1 = max(0, x0), min(w, x1)
    if sy1 > sy0 and sx1 > sx0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    filled = ((sy1 - sy0) * (sx1 - sx0)) / float(size * size)
    # where the REAL image sits inside the crop. Any statistic taken over the
    # crop must be restricted to this window, or the black padding counts as
    # the darkest pixels present -- see the per-image scaling in run().
    valid = (sy0 - y0, sy1 - y0, sx0 - x0, sx1 - x0)
    return out, filled, valid


def resize(a, out_px):
    """Resample a square crop to `out_px`.

    DOWNSAMPLING AREA-AVERAGES. Point-sampling a reduction throws away every
    pixel it does not land on, so fine detail folds back as aliasing -- moire on
    the chorion, jagged edges, and noise that survives where it should have
    averaged out. This matters for higher-resolution optics: a 4x unbinned
    acquisition yields a 1152 px crop that must become 576, where a 2x-binned
    one yields 576 and never resamples at all. Same nominal pipeline, but only
    the sharper data goes through this path -- so it has to be the good one.

    Integer reductions are a block mean; anything else falls back to bilinear.
    """
    if a.shape[0] == out_px and a.shape[1] == out_px:
        return a
    n = a.shape[0]
    if n > out_px and n % out_px == 0:
        k = n // out_px
        return (a.reshape(out_px, k, out_px, k)
                 .mean(axis=(1, 3))
                 .astype(a.dtype))
    idx = np.linspace(0, n - 1, out_px)
    i0 = np.floor(idx).astype(int)
    i1 = np.minimum(i0 + 1, n - 1)
    f = (idx - i0).astype(np.float32)
    rows = (a[i0, :].astype(np.float32) * (1 - f)[:, None]
            + a[i1, :].astype(np.float32) * f[:, None])
    cols = (rows[:, i0] * (1 - f)[None, :] + rows[:, i1] * f[None, :])
    return cols.astype(a.dtype) if a.dtype == np.uint16 else cols


# ---------------------------------------------------------------------------
# 6. intensity
# ---------------------------------------------------------------------------
def accumulate(hist, img):
    hist += np.bincount(np.asarray(img, dtype=np.uint16).ravel(),
                        minlength=HIST_BINS)[:HIST_BINS]
    return hist


def percentiles_from_hist(hist, pcts):
    total = hist.sum()
    if total == 0:
        return [0.0 for _ in pcts]
    c = np.cumsum(hist)
    return [float(np.searchsorted(c, total * p / 100.0)) for p in pcts]


def apply_scale(a, mode, lo, hi):
    """16-bit counts -> the stored pixel value. Returns (array, dtype_name)."""
    if mode == "raw16":
        return np.asarray(a, dtype=np.uint16), "uint16"
    if hi <= lo:
        hi = lo + 1
    out = (np.asarray(a, dtype=np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(out, 0, 255).astype(np.uint8), "uint8"


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------
class Settings:
    """Everything the GUI can set. Saved verbatim into the output manifest."""

    def __init__(self, **kw):
        self.raw_dir = kw.get("raw_dir")
        self.out_dir = kw.get("out_dir")
        self.plate = kw.get("plate") or ""
        self.channels = kw.get("channels") or []          # [] = all
        self.slices = kw.get("slices") or []              # [] = all
        self.positions = kw.get("positions") or []        # [] = all wells
        self.fov_mm = float(kw.get("fov_mm", DEFAULT_FOV_MM))
        self.output_px = int(kw.get("output_px", DEFAULT_OUTPUT_PX))
        self.um_per_px_override = kw.get("um_per_px_override") or None
        # one mode, or a per-channel spec like "plate,CO6=image"
        self.scaling = kw.get("scaling", "plate")
        # convenience: mode for whichever channel is brightfield
        self.bf_scaling = kw.get("bf_scaling") or None
        # specimen size in mm; sets the detection smoothing radius
        self.egg_mm = float(kw.get("egg_mm") or EGG_MM)
        self.low_pct = float(kw.get("low_pct", 1.0))
        self.high_pct = float(kw.get("high_pct", 99.9))
        self.fixed_lo = kw.get("fixed_lo")
        self.fixed_hi = kw.get("fixed_hi")
        self.stats_sample = int(kw.get("stats_sample", 400))   # 0 = every frame
        self.detect_every = int(kw.get("detect_every", 0))     # 0 = fixed centre
        # Free-text plate fields. PlateNotate auto-fills its plate form from
        # these. Empty ones are OMITTED from plate_metadata.json rather than
        # stored blank, so "not a knockout experiment" stays distinguishable
        # from "somebody forgot to fill it in".
        self.line = kw.get("line", "") or ""
        self.guide = kw.get("guide", "") or ""
        self.assay = kw.get("assay", "") or ""
        self.dir_template = kw.get(
            "dir_template", "{plate}/{channel}/{pos}/{slice}")
        self.file_template = kw.get(
            "file_template", "{plate}_{pos}_LO{tp:03d}_{channel}_{slice}.tif")
        self.workers = int(kw.get("workers", 8))
        self.overwrite = bool(kw.get("overwrite", False))
        # Re-emit plate_metadata.json for an ALREADY-processed plate
        # without touching a single image. Detection and the sampled
        # calibration are deterministic, so the values match the run
        # that produced the crops.
        self.metadata_only = bool(kw.get("metadata_only", False))

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


def _sample_frames(idx, channel, n):
    """Frames spread evenly over EVERY well, timepoint AND z-slice.

    The sample must span the whole channel, not just the detection slice: a
    z-stack is not uniformly bright, so percentiles taken from one slice give a
    map that clips or flattens the others. Sorting by (well, timepoint,
    channel, slice) and then striding takes an even spread across all three.
    """
    keys = sorted(k for k in idx.frames if k[2] == channel)
    if n <= 0 or n >= len(keys):
        return keys
    step = len(keys) / float(n)
    return [keys[int(i * step)] for i in range(n)]


def compute_calibration(idx, st, channel, mode, log=print):
    """One LO/HI for `channel`, over the scope `mode` asks for."""
    if mode == "raw16":
        return {"mode": "raw16", "lo": None, "hi": None}
    if mode == "fixed":
        if st.fixed_lo is None or st.fixed_hi is None:
            raise RuntimeError("scaling 'fixed' needs --fixed-lo and --fixed-hi "
                               "(the two raw counts that map to 0 and 255)")
        return {"mode": "fixed", "lo": float(st.fixed_lo),
                "hi": float(st.fixed_hi)}
    if mode == "image":
        return {"mode": "image", "lo": None, "hi": None}
    if mode == "well":
        return {"mode": "well", "lo": None, "hi": None}

    keys = _sample_frames(idx, channel, st.stats_sample)
    hist = np.zeros(HIST_BINS, dtype=np.int64)
    t0 = time.perf_counter()

    def rd(k):
        try:
            return read_frame(os.path.join(idx.image_dir, idx.frames[k][2]))
        except (OSError,) + UNREADABLE:
            return None                      # zero-byte frame: not in the stats

    n_bad = 0
    with ThreadPoolExecutor(max_workers=st.workers) as ex:
        for img in ex.map(rd, keys):
            if img is None:
                n_bad += 1
                continue
            accumulate(hist, img)
    if n_bad:
        log(f"  {channel}: {n_bad}/{len(keys)} sampled frames unreadable "
            f"(zero-byte); excluded from the statistics")
    lo, hi = percentiles_from_hist(hist, [st.low_pct, st.high_pct])
    log(f"  {channel}: plate calibration from {len(keys)} frames in "
        f"{time.perf_counter()-t0:.1f}s -> "
        f"p{st.low_pct}={lo:.0f}  p{st.high_pct}={hi:.0f}")
    return {"mode": "plate", "lo": lo, "hi": hi,
            "low_pct": st.low_pct, "high_pct": st.high_pct,
            "n_frames_sampled": len(keys),
            "sampled_all": st.stats_sample <= 0 or len(keys) == len(idx.frames)}


def resolve_scaling(spec, channels, detect_channel, bf_scaling=None):
    """Work out the scaling mode for EVERY channel. Returns {channel: mode}.

    Brightfield and fluorescence want opposite things and usually sit in the
    same plate, so one mode for the whole plate is not enough:

        fluorescence -> `plate`, so wells and timepoints are comparable
        brightfield  -> `image`, because transmitted-light illumination varies
                        well to well and that is an instrument artefact

    `spec` is either one mode for everything ("plate"), or a comma-separated
    list where a bare word sets the default and CHANNEL=MODE overrides one
    channel:

        "image"                     every channel per-image
        "plate,CO6=image"           fluorescence by plate, CO6 per-image
        "CO6=image,CO2=raw16"       explicit, default stays "plate"

    `bf_scaling` is the convenience form: it applies to whichever channel was
    auto-detected as brightfield, so you do not have to know its number.
    """
    default = "plate"
    per = {}
    for tok in str(spec or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "=" in tok:
            ch, _, mode = tok.partition("=")
            per[ch.strip()] = mode.strip()
        else:
            default = tok
    if bf_scaling and detect_channel:
        per[detect_channel] = bf_scaling
    out = {ch: per.get(ch, default) for ch in channels}
    bad = {c: m for c, m in out.items() if m not in SCALING_MODES}
    if bad:
        raise RuntimeError(
            f"unknown scaling mode(s) {bad}; pick from {list(SCALING_MODES)}")
    return out


def run(st: Settings, log=print, progress=None, should_stop=None):
    """Process one plate. `progress(done, total, message)` drives the GUI."""
    t_start = time.perf_counter()
    log(f"indexing {st.raw_dir} ...")
    idx = index_folder(st.raw_dir)
    plate = st.plate or os.path.basename(os.path.normpath(st.raw_dir))
    um = st.um_per_px_override or idx.um_per_px
    if not um:
        raise RuntimeError("pixel size unknown; set the objective override")
    crop_px = int(round(st.fov_mm * 1000.0 / um))

    # A crop can never be larger than the frame it is cut from. If it is, the
    # pixel size is wrong -- almost always a bad PX token in the filenames --
    # and every output would be a small embryo adrift in black padding. Refuse
    # rather than write hundreds of thousands of unusable files.
    #
    # Real example: AQV02/AQV03 carry PX01625 (162.5 nm) where every other
    # plate from the same microscope carries PX16250 (1625 nm). That is a 10x
    # error, giving um/px 0.325 and a 5760 px crop on a 1024 px frame.
    if idx.frame_shape and crop_px > min(idx.frame_shape[:2]):
        raise RuntimeError(
            f"crop of {crop_px}px does not fit in a "
            f"{idx.frame_shape[1]}x{idx.frame_shape[0]} frame.\n"
            f"  um/px was derived as {um} from PX={idx.px_sensor_nm}nm x "
            f"binning {idx.binning}.\n"
            f"  That pixel size is almost certainly wrong in the filenames.\n"
            f"  Override it, e.g.  --um-per-px 3.25   (or set --fov-mm smaller "
            f"if the crop really should be this large).")
    channels = st.channels or idx.channels
    slices = st.slices or idx.slices
    positions = st.positions or idx.positions

    log(idx.summary(st.fov_mm, st.output_px, um))
    log(f"crop {crop_px} px @ {um} um/px  ->  {st.output_px}px output")
    modes = resolve_scaling(st.scaling, channels, idx.detect_channel,
                            st.bf_scaling)
    log(f"channels {channels}  slices {slices}")
    for ch in channels:
        tag = "  (brightfield/detect)" if ch == idx.detect_channel else ""
        log(f"  scaling {ch} = {modes[ch]}{tag}")

    # --- 3. detect once per well, on the detect slice/channel -----------
    log("detecting embryos ...")
    centers = {}
    dch, dsl = idx.detect_channel, idx.detect_slice
    tps = idx.timepoints
    probe_tps = [tps[0], tps[len(tps) // 2], tps[-1]] if len(tps) > 2 else tps
    # spares to fall back on when a probe frame turns out to be unreadable
    MAX_DETECT_TRIES = 12
    extra_tps = [tps[int(len(tps) * f)] for f in
                 (0.1, 0.25, 0.4, 0.6, 0.75, 0.9)] if len(tps) > 10 else []

    def det(pos):
        """Median centre from a few probe frames. Unreadable frames are skipped
        and further timepoints tried, so a handful of zero-byte files cannot
        cost the whole well -- let alone the whole plate."""
        pts, tried, bad = [], 0, 0
        for tp in probe_tps + extra_tps:
            if len(pts) >= len(probe_tps) or tried >= MAX_DETECT_TRIES:
                break
            k = (pos, tp, dch, dsl)
            if k not in idx.frames:
                continue
            tried += 1
            try:
                img = read_frame(os.path.join(idx.image_dir, idx.frames[k][2]))
            except (OSError,) + UNREADABLE:
                bad += 1
                continue
            pts.append(detect_center(img, um_per_px=um, egg_mm=st.egg_mm))
        if not pts:
            return pos, None
        return pos, (int(np.median([p[0] for p in pts])),
                     int(np.median([p[1] for p in pts])))

    with ThreadPoolExecutor(max_workers=st.workers) as ex:
        for i, (pos, c) in enumerate(ex.map(det, positions)):
            centers[pos] = c
            if progress:
                progress(i + 1, len(positions), f"detect {pos}")
    bad = [p for p, c in centers.items() if c is None]
    if bad:
        log(f"  WARNING: no detection for {bad}")

    # --- 6a. calibration per channel ------------------------------------
    log("computing intensity calibration ...")
    calib = {ch: compute_calibration(idx, st, ch, modes[ch], log)
             for ch in channels}

    # per-well calibration if that is the chosen scope
    well_calib = {}
    if "well" in modes.values():
        for ch in [c for c in channels if modes[c] == "well"]:
            for pos in idx.positions:
                keys = [k for k in idx.frames
                        if k[0] == pos and k[2] == ch and k[3] == dsl]
                hist = np.zeros(HIST_BINS, dtype=np.int64)
                for k in sorted(keys)[:60]:
                    accumulate(hist, tifffile.imread(
                        os.path.join(idx.image_dir, idx.frames[k][2])))
                lo, hi = percentiles_from_hist(hist, [st.low_pct, st.high_pct])
                well_calib[(ch, pos)] = (lo, hi)

    # --- 4,5,6,7. the write pass ----------------------------------------
    jobs = [k for k in idx.frames if k[2] in channels and k[3] in slices
            and k[0] in set(positions) and centers.get(k[0]) is not None]
    jobs.sort()
    total = len(jobs)
    log(f"writing {total:,} crops with {st.workers} workers ...")
    all_dirs = {os.path.join(st.out_dir, st.dir_template.format(
        plate=plate, channel=k[2], pos=k[0],
        well=idx.frames[k][1] or k[0], slice=k[3], tp=k[1])) for k in jobs}
    done = [0]
    written = [0]
    skipped = [0]
    failed = []
    fill = {}          # pos -> smallest fraction of the crop that is real image
    t0 = time.perf_counter()

    def one(k):
        pos, tp, ch, sl = k
        rec, well, fname = idx.frames[k]
        d = os.path.join(st.out_dir, st.dir_template.format(
            plate=plate, channel=ch, pos=pos, well=well or pos, slice=sl, tp=tp))
        fn = st.file_template.format(
            plate=plate, channel=ch, pos=pos, well=well or pos, slice=sl, tp=tp)
        dst = os.path.join(d, fn)
        if not st.overwrite and os.path.exists(dst):
            skipped[0] += 1
            return
        try:
            img = read_frame(os.path.join(idx.image_dir, fname))
        except (OSError,) + UNREADABLE as e:
            failed.append((str(k), f"read: {type(e).__name__}: {e}"))
            return
        cy, cx = centers[pos]
        c, f = crop_at(img, cy, cx, crop_px)
        # keep the WORST fill seen for this well; a perfect 1.0 must still be
        # recorded, so every well appears in the manifest
        fill[pos] = min(fill.get(pos, 1.0), f)
        c = resize(c, st.output_px)
        mode = modes[ch]
        if mode == "image":
            lo, hi = percentiles_from_hist(
                accumulate(np.zeros(HIST_BINS, dtype=np.int64), c),
                [st.low_pct, st.high_pct])
        elif mode == "well":
            lo, hi = well_calib[(ch, pos)]
        else:
            lo, hi = calib[ch]["lo"], calib[ch]["hi"]
        out, _dt = apply_scale(c, mode, lo or 0, hi or 1)
        # Write to a temp name and rename: a half-written file left behind by a
        # drive that vanished mid-write would otherwise look "already done" on
        # the resume pass and never be repaired. Rename is atomic.
        try:
            _retry(os.makedirs, d, exist_ok=True)
            tmp = dst + ".part"
            _retry(tifffile.imwrite, tmp, out)
            _retry(os.replace, tmp, dst)
        except (OSError,) + UNREADABLE as e:
            failed.append((str(k), f"write: {type(e).__name__}: {e}"))
            return
        written[0] += 1

    if st.metadata_only:
        log("metadata-only: no crops written, existing files untouched")
    else:
      with ThreadPoolExecutor(max_workers=st.workers) as ex:
        for _ in ex.map(one, jobs):
            done[0] += 1
            if should_stop and should_stop():
                log("STOPPED by request")
                break
            if progress and done[0] % 50 == 0:
                progress(done[0], total, f"{written[0]:,} written")
    if progress:
        progress(done[0], total, "done")

    dt = time.perf_counter() - t0
    off = {p: round(f, 3) for p, f in fill.items() if f < 0.995}
    if off:
        log(f"  WARNING: {len(off)} well(s) have a crop running OFF the frame — "
            f"the detected centre is closer to an edge than half the crop, so "
            f"part of the image is black padding:")
        for p, f in sorted(off.items(), key=lambda kv: kv[1])[:8]:
            log(f"    {p}: only {f*100:.0f}% of the crop is real image")
    n_corrupt = sum(1 for _k, why in failed if "TiffFileError" in why)
    if n_corrupt:
        log(f"  {n_corrupt} frame(s) were UNREADABLE (zero-byte raw files the "
            f"microscope never finished writing). They are skipped, not "
            f"retried; every other frame was processed.")
    if failed:
        log(f"  {len(failed)} frame(s) FAILED after {IO_ATTEMPTS} attempts "
            f"(drive dropped?). Re-run the same command to fill them in — "
            f"finished files are skipped.")
        for k, why in failed[:5]:
            log(f"    {k}  {why}")
    manifest = {
        "plate": plate,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "raw_dir": idx.image_dir,
        "out_dir": st.out_dir,
        "n_written": written[0], "n_skipped": skipped[0],
        "n_failed": len(failed), "n_unreadable": n_corrupt,
        "crop_fill_frac": {p: round(f, 4) for p, f in sorted(fill.items())},
        "wells_cropping_off_frame": sorted(off),
        "failed": failed[:200],
        "seconds": round(dt, 1),
        "files_per_sec": round(written[0] / dt, 1) if dt else None,
        "raw_frame": list(idx.frame_shape) if idx.frame_shape else None,
        "binning": idx.binning, "binning_reason": idx.binning_reason,
        "px_sensor_nm": idx.px_sensor_nm, "um_per_px": um,
        "fov_mm": st.fov_mm, "crop_px": crop_px, "output_px": st.output_px,
        "egg_mm": st.egg_mm,
        "temperature_C": idx.temperature_C,
        "detect_slice": dsl, "detect_channel": dch,
        "centers": {p: list(c) for p, c in centers.items() if c},
        "intensity_calibration": calib,
        "scaling_modes": modes,
        "settings": st.to_dict(),
    }
    # --- the plate folder, derived from what was actually written ---------
    # The output template is user-editable, so the plate root is the common
    # parent of every directory written, walked up to the level directly under
    # out_dir. Never guessed from the template string.
    plate_dir = st.out_dir
    if all_dirs:
        common = os.path.commonpath(list(all_dirs))
        rel = os.path.relpath(common, st.out_dir)
        top = rel.split(os.sep)[0]
        if top not in (".", ".."):
            plate_dir = os.path.join(st.out_dir, top)

    # --- plate_metadata.json: what makes PlateNotate SEE this plate -------
    # _list_plates() only accepts a per-channel plate when this file sits at
    # the plate root. Without it the plate is invisible at every level, no
    # matter how correct the images are. It also carries the display range, so
    # the annotator shows crops through ONE constant map per channel instead of
    # falling back to a per-frame stretch.
    chan_cal = {ch: {"p_low": c.get("lo"), "p_high": c.get("hi"),
                     "low_percentile": c.get("low_pct"),
                     "high_percentile": c.get("high_pct"),
                     "mode": c.get("mode"),
                     "n_frames_sampled": c.get("n_frames_sampled")}
                for ch, c in calib.items()
                if c.get("lo") is not None and c.get("hi") is not None}
    plate_meta = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pipeline_version": "embryo_crop 1.0",
        "raw_dir": idx.image_dir,
        "n_processed": len(positions),
        "crop_size": crop_px,
        "output_size": st.output_px,
        "detect_slice": dsl,
        "bf_channel": dch,
        "channels": channels,
        "z_slices": slices,
        "um_per_px": um,
        "binning": idx.binning,
        "incubation_temp_c": idx.temperature_C,
        "channel_calibrations": chan_cal,
        "scaling_modes": modes,
        "fluorescence_calibration": {"channel_calibrations": chan_cal},
        "position_to_well_id": {p: (idx.pos_to_well.get(p) or p)
                                for p in positions},
        "positions": {p: {"plate_position": p,
                          "well_id": idx.pos_to_well.get(p) or p,
                          "status": "ok" if centers.get(p) else "no_detection",
                          "center_yx": list(centers[p]) if centers.get(p) else None,
                          "crop_fill_frac": round(fill.get(p, 1.0), 4)}
                      for p in positions},
    }
    if idx.interval_min:
        plate_meta["timepoint_interval_min"] = idx.interval_min
    for k in ("line", "guide", "assay"):          # omitted when not set
        if getattr(st, k):
            plate_meta[k] = getattr(st, k)

    os.makedirs(plate_dir, exist_ok=True)
    ppath = os.path.join(plate_dir, "plate_metadata.json")
    with open(ppath, "w") as fh:
        json.dump(plate_meta, fh, indent=2)
    manifest["plate_dir"] = plate_dir
    manifest["plate_metadata"] = ppath

    os.makedirs(st.out_dir, exist_ok=True)
    mpath = os.path.join(st.out_dir, f"{plate}_manifest.json")
    with open(mpath, "w") as fh:
        json.dump(manifest, fh, indent=2)
    log(f"plate_metadata.json -> {ppath}")
    log(f"OPEN IN PLATENOTATE:  {os.path.dirname(plate_dir)}   "
        f"(then choose '{os.path.basename(plate_dir)}')")
    log(f"wrote {written[0]:,} crops ({skipped[0]:,} skipped) in {dt/60:.1f} min "
        f"= {written[0]/dt:.0f} files/s")
    log(f"manifest -> {mpath}")
    log(f"TOTAL {(time.perf_counter()-t_start)/60:.1f} min")
    return manifest


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="crop every embryo in a plate")
    p.add_argument("raw_dir")
    p.add_argument("out_dir")
    p.add_argument("--plate", default=None)
    p.add_argument("--channels", default="")
    p.add_argument("--wells", default="")
    p.add_argument("--line", default="")
    p.add_argument("--guide", default="")
    p.add_argument("--assay", default="")
    p.add_argument("--scaling", default="plate",
                   help="one mode for all channels, or a per-channel spec: "
                        "'plate,CO6=image'. Modes: " + "|".join(SCALING_MODES))
    p.add_argument("--um-per-px", type=float, default=None,
                   help="override the pixel size when the PX token in the "
                        "filenames is wrong")
    p.add_argument("--egg-mm", type=float, default=None,
                   help=f"specimen diameter in mm (default {EGG_MM}); sets the "
                        "detection smoothing radius")
    p.add_argument("--bf-scaling", default=None,
                   help="mode for the auto-detected brightfield channel, "
                        "e.g. --scaling plate --bf-scaling image")
    p.add_argument("--fixed-lo", type=float, default=None,
                   help="low count for --scaling fixed")
    p.add_argument("--fixed-hi", type=float, default=None,
                   help="high count for --scaling fixed")
    p.add_argument("--low-pct", type=float, default=1.0)
    p.add_argument("--high-pct", type=float, default=99.9)
    p.add_argument("--fov-mm", type=float, default=DEFAULT_FOV_MM)
    p.add_argument("--output-px", type=int, default=DEFAULT_OUTPUT_PX)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--stats-sample", type=int, default=400)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--metadata-only", action="store_true",
                   help="only (re)write plate_metadata.json; touch no images")
    a = p.parse_args()
    run(Settings(raw_dir=a.raw_dir, out_dir=a.out_dir, plate=a.plate,
                 channels=[c for c in a.channels.split(",") if c],
                 positions=[w for w in a.wells.split(",") if w],
                 line=a.line, guide=a.guide, assay=a.assay,
                 scaling=a.scaling, bf_scaling=a.bf_scaling, egg_mm=a.egg_mm,
                 low_pct=a.low_pct, high_pct=a.high_pct,
                 fixed_lo=a.fixed_lo, fixed_hi=a.fixed_hi,
                 fov_mm=a.fov_mm, output_px=a.output_px, workers=a.workers,
                 um_per_px_override=a.um_per_px,
                 stats_sample=a.stats_sample, overwrite=a.overwrite,
                 metadata_only=a.metadata_only))
