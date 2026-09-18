"""Read a Nikon Ti2 `.nd2` folder with the same interface as an ACQUIFER folder.

The ACQUIFER writes one TIFF per frame with the acquisition parameters encoded
in the filename. The Ti2 writes ONE `.nd2` per (well, channel) holding the whole
5-D stack, with the parameters inside the file. So the only real difference is
where a frame lives and where the calibration comes from -- the detection,
cropping and scaling in `process.py` are identical.

`ND2Index` therefore exposes exactly what `acquifer.RawIndex` exposes, and
`process.py` reads every frame through `idx.read(key)` so it never needs to know
which of the two it is holding.

Layout this was written against (2026-08-17):

    <session>/Well<WW>_Channel<name>_Seq<NNNN>.nd2
    dims  T=50  P=4  Z=11  Y=2304  X=2304  uint16
    24 wells x 4 positions = 96 fields, channels Dia / 475 / mscarlet

The `P` axis inside each file is a set of stage positions WITHIN the well, so a
field is identified as `<well>_P<n>` and each becomes its own crop series.

Calibration comes from `voxel_size().x` (1.6151 um/px at Plan Apo 4x), which is
the true image pixel size -- no binning arithmetic and no filename token to
mistrust, unlike [[aqv_plate_pixel_sizes]].
"""
from __future__ import annotations

import os
import re
import threading
from collections import defaultdict

import numpy as np

try:
    import nd2
except ImportError:                                       # pragma: no cover
    nd2 = None

DEFAULT_FOV_MM = 1.872
DEFAULT_OUTPUT_PX = 576

# Well<WW>_Channel<name>_Seq<NNNN>.nd2
_NAME = re.compile(r"Well(?P<well>[A-Z]\d+)_Channel(?P<ch>.+?)_Seq(?P<seq>\d+)\.nd2$")

# Transmitted-light channel names, in preference order. Detection must run on
# the embryo outline, never on a fluorescence channel: fluorescence shows the
# injection, so an uninjected well would have nothing to centre on.
_BF_NAMES = ("dia", "bf", "brightfield", "trans", "dic", "phase")


def looks_like_nd2_dir(path: str) -> bool:
    """True if this folder (or its immediate children) holds .nd2 files."""
    try:
        if any(f.endswith(".nd2") for f in os.listdir(path)):
            return True
        for d in sorted(os.listdir(path)):
            sub = os.path.join(path, d)
            if os.path.isdir(sub) and any(
                    f.endswith(".nd2") for f in os.listdir(sub)):
                return True
    except OSError:
        pass
    return False


def _tile_offsets(points, um_per_px, frame_px):
    """Pixel offsets for a set of stage positions, orientation DERIVED.

    The stage axes do not have a fixed sign relative to the image axes -- it
    depends on the microscope's mounting and on how the camera is rotated -- so
    guessing costs you a mosaic that is mirrored and silently wrong. Instead all
    four sign combinations are returned for the caller to score against the
    actual overlap, and the winner is the one where overlapping pixels agree.
    """
    xs = np.array([p[0] for p in points], float)
    ys = np.array([p[1] for p in points], float)
    out = {}
    for sx in (+1, -1):
        for sy in (+1, -1):
            px = sx * (xs - xs.min() if sx > 0 else xs - xs.max()) / um_per_px
            py = sy * (ys - ys.min() if sy > 0 else ys - ys.max()) / um_per_px
            px = np.round(px - px.min()).astype(int)
            py = np.round(py - py.min()).astype(int)
            out[(sx, sy)] = list(zip(py, px))
    return out


def _score_layout(tiles, offsets, frame_px):
    """Mean CORRELATION between neighbouring tiles over the pixels they share.

    Higher is better; a correct layout puts the same content in both tiles'
    overlap. Correlation rather than a difference/variance measure because most
    of the overlap is dark background that matches under every orientation --
    measured on this data, variance separated the four candidates by only 1.15x
    (useless) while correlation separates them 0.98 vs 0.07.
    """
    tot_w = tot_r = 0.0
    for i in range(len(tiles)):
        for j in range(i + 1, len(tiles)):
            (ay, ax), (by, bx) = offsets[i], offsets[j]
            y0, y1 = max(ay, by), min(ay, by) + frame_px
            x0, x1 = max(ax, bx), min(ax, bx) + frame_px
            if y1 - y0 < 32 or x1 - x0 < 32:
                continue
            A = tiles[i][y0 - ay:y1 - ay, x0 - ax:x1 - ax].astype(np.float64)
            B = tiles[j][y0 - by:y1 - by, x0 - bx:x1 - bx].astype(np.float64)
            if A.std() < 1e-6 or B.std() < 1e-6:
                continue
            r = float(np.corrcoef(A.ravel(), B.ravel())[0, 1])
            area = (y1 - y0) * (x1 - x0)
            tot_r += r * area
            tot_w += area
    return (tot_r / tot_w) if tot_w else -1.0


class _Handles:
    """One open ND2File per thread per path.

    `process.py` crops with a thread pool, and an ND2 reader handle is not
    documented as thread-safe, so handles are thread-local rather than shared.
    Reads are memory-mapped, so keeping them open is cheap and reopening a
    21 GB file per frame would not be.
    """

    def __init__(self):
        self._local = threading.local()

    def get(self, path):
        cache = getattr(self._local, "cache", None)
        if cache is None:
            cache = self._local.cache = {}
        h = cache.get(path)
        if h is None:
            h = cache[path] = nd2.ND2File(path)
        return h

    def close(self):
        cache = getattr(self._local, "cache", None)
        if cache:
            for h in cache.values():
                try:
                    h.close()
                except Exception:
                    pass
            cache.clear()


class ND2Index:
    """Same public surface as acquifer.RawIndex, backed by .nd2 files."""

    def __init__(self, image_dir, frames, frame_shape, frame_dtype,
                 um_per_px, interval_min, tp_time_ms, channel_order,
                 mosaic=False, tile_offsets=None, tile_px=None,
                 mosaic_rms=None):
        self.image_dir = image_dir
        self.frames = frames            # (pos, tp, ch, sl) -> (meta, well, path, seq)
        # In mosaic mode one "frame" is the 2x2 stitch of a well, and `seq` is
        # a list of (path, seq) tiles rather than a single plane.
        self.mosaic = mosaic
        self.tile_offsets = tile_offsets or []
        self.tile_px = tile_px
        self.mosaic_rms = mosaic_rms
        self.frame_shape = frame_shape
        self.frame_dtype = frame_dtype

        self.positions = sorted({k[0] for k in frames})
        self.timepoints = sorted({k[1] for k in frames})
        self.channels = [c for c in channel_order
                         if c in {k[2] for k in frames}]
        self.slices = sorted({k[3] for k in frames})
        # A field keeps its own id (`A01_P2`) as its "well", so output
        # filenames stay unique across the 4 stage positions inside one well.
        # The real plate well is tracked separately, for reporting only.
        self.pos_to_well = {}
        for (pos, _, _, _), rec in frames.items():
            self.pos_to_well.setdefault(pos, rec[1])
        self.wells = sorted({p.split("_P")[0] for p in self.positions})
        self.fields_per_well = max(
            1, len(self.positions) // max(1, len(self.wells)))

        # No binning arithmetic: the Ti2 reports the true image pixel size, so
        # there is nothing to derive and nothing to get wrong.
        self.binning = 1
        self.binning_reason = "Ti2 reports um/px directly; no binning inferred"
        self.px_sensor_nm = round(um_per_px * 1000.0, 1) if um_per_px else None
        self.um_per_px = round(um_per_px, 4) if um_per_px else None

        # Not recorded by the Ti2 job -- absent rather than invented.
        self.temperature_C = None

        self.interval_min = interval_min
        self.tp_time_ms = tp_time_ms or {}
        if self.tp_time_ms:
            t0 = min(self.tp_time_ms.values())
            self.tp_minutes = {tp: round((t - t0) / 60000.0, 4)
                               for tp, t in self.tp_time_ms.items()}
        else:
            self.tp_minutes = {}

        self.segments = None
        self.source_dirs = [image_dir]
        self._handles = _Handles()

        self.detect_slice = (self.slices[len(self.slices) // 2]
                             if self.slices else None)
        self.detect_channel = None
        for c in self.channels:
            if c.lower() in _BF_NAMES:
                self.detect_channel = c
                break
        if self.detect_channel is None and self.channels:
            # Refuse to guess silently: pick one, but say so loudly upstream.
            self.detect_channel = self.channels[0]
            self.detect_channel_guessed = True
        else:
            self.detect_channel_guessed = False

    # -- the physical-FOV rule, identical to the ACQUIFER path -------------
    def crop_px(self, fov_mm=DEFAULT_FOV_MM, um_per_px=None):
        u = um_per_px or self.um_per_px
        return int(round(fov_mm * 1000.0 / u)) if u else None

    def path(self, key):
        """The .nd2 the frame lives in (a frame is not its own file here)."""
        return self.frames[key][2]

    def _plane(self, path, seq):
        # Copy: read_frame returns a view into the mmap, and anything that
        # outlives the handle segfaults rather than raising.
        a = np.array(self._handles.get(path).read_frame(seq))
        return a[..., 0] if a.ndim == 3 and a.shape[-1] == 1 else a

    def read(self, key):
        """The 2-D image for this key.

        Plain mode: one plane out of the .nd2. Mosaic mode: the well's tiles
        stitched onto one canvas at their stage offsets, averaging where they
        overlap so the seam does not show up as an edge the detector chases.
        """
        _meta, _well, path, seq = self.frames[key]
        if not self.mosaic:
            return self._plane(path, seq)

        n = self.tile_px
        h = max(o[0] for o in self.tile_offsets) + n
        w = max(o[1] for o in self.tile_offsets) + n
        acc = np.zeros((h, w), np.float32)
        cnt = np.zeros((h, w), np.float32)
        for (p, s), (oy, ox) in zip(seq, self.tile_offsets):
            acc[oy:oy + n, ox:ox + n] += self._plane(p, s)
            cnt[oy:oy + n, ox:ox + n] += 1
        np.maximum(cnt, 1, out=cnt)
        return (acc / cnt).astype(self.frame_dtype)

    def files_for(self, pos, channel, slice_=None):
        out = []
        for (p, tp, ch, sl) in self.frames:
            if p == pos and ch == channel and (slice_ is None or sl == slice_):
                out.append((tp, sl, self.path((p, tp, ch, sl))))
        return sorted(out)

    def close(self):
        self._handles.close()

    def summary(self, fov_mm=DEFAULT_FOV_MM, output_px=DEFAULT_OUTPUT_PX,
                um_per_px=None):
        u = um_per_px or self.um_per_px
        c = self.crop_px(fov_mm, u)
        w = self.wells
        L = [
            f"folder        {self.image_dir}",
            f"source        Nikon Ti2 .nd2",
            f"frames        {len(self.frames):,}",
            f"wells         {len(w)}  ({w[0]}..{w[-1]})" if w else "wells none",
            (f"mosaic        {len(self.tile_offsets)} tiles per well stitched "
             f"into {self.frame_shape[1]}x{self.frame_shape[0]}  "
             f"[overlap r={self.mosaic_rms:.3f}]")
            if self.mosaic else
            f"fields        {len(self.positions)}  "
            f"({self.positions[0]}..{self.positions[-1]})",
            f"channels      {', '.join(self.channels)}   "
            f"detect on {self.detect_channel}"
            + ("   <-- GUESSED, no transmitted-light channel found"
               if self.detect_channel_guessed else ""),
            f"z-slices      {len(self.slices)}  ({', '.join(self.slices)})   "
            f"detect {self.detect_slice}",
            f"timepoints    {len(self.timepoints)}  "
            f"(LO{min(self.timepoints):03d}..LO{max(self.timepoints):03d})"
            if self.timepoints else "timepoints none",
            f"raw frame     {self.frame_shape[1]}x{self.frame_shape[0]} "
            f"{self.frame_dtype}" if self.frame_shape else "raw frame unknown",
            f"pixel size    {u} um/px  [Ti2 voxel_size, not derived]",
            f"interval      {self.interval_min} min" if self.interval_min else "",
            "",
            f"CROP          {c} px  =  {fov_mm} mm  ->  resized to "
            f"{output_px}x{output_px}"
            if c else "CROP          cannot be derived (no pixel size)",
        ]
        return "\n".join(x for x in L if x)


def _session_dirs(root):
    """The folder(s) actually holding .nd2 files."""
    if any(f.endswith(".nd2") for f in os.listdir(root)):
        return [root]
    return [os.path.join(root, d) for d in sorted(os.listdir(root))
            if os.path.isdir(os.path.join(root, d))
            and any(f.endswith(".nd2") for f in os.listdir(os.path.join(root, d)))]


def index_nd2_folder(raw_dir, progress=None):
    """Index one Ti2 session folder of .nd2 files."""
    if nd2 is None:
        raise RuntimeError(
            "reading .nd2 needs the `nd2` package: pip install nd2")

    dirs = _session_dirs(raw_dir)
    if not dirs:
        raise RuntimeError(f"no .nd2 files under {raw_dir}")
    image_dir = dirs[0]
    if len(dirs) > 1:
        raise RuntimeError(
            "several Ti2 sessions found; point at ONE of them:\n  " +
            "\n  ".join(dirs))

    files = sorted(f for f in os.listdir(image_dir) if f.endswith(".nd2"))
    frames = {}
    shape = dtype = None
    um = None
    channel_order = []
    times_by_tp = defaultdict(list)
    stage_pts = None
    tiles_by_key = defaultdict(list)     # (well, tp, ch, sl) -> [(path, seq)]

    for i, fn in enumerate(files):
        m = _NAME.search(fn)
        if not m:
            continue
        well, ch = m.group("well"), m.group("ch")
        if ch not in channel_order:
            channel_order.append(ch)
        full = os.path.join(image_dir, fn)
        if progress:
            progress(i, len(files), fn)

        with nd2.ND2File(full) as n:
            sizes = dict(n.sizes)
            nT = sizes.get("T", 1)
            nP = sizes.get("P", 1)
            nZ = sizes.get("Z", 1)
            if shape is None:
                shape = (sizes["Y"], sizes["X"])
                dtype = str(n.dtype)
                v = n.voxel_size()
                um = float(v.x)
                # Stage XY of each P, straight from the XYPosLoop. If the P
                # points are closer together than one frame is wide, they are
                # overlapping TILES of one well, not separate fields.
                for loop in n.experiment:
                    pts = getattr(getattr(loop, "parameters", None),
                                  "points", None)
                    if pts:
                        stage_pts = [(p.stagePositionUm.x, p.stagePositionUm.y)
                                     for p in pts]
                        break

            # Per-timepoint wall clock, so elapsed time is read rather than
            # assumed -- the same reason the ACQUIFER path keeps tp_time_ms.
            try:
                for ev in n.events():
                    t = ev.get("Time [s]")
                    idx_t = ev.get("Index")
                    if t is not None and idx_t is not None:
                        tp = int(idx_t) // (nP * nZ) + 1
                        if 1 <= tp <= nT:
                            times_by_tp[tp].append(float(t) * 1000.0)
            except Exception:
                pass

            for t in range(nT):
                for z in range(nZ):
                    sl = f"SL{z+1:03d}"
                    for p in range(nP):
                        seq = (t * nP + p) * nZ + z
                        tiles_by_key[(well, t + 1, ch, sl)].append((full, seq))
                        frames[(f"{well}_P{p+1}", t + 1, ch, sl)] = (
                            {}, f"{well}_P{p+1}", full, seq)

    if not frames:
        raise RuntimeError(f"no readable .nd2 frames in {image_dir}")

    # One (well, channel) must come from exactly ONE file. A session can hold a
    # retry -- e.g. 20260810_143425_865 has BOTH WellA01_ChannelDia_Seq0000 and
    # _Seq0002 -- and merging them would silently stack 8 tiles onto 4 stage
    # offsets. Name the files and stop; picking one for the user would be a
    # guess about which acquisition they meant.
    per_wc = defaultdict(set)
    for (well, _tp, ch, _sl), tl in tiles_by_key.items():
        per_wc[(well, ch)].update(p for p, _s in tl)
    dupes = {k: sorted(v) for k, v in per_wc.items() if len(v) > 1}
    if dupes:
        lines = []
        for (well, ch), paths in sorted(dupes.items())[:5]:
            lines.append(f"  {well} / {ch}:")
            lines += [f"      {os.path.basename(p)}" for p in paths]
        raise RuntimeError(
            f"{len(dupes)} (well, channel) pair(s) appear in MORE THAN ONE .nd2 "
            f"in {image_dir} -- probably a repeated/retried acquisition.\n"
            + "\n".join(lines)
            + "\nMove the ones you do not want out of the folder and re-run; "
              "merging them would stack tiles onto the wrong stage offsets.")

    # --- mosaic or separate fields? Decided by the stage geometry -----------
    mosaic = False
    offsets = None
    rms = None
    frame_px = shape[0]
    if stage_pts and len(stage_pts) > 1:
        span_x = max(p[0] for p in stage_pts) - min(p[0] for p in stage_pts)
        span_y = max(p[1] for p in stage_pts) - min(p[1] for p in stage_pts)
        step = max(span_x, span_y) / max(1, int(round(len(stage_pts) ** 0.5)) - 1)
        if step < frame_px * um * 0.98:      # neighbours overlap => tiles
            mosaic = True

    if mosaic:
        cand = _tile_offsets(stage_pts, um, frame_px)

        # Score every orientation on REAL pixels and keep the one whose overlaps
        # actually agree. Never assume a stage/camera sign convention.
        #
        # The probe frame has to CONTAIN STRUCTURE or every orientation scores
        # the same: a featureless out-of-focus fluorescence plane made the first
        # attempt tie at 3.8 vs 3.8. So probe the transmitted-light channel at
        # the middle z (in focus), across a few timepoints, and require the
        # winner to win consistently.
        chans = sorted({k[2] for k in tiles_by_key})
        bf = next((c for c in chans if c.lower() in _BF_NAMES), chans[0])
        sls = sorted({k[3] for k in tiles_by_key})
        mid_sl = sls[len(sls) // 2]
        tps = sorted({k[1] for k in tiles_by_key})
        probe_tps = [tps[len(tps) // 4], tps[len(tps) // 2],
                     tps[(3 * len(tps)) // 4]]

        wins = defaultdict(int)
        margins = []
        for tp in dict.fromkeys(probe_tps):
            wkey = sorted({k[0] for k in tiles_by_key})[0]
            pk = (wkey, tp, bf, mid_sl)
            if pk not in tiles_by_key:
                continue
            with nd2.ND2File(tiles_by_key[pk][0][0]) as n:
                # np.array(...) COPIES. read_frame hands back a view into the
                # memory-mapped file, and touching one after the handle closes
                # is a segfault, not an exception.
                tl = [np.array(n.read_frame(s)) for _p, s in tiles_by_key[pk]]
                tl = [a[..., 0] if a.ndim == 3 else a for a in tl]
            # higher correlation = better, so sort descending
            sc = sorted(((_score_layout(tl, o, frame_px), k)
                         for k, o in cand.items()),
                        key=lambda r: r[0], reverse=True)
            wins[sc[0][1]] += 1
            margins.append((sc[0][0], sc[1][0]))

        if not wins:
            raise RuntimeError("no usable probe frame to orient the mosaic")
        best_signs, n_win = max(wins.items(), key=lambda kv: kv[1])
        offsets = cand[best_signs]
        best_r = min(m[0] for m in margins)
        next_r = max(m[1] for m in margins)
        if n_win < len(margins) or best_r < 0.5 or best_r - next_r < 0.2:
            raise RuntimeError(
                "cannot tell the mosaic orientation apart (winner took "
                f"{n_win}/{len(margins)} probes, best overlap r={best_r:.3f} "
                f"vs next {next_r:.3f}). Refusing to stitch blind.")
        rms = best_r
        # Plain ints: numpy scalars leak all the way into the manifest and
        # json.dump refuses them ("Object of type int64 is not JSON
        # serializable") only at the very end, after the whole plate is written.
        offsets = [(int(a), int(b)) for a, b in offsets]
        h = int(max(o[0] for o in offsets) + frame_px)
        w = int(max(o[1] for o in offsets) + frame_px)
        shape = (h, w)
        frames = {}
        for (well, tp, ch, sl), tl in tiles_by_key.items():
            frames[(well, tp, ch, sl)] = ({}, well, tl[0][0], tl)

    tp_time_ms = {tp: int(np.median(v)) for tp, v in times_by_tp.items() if v}
    interval = None
    if len(tp_time_ms) > 1:
        ordered = [tp_time_ms[k] for k in sorted(tp_time_ms)]
        gaps = [b - a for a, b in zip(ordered, ordered[1:]) if b > a]
        if gaps:
            interval = round(float(np.median(gaps)) / 60000.0, 4)

    return ND2Index(image_dir, frames, shape, dtype, um, interval,
                    tp_time_ms, channel_order,
                    mosaic=mosaic, tile_offsets=offsets, tile_px=frame_px,
                    mosaic_rms=rms)
