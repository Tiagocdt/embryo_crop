#!/usr/bin/env python3
"""Read an ACQUIFER raw folder and say exactly what is in it.

Nothing here is configured by hand. Every acquisition parameter is derived
from the filenames plus one TIFF header:

    -A001--PO01--LO001--CO1--SL001--PX32500--PW0060--IN0017--TM263--...--WE00001.tif
     |     |     |      |     |      |                        |
     |     |     |      |     |      |                        TM tenths-degC
     |     |     |      |     |      PX tenths-nm at the sensor
     |     |     |      |     SL z-slice
     |     |     |      CO channel
     |     |     LO loop = timepoint
     |     plate position
     well coordinate

Two naming styles exist and both are handled:
    EMBL ACQUIFER    WE00076---G004--PO01--LO001--CO2--SL001--PX16250--...
    Wittbrodt IM     -A001--PO01--LO001--CO1--SL001--PX32500--...--WE00001

THE ONE IDEA THAT MATTERS
-------------------------
The crop is defined by a PHYSICAL field of view, never by a pixel count:

    um_per_px = PX_nm / 1000 * binning
    crop_px   = FOV_MM * 1000 / um_per_px
    output    = resize(crop, OUTPUT_PX)

so a 2x objective, a 4x objective, binned and unbinned data all produce
images of the same physical extent AND the same pixel dimensions. Binning is
measured (sensor px / frame width), never assumed -- assuming "always 2x2"
silently doubled every physical size on unbinned data.
"""
from __future__ import annotations

import os
import re
from collections import Counter, defaultdict

import numpy as np
import tifffile

# The ACQUIFER camera sensor, pixels per side. Binning = this / frame width.
NATIVE_SENSOR_PX = 2048

# The field of view that defines the dataset, in millimetres. 576 px at
# 3.25 um/px, i.e. what every existing crop and every trained model uses.
# Change this and you are making a NEW dataset, not reprocessing the old one.
DEFAULT_FOV_MM = 1.872
DEFAULT_OUTPUT_PX = 576

_CORE = re.compile(
    r"(?P<row>[A-Z])(?P<col>\d+)--PO\d+--LO(?P<tp>\d+)--CO(?P<ch>\d+)--SL(?P<sl>\d+)--")
_TOKEN = re.compile(r"--(?P<k>[A-Z]{1,2})(?P<v>-?\d+)")
_WELL = re.compile(r"WE(?P<we>\d{4,6})")


def parse_name(name: str):
    """Parse one ACQUIFER filename into a dict, or None if it is not one."""
    m = _CORE.search(name)
    if not m:
        return None
    rec = {
        "pos": f"{m.group('row')}{int(m.group('col')):02d}",
        "tp": int(m.group("tp")),
        "channel": f"CO{int(m.group('ch'))}",
        "slice": f"SL{int(m.group('sl')):03d}",
    }
    w = _WELL.search(name)
    rec["well"] = f"WE{w.group('we')}" if w else None
    for t in _TOKEN.finditer(name):
        rec[t.group("k")] = t.group("v")
    return rec


def find_image_dir(root: str) -> str:
    """The newer ACQUIFER layout nests the frames one level down:

        20241206_V03/
            2024-11-28_standard.exf
            20241206_121834_V03/      <- the frames live here
                Log/  ~PlateViewer*.tmp/  *.tif

    If `root` holds no frames but exactly one subdirectory does, descend.
    """
    if _has_tifs(root):
        return root
    subs = [os.path.join(root, d) for d in sorted(os.listdir(root))
            if os.path.isdir(os.path.join(root, d)) and not d.startswith("~")]
    hits = [d for d in subs if _has_tifs(d)]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise ValueError(f"{root} has several image folders: {hits}. Point me at one.")
    raise ValueError(f"no ACQUIFER .tif frames found in {root}")


def _has_tifs(d: str, probe: int = 400) -> bool:
    try:
        with os.scandir(d) as it:
            for i, e in enumerate(it):
                if i > probe:
                    break
                if e.name.lower().endswith(".tif") and parse_name(e.name):
                    return True
    except (NotADirectoryError, PermissionError, FileNotFoundError):
        pass
    return False


class RawIndex:
    """Everything known about a raw folder, with nothing typed by hand."""

    def __init__(self, image_dir, frames, frame_shape, frame_dtype):
        self.image_dir = image_dir
        self.frames = frames                      # (pos, tp, channel, slice) -> filename
        self.frame_shape = frame_shape
        self.frame_dtype = frame_dtype

        self.positions = sorted({k[0] for k in frames})
        self.timepoints = sorted({k[1] for k in frames})
        self.channels = sorted({k[2] for k in frames},
                               key=lambda c: int(c[2:]))
        self.slices = sorted({k[3] for k in frames})
        self.pos_to_well = {}
        for (pos, _, _, _), rec in frames.items():
            self.pos_to_well.setdefault(pos, rec[1])

        # --- binning: MEASURED, never assumed ---------------------------
        self.binning = None
        self.binning_reason = "frame shape unknown"
        if frame_shape:
            w = frame_shape[1]
            if NATIVE_SENSOR_PX % w == 0:
                self.binning = NATIVE_SENSOR_PX // w
                self.binning_reason = (f"{w}px frame / {NATIVE_SENSOR_PX}px sensor "
                                       f"= {self.binning}x{self.binning}")
            else:
                self.binning_reason = (f"{w}px frame does not divide "
                                       f"{NATIVE_SENSOR_PX}px sensor")

        # --- pixel size --------------------------------------------------
        px = Counter(rec[0].get("PX") for rec in frames.values()
                     if rec[0].get("PX"))
        self.px_sensor_nm = float(px.most_common(1)[0][0]) * 0.1 if px else None
        self.um_per_px = (round(self.px_sensor_nm * self.binning / 1000.0, 4)
                          if (self.px_sensor_nm and self.binning) else None)

        tm = Counter(rec[0].get("TM") for rec in frames.values() if rec[0].get("TM"))
        self.temperature_C = float(tm.most_common(1)[0][0]) * 0.1 if tm else None

        # --- timepoint interval, from the T millisecond clock --------------
        # Median positive gap between adjacent timepoints of the same
        # (position, channel, slice), so a slow z-stack inside one timepoint
        # is never mistaken for the interval between timepoints.
        self.interval_min = None
        series = defaultdict(dict)
        for (pos, tp, ch, sl), (rec, _w, _f) in frames.items():
            if rec.get("T") is not None:
                series[(pos, ch, sl)][tp] = int(rec["T"])
        gaps = []
        for tps in series.values():
            ordered = [tps[k] for k in sorted(tps)]
            gaps += [b - a for a, b in zip(ordered, ordered[1:]) if b > a]
        if gaps:
            self.interval_min = round(float(np.median(gaps)) / 60000.0, 4)

        self.detect_slice = (self.slices[len(self.slices) // 2]
                             if self.slices else None)
        # CO6 is the brightfield LED on the EMBL machine; otherwise the
        # channel with the most frames is the one to detect on.
        if "CO6" in self.channels:
            self.detect_channel = "CO6"
        else:
            cc = Counter(k[2] for k in frames)
            self.detect_channel = cc.most_common(1)[0][0] if cc else None

    # -- the physical-FOV rule -------------------------------------------
    def crop_px(self, fov_mm=DEFAULT_FOV_MM, um_per_px=None):
        """Pixels to cut from THIS data to cover `fov_mm` millimetres."""
        u = um_per_px or self.um_per_px
        if not u:
            return None
        return int(round(fov_mm * 1000.0 / u))

    def files_for(self, pos, channel, slice_=None):
        out = []
        for (p, tp, ch, sl), (rec, _well, fname) in self.frames.items():
            if p == pos and ch == channel and (slice_ is None or sl == slice_):
                out.append((tp, sl, os.path.join(self.image_dir, fname)))
        return sorted(out)

    def summary(self, fov_mm=DEFAULT_FOV_MM, output_px=DEFAULT_OUTPUT_PX,
                um_per_px=None):
        u = um_per_px or self.um_per_px
        c = self.crop_px(fov_mm, u)
        L = [
            f"folder        {self.image_dir}",
            f"frames        {len(self.frames):,}",
            f"wells         {len(self.positions)}  "
            f"({self.positions[0]}..{self.positions[-1]})" if self.positions else "wells none",
            f"channels      {', '.join(self.channels)}   detect on {self.detect_channel}",
            f"z-slices      {len(self.slices)}  ({', '.join(self.slices)})   "
            f"detect {self.detect_slice}",
            f"timepoints    {len(self.timepoints)}  "
            f"(LO{min(self.timepoints):03d}..LO{max(self.timepoints):03d})"
            if self.timepoints else "timepoints none",
            f"raw frame     {self.frame_shape[1]}x{self.frame_shape[0]} {self.frame_dtype}"
            if self.frame_shape else "raw frame unknown",
            f"binning       {self.binning}x{self.binning}   [{self.binning_reason}]"
            if self.binning else f"binning       UNKNOWN  [{self.binning_reason}]",
            f"pixel size    {self.px_sensor_nm} nm sensor  ->  {u} um/px on the image"
            if u else "pixel size    UNKNOWN",
            f"interval      {self.interval_min} min  [T clock]"
            if self.interval_min else "",
            f"temperature   {self.temperature_C} C" if self.temperature_C else "",
            "",
            f"CROP          {c} px  =  {fov_mm} mm  ->  resized to {output_px}x{output_px}"
            if c else "CROP          cannot be derived (no pixel size)",
        ]
        return "\n".join(x for x in L if x)


def index_folder(raw_dir: str, read_shape: bool = True,
                 progress=None) -> RawIndex:
    """Scan a raw folder. Filenames only -- no per-file stat(), which is what
    makes this fast on a slow external/exFAT drive."""
    image_dir = find_image_dir(raw_dir)
    frames, first = {}, None
    n = 0
    with os.scandir(image_dir) as it:
        for e in it:
            if not e.name.lower().endswith(".tif"):
                continue
            rec = parse_name(e.name)
            if not rec:
                continue
            frames[(rec["pos"], rec["tp"], rec["channel"], rec["slice"])] = (
                rec, rec["well"], e.name)
            first = first or e.name
            n += 1
            if progress and n % 5000 == 0:
                progress(n)

    shape = dtype = None
    if read_shape and first:
        with tifffile.TiffFile(os.path.join(image_dir, first)) as t:
            p = t.pages[0]
            shape, dtype = tuple(p.shape), str(p.dtype)
    return RawIndex(image_dir, frames, shape, dtype)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise SystemExit("usage: acquifer.py <raw-folder>")
    print(index_folder(sys.argv[1]).summary())
