#!/usr/bin/env python3
"""Re-detect the wells the geometric detector was not confident about, using
EmbryoNet, and write the corrected centres back into plate_metadata.json.

    python3 rescue_wells.py <plate-dir> [--raw RAW_DIR] [--dry-run]

Then re-crop just those wells, reusing the corrected centres:

    sbatch cluster_job.sh RAW OUT PLATE --reuse-centers --wells A04,C08 --overwrite

WHY THIS IS A SEPARATE STEP
---------------------------
EmbryoNet needs TensorFlow, and on this cluster the TF module ships numpy 1.25
while the pipeline's venv wants numpy 2. Rather than fight two module stacks in
one process, the rescue runs on its own under the TF module -- it only needs to
read frames and write JSON. The crop pass then runs normally.

A doubtful score means the geometric heuristic could not commit. It is NOT
evidence that a well is empty. This asks a model that was trained on these
specimens instead of guessing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plate_dir", help="the processed plate dir (holds plate_metadata.json)")
    ap.add_argument("--raw", default=None, help="raw dir; default: read from the metadata")
    ap.add_argument("--model", default=None)
    ap.add_argument("--min-score", type=float, default=0.1)
    ap.add_argument("--wells", default="", help="override which wells to redo")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    import numpy as np
    import tifffile
    from acquifer import index_folder, index_folders, find_image_dirs

    pm_path = os.path.join(a.plate_dir, "plate_metadata.json")
    if not os.path.exists(pm_path):
        raise SystemExit(f"no plate_metadata.json in {a.plate_dir}")
    pm = json.load(open(pm_path))

    wells = [w for w in a.wells.split(",") if w]
    if not wells:
        wells = sorted(
            p for p, v in (pm.get("positions") or {}).items()
            if (v.get("detection") or {}) and (
                (v["detection"].get("prominence") or 9) < 2.0
                or v["detection"].get("on_kernel_edge")
                or (v["detection"].get("timepoint_spread_px") or 0) > 120))
    if not wells:
        print("no doubtful wells recorded — nothing to do")
        return
    print(f"{len(wells)} well(s) to re-detect: {', '.join(wells)}")

    raw = a.raw or pm.get("raw_dir")
    if not raw or not os.path.isdir(raw):
        raise SystemExit(f"raw dir not usable: {raw}")
    dirs = find_image_dirs(os.path.dirname(raw)) if pm.get("segments") else [raw]
    idx = index_folders(dirs) if len(dirs) > 1 else index_folder(raw)

    from embryonet import EmbryoNet
    net = EmbryoNet(a.model, a.min_score)
    print(f"EmbryoNet on {'GPU' if net.on_gpu else 'CPU'}")

    dch = pm.get("bf_channel") or idx.detect_channel
    dsl = pm.get("detect_slice") or idx.detect_slice
    tps = idx.timepoints
    probe = [tps[0], tps[len(tps) // 2], tps[-1]] if len(tps) > 2 else tps

    changed = {}
    for w in wells:
        pts = []
        for tp in probe:
            k = (w, tp, dch, dsl)
            if k not in idx.frames:
                continue
            try:
                yx = net.detect(tifffile.imread(idx.path(k)))
            except Exception as e:                        # noqa: BLE001
                print(f"  {w}: read/detect failed at tp{tp}: {e}")
                continue
            if yx:
                pts.append(yx)
        if not pts:
            print(f"  {w}: EmbryoNet found nothing either — leaving as is")
            continue
        new = (int(np.median([p[0] for p in pts])), int(np.median([p[1] for p in pts])))
        old = (pm.get("positions", {}).get(w) or {}).get("center_yx")
        moved = float(np.hypot(new[0] - old[0], new[1] - old[1])) if old else 0.0
        print(f"  {w}: {old} -> {list(new)}   moved {moved:.0f}px   from {len(pts)} frame(s)")
        changed[w] = {"was": old, "now": list(new), "moved_px": round(moved, 1)}
        pm.setdefault("positions", {}).setdefault(w, {})["center_yx"] = list(new)
        pm["positions"][w]["center_source"] = "embryonet"

    if not changed:
        print("nothing changed")
        return
    if a.dry_run:
        print(f"\nDRY RUN — {len(changed)} well(s) would change; nothing written")
        return

    pm["embryonet_rescued"] = {**(pm.get("embryonet_rescued") or {}), **changed}
    tmp = pm_path + ".part"
    with open(tmp, "w") as fh:
        json.dump(pm, fh, indent=2)
    os.replace(tmp, pm_path)
    print(f"\nupdated {pm_path} for {len(changed)} well(s)")
    print("now re-crop just those wells:")
    print(f"  sbatch cluster_job.sh <RAW> <OUT> <PLATE> --reuse-centers "
          f"--wells {','.join(sorted(changed))} --overwrite")


if __name__ == "__main__":
    main()
