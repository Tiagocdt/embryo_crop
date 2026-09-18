# Failure modes

Every bug this tool has had, what it looked like, and why it happened. Written
because most of them were silent — the pipeline reported success and produced
unusable data — and because several are not specific to this code at all.

The recurring theme, worth stating once: **a physical quantity written as a
pixel count or a frame fraction will eventually meet data at a different scale
and be wrong without complaining.** Six separate bugs below are that same
mistake wearing different clothes.

---

## 1. Scale and geometry

### 1.1 Binning assumed rather than measured
**Symptom:** every physical size 2× too large on unbinned data.
**Cause:** `px_size_nm * 2` hard-coded, "because ACQUIFER is always 2×2 binned".
True for the EMBL plates, false for the 2048² unbinned ones.
**Fix:** binning = `NATIVE_SENSOR_PX / measured frame width`. If the frame shape
was not read, report `UNDETERMINED` rather than guessing — a blank is better
than a silent factor of two.

### 1.2 The crop was a pixel count
**Symptom:** every new optics configuration needed a special case.
**Cause:** `--crop-size 576` as the input. 576 px only means the same thing at
one pixel size.
**Fix:** the crop is a **physical field of view**; the pixel count is derived:
```
um_per_px = PX_nm/1000 * binning
crop_px   = FOV_MM*1000 / um_per_px
output    = resize(crop, OUTPUT_PX)
```
4× binned, 2× unbinned and 4× unbinned then all produce the same physical
extent and the same output dimensions, with no enumeration of cases.

### 1.3 The detection radius was a fraction of the frame
**Symptom:** **94 of 96 wells** mis-centred by >57 px on 1024² plates; centres
jumping 300+ px between timepoints. Invisible on 2048² data.
**Cause:** `egg_frac=0.23` meaning "the egg is 23% of the frame". An egg is
~1.65 mm however you image it: at 3.25 µm/px that is 23% of a 2048 frame but
**45%** of a 1024 one, so the smoothing radius was half what it should be and
the argmax settled on a sub-feature.
**Fix:** radius derived from µm/px. Re-validated against 96 reference centres:
median error unchanged where it was already right.

### 1.4 The PX token is not trustworthy, and neither is a neighbour's value
**Symptom:** AQV02/03 produced 5760 px crops on a 1024 px frame (97% black).
Then, after "fixing" it, embryos at half size and off-centre.
**Cause:** those plates carry `PX01625` (162.5 nm) where every other plate from
the same microscope carries `PX16250`. The first fix substituted another
plate's value (3.25 µm/px) — but they were imaged at a different magnification
and are actually **6.5 µm/px**. A guess dressed as a fix.
**Fix:** a crop larger than its frame is now a hard error naming the derived
µm/px and the token it came from. `--um-per-px` overrides it. **Derive the
scale by measuring the specimen, not by copying a sibling plate.**

### 1.5 `PX01625` again, on a 2048² frame
**Symptom:** OLVAS V1 (`20260828_OLVAS_V1`) derived 0.1625 µm/px and an
11520 px crop from a 2048 px frame — the hard error from 1.4 firing correctly.
**Cause:** the same bad token as AQV02/03. It appears whenever this ACQUIFER
runs the **2× objective**, whatever the binning.
**Fix:** measured the chorion in the actual frames (~490 px across a 2048²
unbinned frame ⇒ 1.6 mm / 490 px) and confirmed by cropping at that scale and
looking: **3.25 µm/px**, the same as a 4× 2×2-binned plate. `--um-per-px 3.25`.
Rule of thumb for this microscope: `PX01625` means 2×, and 2× means 3.25 µm/px
unbinned, 6.5 µm/px at 2×2.

---

## 2. Detection

### 2.1 Zero-padding creates a boundary attractor
**Symptom:** centres pinned to exactly 252 or 768 px — i.e. 63 and 192
downsampled, with kernel radius 63.
**Cause:** the box filter zero-pads, so windows near the border are diluted and
the first offset where the kernel fully fits becomes the fallback maximum.
**Measured alternatives** (96 reference wells):

| padding | wells off |
|---|---|
| **zero-pad** | **0** |
| valid-count | 20, one pinned to corner (0,0) |
| reflect | 12, same corner problem |

Zero-padding wins globally, so it stays and the artifact is **detected** rather
than eliminated: `detect_center(..., with_confidence=True)` reports peak
prominence, whether the argmax sits on the kernel boundary, and how far probe
timepoints disagree.

### 2.2 A low confidence score is not evidence of an empty well
**Symptom:** I reported two wells as empty. They contained embryos.
**Cause:** inferring "nothing there" from "my heuristic could not commit".
**Fix:** the warning now says the score describes the detector, not the
specimen; flagged wells get a second opinion from EmbryoNet
(`rescue_wells.py`), which found an embryo in every one of them.

### 2.3 Measuring a proxy and reporting it as the thing
**Symptom:** I said brightfield cropping was fine — 1.1% black pixels, matching
the previous pipeline — while 94 of 96 wells were mis-centred.
**Cause:** black-pixel fraction only detects a crop leaving the *frame*. A crop
150 px off-centre but still inside it has **no** black padding and cuts the
specimen. Proxy measured, conclusion asserted.
**Fix:** compare against ground truth — the previous pipeline's own
`centers_per_frame` — and record `crop_fill_frac` per well so off-frame crops
are visible without anyone looking.

### 2.4 One centre per well assumes the specimen cannot go far
**Symptom:** on OLVAS V1 the detected centre moved **up to 176 px inside one
run** and by a median of **145 px (max 431)** across the restart seam, against
a crop margin of only (576 − 490)/2 ≈ 43 px. A single centre clips those
embryos, and — per 2.3 — `crop_fill_frac` stays 1.0 the whole time, because
the crop never leaves the *frame*.
**Cause:** "the specimen barely moves" is true when the frame is not much
wider than the specimen. A 1024 px frame at 3.25 µm/px shows 3.3 mm of well;
a 2048 px one shows **6.7 mm** for a 1.6 mm egg, and the egg wanders in it.
The assumption was a property of the old plates, not of embryos.
**Fix:** `--detect-every N` follows the specimen — one detection every N
timepoints, each timepoint taking the nearest sample **from its own segment**
(never across a seam, where the plate was handled and the change is a step,
not a drift), median-of-3 to absorb a detection that lands on debris. The
centre used for every frame is written to `centers_per_tp`, and the distance
travelled per well to `center_travel_px`.

---

## 3. Intensity

### 3.1 Statistics over whole frames, applied to crops
**Symptom:** fluorescence clipped badly at the 99.9th percentile, which sounds
impossible.
**Cause:** the histogram accumulated whole raw frames, which are mostly empty
well (median pixel = background). "0.1% of all pixels" is a far harder cut than
"0.1% of the pixels actually written".
**Measured:** same percentile, over frames `hi=12243` with the worst well
clipping 3.9%; over crop regions `hi=21308`, worst well 2.2%, p90 well 0.00%.

### 3.2 Statistics from one z-slice, applied to all
**Cause:** `_sample_frames` filtered to the detection slice. A stack is not
uniformly bright, so that map clipped or flattened the others.

### 3.3 Sensor-saturated pixels dragging the ceiling
**Symptom:** plate `p99.9 = 65535` — the maximum itself — rendering a median
well at 37/255.
**Cause:** 3 of 25 wells over-exposed. Those pixels were clipped at capture:
their true value is unknown, so they are not measurements.
**Fix:** excluded from the histogram, count recorded as
`saturated_px_excluded` so over-exposure stays visible.

### 3.4 Per-image percentiles taken over the padding
**Symptom:** an off-frame well came out both cut **and** washed out.
**Cause:** padding is exact zeros, so it *became* the low percentile: `p_low`
collapsed to 0 and the content was squeezed into the top of the range.
**Measured** on a crop 81% real: `lo=0 hi=46710` → content std 30.4, range
67–255; real pixels only `lo=8099` → std 46.7, range 0–255.
**Fix:** `crop_at` returns the valid window; per-image stats use only it.

### 3.5 8-bit cannot hold a 5× well-to-well spread
Not a bug — arithmetic. With per-well p99.9 spanning 3,403 to 31,513, no single
8-bit map serves both. Choose `raw16` for quantitative work, or accept clipping
somewhere.

---

## 4. Robustness

### 4.1 A corrupt frame killed the plate
**Symptom:** two plates died 6 seconds in.
**Cause:** ~1% of their frames are zero bytes (writes the microscope never
finished). `tifffile` raises `TiffFileError`, which is **not** an `OSError`, so
it slipped past the retry guard — and the crash was in detection, which had no
guard at all. The plates that succeeded did so only because they had no corrupt
frames.
**Fix:** transient I/O is retried, a corrupt file is not (retrying cannot fix
zero bytes), and every read site treats it as a skipped frame. Counted
separately as `n_unreadable` — damaged source data and a flaky drive mean
different things.

### 4.2 Padding instead of shifting
The specimen (~519 px) is smaller than the crop (576 px), so a window running
off the frame can simply **slide back inside** and keep the whole specimen.
Padding threw away real image *and* poisoned the contrast stretch.

### 4.3 Partial reruns clobbering metadata
**Twice.** A `--channels CO3` run rewrote `plate_metadata.json` as though the
plate had only that channel, discarding how the others were scaled while their
images sat on disk. Later, a `--reuse-centers` re-crop dropped `center_source`
and `embryonet_rescued` — the centres survived, the record that they came from
EmbryoNet did not. **Data right, audit trail gone** is the harder kind of wrong
to notice.
**Fix:** metadata is merged, not replaced, including per-position provenance.

### 4.4 The T clock is 32-bit and it wraps
**Symptom:** OLVAS V1's first run reads `T4171231685` at LO001 and `T0000464389`
at LO070. `tp_minutes` put timepoint 1 roughly **48 days after** timepoint 70,
one gap silently dropped out of the interval median, and the run ordering used
by `--merge-runs` was decided by a number that had rolled over.
**Cause:** `T` is milliseconds in a 32-bit counter — it returns to zero every
**49.7 days** and the microscope does not reset it per acquisition. A clock
was treated as monotonic because it had never been watched for long enough.
**Fix:** `unwrap_clock()` adds a period whenever the reading drops by more than
half of one, applied to the per-timepoint median *and* inside a timepoint (a
z-stack can straddle the boundary). Runs are ordered by the timestamp in the
folder name (`260825102428_P01`), the one signal a wrap cannot corrupt, with
the unwrapped clock only as a fallback. `clock_wraps` is reported.

### 4.5 Merging refused when a well was dropped
**Symptom:** `cannot merge runs: wells differ` on a plate that was restarted
after one embryo was discontinued — 30 wells in the second run against 31 in
the first.
**Cause:** the guard existed to stop two *unrelated* experiments being glued
together, and used the well list to decide. But a well that died, hatched or
was deliberately removed is exactly what a restarted run looks like.
**Fix:** channels, z-slices and geometry still have to match; wells are taken
as a **union**, and each segment records `wells` and `wells_absent` so nothing
downstream assumes every well spans the whole time course.

---

## 5. Environment and platform

| trap | symptom | fix |
|---|---|---|
| numpy 1.x + modern tifffile | `TypeError: 'copy' is an invalid keyword` on first `imread` | require numpy>=2; tifffile does not declare it |
| EasyBuild modules export `PYTHONPATH` | a module's older numpy shadows the venv's; a venv does **not** isolate it | load a module that already has numpy≥2, or unset `PYTHONPATH` |
| SLURM copies the job script to a node spool dir | `mkdir logs` fails with *Permission denied* on a writable directory | use `$SLURM_SUBMIT_DIR`, not `${BASH_SOURCE[0]}` |
| `/tmp` is node-local | a script scp'd to the login node is invisible to the compute node | stage on shared storage |
| `tar` while files are being rewritten | `file changed as we read it`, exit 1 | never tar a tree a job is writing |
| exFAT with 1 MB allocation blocks | 332 KB crop occupies 1024 KB; 2.04 M crops need 2.14 TB instead of 0.68 | keep archives as tars, or reformat |
| streaming `tar cf - \| > file` over a network | 11 MB/s — gated by small-file reads | tar server-side first, then copy one sequential file: **58–113 MB/s** |
| `nohup` over SSH | process gone, log empty when the session ends | use a batch job |
| `ssh -f` with `-J` | hangs before the second hop completes | run the tunnel as an ordinary background process |
| stale `ControlMaster` forward | new `-L` refused while pointing at the old target | `ssh -O cancel` first |
| a server binding `127.0.0.1` on a compute node | one `-L` to the login node reaches a closed door | two hops; keep the loopback bind |
| `ControlPersist 8h` | SSH silently stops working after an idle stretch | one interactive `ssh <host>` |

---

## 6. Process failures (mine)

These produced no bad data but cost real time, and they are the ones most worth
not repeating.

- **A string replace that matched twice.** The anchor existed in two functions;
  `str.replace` hit both, broke indentation, and the file no longer parsed. It
  was committed, pushed, deployed and a job launched **before the syntax error
  was read**. Always assert the match count.
- **A verification grep narrow enough to hide the failure.** `grep -E "crop
  |wrote"` on a run that had already crashed showed the "crop" line and no
  error. The test had failed and was read as passing. Verify on unfiltered
  output.
- **Adding a return value without updating every call site.** `crop_at` grew a
  third return and `idx.frames` a fourth element; both broke jobs *twice*
  because I fixed the site I remembered instead of grepping for all of them.
- **A field referenced but never defined.** `st.min_score` existed only in the
  call. It fired solely when a plate had doubtful wells, so every earlier run
  skipped the branch.
- **`rm -rf` on an unexamined path.** `"/scratch/$USER"` where `$USER` had not
  expanded — a real directory belonging to another group. Nothing was lost only
  because their permissions refused it. **List and stat a target before
  deleting it**, and prefer moving to a trash folder over removing.

---

## What actually catches these

Ranked by how much they have caught here:

1. **Comparing against independent ground truth** — the previous pipeline's
   recorded centres, or the user's own annotations. Every serious detection bug
   was found this way and none by internal consistency checks.
2. **Looking at the images.** Two bugs survived numeric checks and died
   instantly on a contact sheet.
3. **Recording per-well diagnostics** — `crop_fill_frac`,
   `detection_confidence`, `saturated_px_excluded`. They make a silent failure
   visible without anyone thinking to look.
4. **Running the real path on real data with unfiltered output** before
   launching anything long.
5. **Believing the person who looked at the data.** When told the crops were
   wrong, the right move was to go and measure against ground truth, not to
   defend a proxy metric.
