# embryo_crop

Turn a raw ACQUIFER plate acquisition into per-embryo crops: detect every
embryo, cut a fixed **physical** field of view around it, scale the intensities
by a map you choose deliberately, and write the result into a folder structure
you define.

Four files, about 1,200 lines, and two dependencies (numpy and tifffile).

```
acquifer.py    read a raw folder and say what is in it        ~250 lines
process.py     the seven steps that touch a pixel             ~420 lines
resources.py   how much of the machine a run may use          ~110 lines
gui.py         the window, the job queue, the presets         ~430 lines
```

---

## Everything that happens to an image

```
1. parse filename      well, timepoint, channel, z-slice
2. pick detect slice   the middle of the z-stack
3. detect the embryo   a few times per well, not per frame
4. crop                a fixed PHYSICAL field of view
5. resize              always the same output pixel size
6. rescale intensity   16-bit counts -> 8-bit, by one chosen map
7. write               your folders, your filenames
```

Nothing else. No sharpening, no denoising, no gamma, no silent contrast
changes. Every parameter that was used is written to `plate_metadata.json`
beside the images.

## The one idea worth understanding

The crop is a **physical size**, never a pixel count:

```
um_per_px = PX_token_nm / 1000 * binning    # binning is MEASURED, not assumed
crop_px   = FOV_MM * 1000 / um_per_px
output    = resize(crop, OUTPUT_PX)
```

Because the crop is derived from the physical scale, any objective and any
binning setting produce images of the same physical extent **and** the same
pixel dimensions. Two illustrative cases:

| frame | `PX` | binning | µm/px | crop | output |
|---|---|---|---|---|---|
| 1024² | 16250 | 2×2 | 3.25 | 576 px | 576 (no resize) |
| 2048² | 65000 | 1×1 | 6.50 | 288 px | 576 (upscaled ×2) |

There is no table of supported combinations, because there is no enumeration —
the formula covers whatever your microscope wrote, including magnifications and
binning factors this tool has never seen. What matters is that `PX` is the
sensor pixel size and binning is derived from `sensor px ÷ measured frame
width`, rather than assumed. Hard-coding a binning factor silently scales every
physical measurement by that factor when the assumption is wrong.

`FOV_MM` (default 1.872 mm) and `OUTPUT_PX` (default 576) define your dataset.
Changing them makes a *new* dataset rather than reprocessing an old one, so they
are deliberately dull constants at the top of `acquifer.py`. Set them once to
whatever your downstream analysis expects, then leave them alone.

## Detection

The egg is the only textured object in an otherwise flat well, so its centre is
the peak of gradient energy smoothed over roughly an egg radius. That is about
fifteen lines of numpy — no model file, no GPU, nothing to download.

Validated against a TensorFlow object-detection model (EmbryoNet) across all
**96 wells** of one plate:

| | error vs the ML model |
|---|---|
| median | 9 px |
| p90 | 18 px |
| worst | 28 px |
| within 40 px | 100 % |

On a 2048 px frame where the egg is ~480 px across and the crop is 576 px, a
28 px worst case is well inside the margin. Cost is a fraction of a second per
well.

Detection runs a few times per well and the median centre is reused for that
well's whole time course, so a drifting embryo stays centred without re-running
detection on every frame. If your specimens are not round, high-contrast objects
on a plain background, this heuristic is the first thing to re-check — swap
`detect_center()` in `process.py` for whatever suits your images; everything
else is independent of how the centre was found.

## Intensity scaling — choose the scope on purpose

This is the only real decision in the tool.

| mode | statistics from | comparable across | good for |
|---|---|---|---|
| `raw16` | nothing; counts kept | everything, forever | quantitative fluorescence |
| `fixed` | constants you supply | plates and experiments | multi-plate studies |
| `plate` | the whole plate, per channel | wells + timepoints | fluorescence within a plate |
| `well` | one well's own movie | timepoints of that well | uneven illumination between wells |
| `image` | that one frame alone | **nothing** | brightfield viewing, morphology |

Percentiles (default 1 % / 99.9 %) rather than min–max, because a single hot
pixel sets a min–max range.

**Which to pick.** For *fluorescence you intend to measure*, use `raw16`, or
`plate` if you need 8-bit. Never `image` — it rescales every frame to look good
and destroys exactly the signal you are trying to quantify.

For *brightfield*, `image` is often the right answer and `plate` can be worse in
practice: transmitted-light illumination varies from well to well, so one
plate-wide map can leave many specimens looking flat. That variation is an
instrument artefact rather than biology, and normalising it away per frame is
legitimate. The cost is that brightfield pixel values are then not comparable
between frames — usually fine, because you were not going to measure them.

`well` sits between the two: it removes the between-well artefact while keeping
each specimen's time course internally comparable.

Whatever you choose is recorded in `plate_metadata.json` along with the
resulting low/high per channel, so any crop can be traced back — and inverted to
counts when the map is constant (`raw = value/255 * (hi-lo) + lo`).

## Install

```bash
git clone https://github.com/<you>/embryo_crop.git
cd embryo_crop
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

The GUI also needs **tkinter**, which is part of the standard library but is
packaged separately by some distributions. If `import tkinter` fails:

| platform | fix |
|---|---|
| macOS (Homebrew) | `brew install python-tk@3.X` (match your Python version) |
| Debian / Ubuntu | `sudo apt install python3-tk` |
| Fedora | `sudo dnf install python3-tkinter` |
| Windows / python.org | already included |

The command-line path (`process.py`) does not need tkinter.

## Use it

Window:

```bash
./.venv/bin/python gui.py
```

Pick a raw folder and press **Probe**. Everything the machine can know is read
from the data and shown to you — resolution, binning, µm/px, channels,
z-slices, timepoints, and the resulting crop size in pixels *and* millimetres.
You confirm or override; you never type what the files already say. Jobs go in a
queue and run one at a time so two runs never fight over the same disk. Settings
save and reload as JSON.

Command line:

```bash
./.venv/bin/python process.py RAW_DIR OUT_DIR --plate NAME --scaling image
```

Useful flags:

```
--channels CO1,CO2       default: all
--wells A01,A02          default: all
--scaling MODE           plate | well | image | fixed | raw16
--low-pct / --high-pct   percentiles (default 1.0 / 99.9)
--fixed-lo / --fixed-hi  required when --scaling fixed
--fov-mm / --output-px   the physical crop and the output size
--workers N              parallel readers
--stats-sample N         frames sampled for plate statistics; 0 = every frame
--metadata-only          rewrite plate_metadata.json, touch no images
--overwrite              redo files that already exist
```

Inspect without processing anything:

```bash
./.venv/bin/python acquifer.py RAW_DIR      # what is in this folder?
./.venv/bin/python resources.py RAW_DIR     # how many workers should I use?
```

## Output

```
OUT_DIR/<plate>/<channel>/<well>/<z-slice>/<plate>_<well>_LO###_<channel>_<slice>.tif
OUT_DIR/<plate>/plate_metadata.json
OUT_DIR/<plate>_manifest.json
```

Both templates are editable (`{plate} {channel} {pos} {well} {slice} {tp}`), so
you can match whatever layout your downstream tools expect.

`plate_metadata.json` sits at the plate root and carries the calibration,
detected centres, binning, µm/px, crop size and timepoint interval.
`<plate>_manifest.json` additionally records throughput and any frames that
failed.

## Robustness

- **Retries** — several attempts with backoff on every read and write, so a
  removable drive that briefly drops off the bus costs seconds rather than the
  whole plate.
- **Atomic writes** — files are written to `.part` and renamed, so a
  half-written file from an interrupted run cannot masquerade as finished on the
  next pass.
- **Resume** — re-running skips everything already written. A plate that stopped
  halfway continues from where it stopped; failures are reported, not fatal.

## Performance, and the one trap worth knowing

The work is **I/O plus a little numpy**, so throughput tracks your *storage*,
not your CPU. Detection runs a handful of times per well rather than per frame,
so it never dominates. In practice that means:

- reading and writing on fast local storage is roughly an order of magnitude
  faster than working across a slow external or network volume;
- adding workers helps until the disk saturates, then stops helping;
- indexing is effectively free — a 90,000-frame folder is read in a couple of
  seconds, because the indexer parses filenames and never calls `stat()` per
  file.

`resources.py` sizes the worker pool from the machine and the frame size, since
workers-in-flight sets both CPU and peak memory. Run it against your own data
rather than trusting a number measured elsewhere:

```bash
./.venv/bin/python resources.py RAW_DIR
```

**The trap: filesystem allocation size.** A 576² uint8 crop is ~332 KB, but each
file consumes a whole allocation block, and these pipelines produce *hundreds of
thousands of small files*:

| allocation block | on-disk per crop | overhead |
|---|---|---|
| 4 KB (APFS, ext4, NTFS) | 332 KB | ~0 % |
| 128 KB | 384 KB | 16 % |
| 1 MB (common on large exFAT volumes) | **1,024 KB** | **208 %** |

A 90,000-frame plate is 29 GB of image data but can occupy 88 GB on a large
exFAT volume. Check before planning space — `diskutil info /Volumes/NAME` on
macOS, `stat -f` or `tune2fs -l` on Linux.

## Related

[**PlateNotate**](https://github.com/Tiagocdt/platenotate) — annotate the crops
this tool produces: plate, well and frame level, then export what you annotated
as a movie, montage or TIFF hyperstack. `embryo_crop` writes
`plate_metadata.json` in the layout PlateNotate discovers, so point PlateNotate
at the folder *containing* your plate folders and they appear in its plate list.

## License

MIT — see `LICENSE`.
