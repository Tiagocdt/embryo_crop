#!/usr/bin/env python3
"""A small window to set up and run crop jobs. tkinter only -- no install.

    python3 gui.py

Pick a raw folder, press Probe. Everything the machine can know is READ from
the data and shown to you: resolution, binning, um/px, channels, z-slices,
timepoints, and the resulting crop size in pixels and millimetres. You confirm
or override; you never have to type what the files already say.

Jobs go in a queue and run one after another so two runs never fight over the
disk. Settings save to JSON and reload.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import acquifer
import process
import resources

PRESET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets")


class App:
    def __init__(self, root):
        self.root = root
        root.title("embryo_crop")
        root.geometry("1080x820")

        self.idx = None
        self.msgs = queue.Queue()
        self.jobs = []
        self.running = False
        self.stop_flag = False

        self._build()
        self.root.after(100, self._pump)

    # -- layout ----------------------------------------------------------
    def _build(self):
        pad = {"padx": 6, "pady": 3}
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True)
        setup = ttk.Frame(nb)
        runf = ttk.Frame(nb)
        nb.add(setup, text="Setup")
        nb.add(runf, text="Queue && Progress")

        # ---------- folders ----------
        f = ttk.LabelFrame(setup, text="1. folders")
        f.pack(fill="x", **pad)
        self.raw = tk.StringVar()
        self.out = tk.StringVar()
        self.plate = tk.StringVar()
        for label, var, cb in (("raw folder", self.raw, self._pick_raw),
                               ("output folder", self.out, self._pick_out)):
            r = ttk.Frame(f); r.pack(fill="x", **pad)
            ttk.Label(r, text=label, width=14).pack(side="left")
            ttk.Entry(r, textvariable=var).pack(side="left", fill="x", expand=True)
            ttk.Button(r, text="...", width=4, command=cb).pack(side="left", padx=4)
        r = ttk.Frame(f); r.pack(fill="x", **pad)
        ttk.Label(r, text="plate name", width=14).pack(side="left")
        ttk.Entry(r, textvariable=self.plate, width=40).pack(side="left")
        ttk.Button(r, text="Probe raw folder", command=self._probe).pack(side="left", padx=10)

        # ---------- what is in there ----------
        f2 = ttk.LabelFrame(setup, text="2. what the data says (read, not typed)")
        f2.pack(fill="both", **pad)
        self.info = tk.Text(f2, height=11, wrap="none",
                            font=("Menlo", 11))
        self.info.pack(fill="both", expand=True, padx=6, pady=4)

        # ---------- optics / crop ----------
        f3 = ttk.LabelFrame(setup, text="3. optics and crop")
        f3.pack(fill="x", **pad)
        r = ttk.Frame(f3); r.pack(fill="x", **pad)
        self.fov = tk.StringVar(value=str(acquifer.DEFAULT_FOV_MM))
        self.outpx = tk.StringVar(value=str(acquifer.DEFAULT_OUTPUT_PX))
        self.umpx = tk.StringVar(value="")
        ttk.Label(r, text="field of view (mm)").pack(side="left")
        e = ttk.Entry(r, textvariable=self.fov, width=8); e.pack(side="left", padx=4)
        ttk.Label(r, text="output px").pack(side="left", padx=(12, 0))
        ttk.Entry(r, textvariable=self.outpx, width=8).pack(side="left", padx=4)
        ttk.Label(r, text="um/px override").pack(side="left", padx=(12, 0))
        ttk.Entry(r, textvariable=self.umpx, width=8).pack(side="left", padx=4)
        for v in (self.fov, self.outpx, self.umpx):
            v.trace_add("write", lambda *_: self._show_crop())
        self.cropinfo = ttk.Label(r, text="", font=("Menlo", 12, "bold"))
        self.cropinfo.pack(side="left", padx=16)

        # ---------- channels ----------
        f4 = ttk.LabelFrame(setup, text="4. channels and z-slices")
        f4.pack(fill="x", **pad)
        self.chvars, self.slvars = {}, {}
        self.chbox = ttk.Frame(f4); self.chbox.pack(fill="x", **pad)
        self.slbox = ttk.Frame(f4); self.slbox.pack(fill="x", **pad)

        # ---------- scaling ----------
        f5 = ttk.LabelFrame(setup, text="5. intensity scaling")
        f5.pack(fill="x", **pad)
        r = ttk.Frame(f5); r.pack(fill="x", **pad)
        self.scaling = tk.StringVar(value="plate")
        ttk.Label(r, text="default mode").pack(side="left")
        cb = ttk.Combobox(r, textvariable=self.scaling, width=10,
                          state="readonly", values=list(process.SCALING_MODES))
        cb.pack(side="left", padx=4)
        cb.bind("<<ComboboxSelected>>", lambda *_: self._scaling_help())
        self.lowp = tk.StringVar(value="1.0")
        self.highp = tk.StringVar(value="99.9")
        ttk.Label(r, text="low %").pack(side="left", padx=(12, 0))
        ttk.Entry(r, textvariable=self.lowp, width=6).pack(side="left", padx=4)
        ttk.Label(r, text="high %").pack(side="left")
        ttk.Entry(r, textvariable=self.highp, width=6).pack(side="left", padx=4)
        self.flo = tk.StringVar(value="")
        self.fhi = tk.StringVar(value="")
        ttk.Label(r, text="fixed lo/hi").pack(side="left", padx=(12, 0))
        ttk.Entry(r, textvariable=self.flo, width=7).pack(side="left", padx=2)
        ttk.Entry(r, textvariable=self.fhi, width=7).pack(side="left", padx=2)
        self.sample = tk.StringVar(value="400")
        ttk.Label(r, text="stats frames (0=all)").pack(side="left", padx=(12, 0))
        ttk.Entry(r, textvariable=self.sample, width=7).pack(side="left", padx=4)
        self.schelp = ttk.Label(f5, text="", foreground="#555")
        self.schelp.pack(anchor="w", padx=10)
        self._scaling_help()

        # ---------- output structure ----------
        f6 = ttk.LabelFrame(setup, text="6. output structure  "
                                        "(fields: plate channel pos well slice tp)")
        f6.pack(fill="x", **pad)
        self.dirt = tk.StringVar(value="{plate}/{channel}/{pos}/{slice}")
        self.filet = tk.StringVar(value="{plate}_{pos}_LO{tp:03d}_{channel}_{slice}.tif")
        for lab, var in (("folders", self.dirt), ("filename", self.filet)):
            r = ttk.Frame(f6); r.pack(fill="x", **pad)
            ttk.Label(r, text=lab, width=10).pack(side="left")
            ttk.Entry(r, textvariable=var).pack(side="left", fill="x", expand=True)

        # ---------- compute ----------
        f7 = ttk.LabelFrame(setup, text="7. how much of the laptop to use")
        f7.pack(fill="x", **pad)
        r = ttk.Frame(f7); r.pack(fill="x", **pad)
        self.level = tk.StringVar(value="balanced")
        ttk.Label(r, text="load").pack(side="left")
        lb = ttk.Combobox(r, textvariable=self.level, width=10, state="readonly",
                          values=["gentle", "balanced", "all-out", "manual"])
        lb.pack(side="left", padx=4)
        lb.bind("<<ComboboxSelected>>", lambda *_: self._apply_level())
        self.workers = tk.StringVar(value="8")
        ttk.Label(r, text="workers").pack(side="left", padx=(12, 0))
        ttk.Entry(r, textvariable=self.workers, width=6).pack(side="left", padx=4)
        self.overwrite = tk.BooleanVar(value=False)
        ttk.Checkbutton(r, text="overwrite existing",
                        variable=self.overwrite).pack(side="left", padx=12)
        self.resinfo = ttk.Label(f7, text=resources.describe(), foreground="#555")
        self.resinfo.pack(anchor="w", padx=10)

        # ---------- buttons ----------
        r = ttk.Frame(setup); r.pack(fill="x", **pad)
        ttk.Button(r, text="Save preset", command=self._save).pack(side="left", padx=4)
        ttk.Button(r, text="Load preset", command=self._load).pack(side="left", padx=4)
        ttk.Button(r, text="Cluster command…",
                   command=self._show_sbatch).pack(side="left", padx=16)
        ttk.Button(r, text="Add to queue →", command=self._add).pack(side="right", padx=4)

        # ---------- queue tab ----------
        qf = ttk.LabelFrame(runf, text="queue")
        qf.pack(fill="x", **pad)
        self.qlist = tk.Listbox(qf, height=6, font=("Menlo", 11))
        self.qlist.pack(fill="x", padx=6, pady=4)
        r = ttk.Frame(qf); r.pack(fill="x", **pad)
        ttk.Button(r, text="Remove selected", command=self._rm).pack(side="left", padx=4)
        self.runbtn = ttk.Button(r, text="RUN QUEUE", command=self._run)
        self.runbtn.pack(side="left", padx=4)
        ttk.Button(r, text="Stop after current frame",
                   command=self._stop).pack(side="left", padx=4)

        pf = ttk.LabelFrame(runf, text="progress")
        pf.pack(fill="x", **pad)
        self.bar = ttk.Progressbar(pf, maximum=100)
        self.bar.pack(fill="x", padx=6, pady=4)
        self.status = ttk.Label(pf, text="idle", font=("Menlo", 11))
        self.status.pack(anchor="w", padx=8)

        lf = ttk.LabelFrame(runf, text="log")
        lf.pack(fill="both", expand=True, **pad)
        self.log = tk.Text(lf, wrap="none", font=("Menlo", 10))
        self.log.pack(fill="both", expand=True, padx=6, pady=4)

    # -- helpers ---------------------------------------------------------
    def _pick_raw(self):
        d = filedialog.askdirectory(title="raw ACQUIFER folder")
        if d:
            self.raw.set(d)
            if not self.plate.get():
                self.plate.set(os.path.basename(os.path.normpath(d)))
            self._probe()

    def _pick_out(self):
        d = filedialog.askdirectory(title="output folder")
        if d:
            self.out.set(d)

    def _probe(self):
        d = self.raw.get()
        if not d:
            return
        self.info.delete("1.0", "end")
        self.info.insert("end", f"indexing {d} ...\n")
        self.root.update_idletasks()
        try:
            self.idx = acquifer.index_folder(d)
        except Exception as e:
            self.info.delete("1.0", "end")
            self.info.insert("end", f"ERROR: {e}\n")
            return
        self.info.delete("1.0", "end")
        self.info.insert("end", self.idx.summary(
            self._f(self.fov, acquifer.DEFAULT_FOV_MM),
            int(self._f(self.outpx, acquifer.DEFAULT_OUTPUT_PX))))
        self._checkboxes()
        self._show_crop()
        self.resinfo.config(text=resources.describe(self.idx.frame_shape))
        self._apply_level()

    def _checkboxes(self):
        self.chmodes = {}
        for box, store, items, label in (
                (self.chbox, self.chvars, self.idx.channels, "channels"),
                (self.slbox, self.slvars, self.idx.slices, "z-slices")):
            for w in box.winfo_children():
                w.destroy()
            store.clear()
            ttk.Label(box, text=label, width=10).pack(side="left")
            allv = tk.BooleanVar(value=True)
            store["__all__"] = allv

            def toggle_all(s=store, a=allv):
                for k, v in s.items():
                    if k != "__all__":
                        v.set(a.get())
            ttk.Checkbutton(box, text="ALL", variable=allv,
                            command=toggle_all).pack(side="left", padx=6)
            for it in items:
                v = tk.BooleanVar(value=True)
                store[it] = v
                ttk.Checkbutton(box, text=it, variable=v).pack(side="left", padx=3)
                # each CHANNEL also gets its own scaling mode: brightfield and
                # fluorescence want opposite maps and live in the same plate.
                if store is self.chvars:
                    dflt = ("image" if it == self.idx.detect_channel else "plate")
                    m = tk.StringVar(value=dflt)
                    self.chmodes[it] = m
                    cb = ttk.Combobox(box, textvariable=m, width=7,
                                      state="readonly",
                                      values=list(process.SCALING_MODES))
                    cb.pack(side="left", padx=(0, 10))
                    if it == self.idx.detect_channel:
                        ttk.Label(box, text="(BF)",
                                  foreground="#777").pack(side="left", padx=(0, 8))

    def _selected(self, store, all_items):
        picked = [k for k, v in store.items() if k != "__all__" and v.get()]
        return [] if len(picked) == len(all_items) else picked

    def _f(self, var, default):
        try:
            return float(var.get())
        except (ValueError, tk.TclError):
            return default

    def _show_crop(self):
        if not self.idx:
            return
        um = self._f(self.umpx, 0) or self.idx.um_per_px
        fov = self._f(self.fov, acquifer.DEFAULT_FOV_MM)
        outp = int(self._f(self.outpx, acquifer.DEFAULT_OUTPUT_PX))
        if not um:
            self.cropinfo.config(text="um/px unknown - set override")
            return
        c = int(round(fov * 1000.0 / um))
        scale = outp / c if c else 1
        self.cropinfo.config(
            text=f"crop {c}px @ {um} um/px = {fov} mm  →  {outp}px "
                 f"({'no resize' if abs(scale-1) < .01 else f'x{scale:.2f}'})")

    def _scaling_help(self):
        m = self.scaling.get()
        t = {
            "plate": "percentiles over the WHOLE plate, one map per channel. "
                     "Comparable across wells and timepoints. Recommended.",
            "well": "percentiles per well over its movie. Comparable across "
                    "time, NOT across wells.",
            "image": "percentiles of each frame alone = AUTO-CONTRAST. "
                     "Comparable with nothing. Only for looking, never measuring.",
            "fixed": "one constant LO:HI you supply. Comparable across plates.",
            "raw16": "no rescaling at all, uint16 counts kept. Fully "
                     "quantitative; files are 2x bigger.",
        }.get(m, "")
        self.schelp.config(text=t)

    def _apply_level(self):
        lvl = self.level.get()
        if lvl == "manual":
            return
        cf, mf = {"gentle": (0.35, 0.15), "balanced": (0.6, 0.25),
                  "all-out": (0.95, 0.5)}[lvl]
        shape = self.idx.frame_shape if self.idx else None
        r = resources.recommend(shape, cpu_fraction=cf, memory_fraction=mf)
        self.workers.set(str(r["workers"]))
        self.resinfo.config(
            text=f"{r['cores']} cores · {r['ram_gb']} GB RAM · "
                 f"{r['workers']} workers · peak ~{r['est_peak_mb']:.0f} MB "
                 f"(limited by {r['limited_by']})")

    # -- settings <-> dict ----------------------------------------------
    def _settings(self):
        return dict(
            raw_dir=self.raw.get(), out_dir=self.out.get(),
            plate=self.plate.get(),
            channels=self._selected(self.chvars, self.idx.channels) if self.idx else [],
            slices=self._selected(self.slvars, self.idx.slices) if self.idx else [],
            fov_mm=self._f(self.fov, acquifer.DEFAULT_FOV_MM),
            output_px=int(self._f(self.outpx, acquifer.DEFAULT_OUTPUT_PX)),
            um_per_px_override=(self._f(self.umpx, 0) or None),
            scaling=self._scaling_spec(),
            low_pct=self._f(self.lowp, 1.0), high_pct=self._f(self.highp, 99.9),
            fixed_lo=(self._f(self.flo, 0) or None),
            fixed_hi=(self._f(self.fhi, 0) or None),
            stats_sample=int(self._f(self.sample, 400)),
            dir_template=self.dirt.get(), file_template=self.filet.get(),
            workers=int(self._f(self.workers, 8)),
            overwrite=self.overwrite.get())

    def _scaling_spec(self):
        """One mode, or "default,CH=mode,..." when channels differ."""
        modes = getattr(self, "chmodes", {})
        if not modes:
            return self.scaling.get()
        vals = {c: v.get() for c, v in modes.items()}
        if len(set(vals.values())) == 1:
            return next(iter(vals.values()))
        # most common mode becomes the default, the rest are overrides
        from collections import Counter
        default = Counter(vals.values()).most_common(1)[0][0]
        parts = [default] + [f"{c}={m}" for c, m in sorted(vals.items())
                             if m != default]
        return ",".join(parts)

    def _sbatch_cmd(self):
        """The exact cluster command for the settings currently on screen."""
        s = self._settings()
        q = lambda v: f"'{v}'" if (" " in str(v) or not str(v)) else str(v)
        parts = ["sbatch cluster_job.sh", q(s["raw_dir"]), q(s["out_dir"]),
                 q(s["plate"] or "PLATE"), f"--scaling {q(s['scaling'])}"]
        if s.get("channels"):
            parts.append("--channels " + ",".join(s["channels"]))
        if s.get("positions"):
            parts.append("--wells " + ",".join(s["positions"]))
        if s["low_pct"] != 1.0:
            parts.append(f"--low-pct {s['low_pct']}")
        if s["high_pct"] != 99.9:
            parts.append(f"--high-pct {s['high_pct']}")
        if s.get("fixed_lo") is not None:
            parts.append(f"--fixed-lo {s['fixed_lo']} --fixed-hi {s['fixed_hi']}")
        if s["fov_mm"] != acquifer.DEFAULT_FOV_MM:
            parts.append(f"--fov-mm {s['fov_mm']}")
        if s["output_px"] != acquifer.DEFAULT_OUTPUT_PX:
            parts.append(f"--output-px {s['output_px']}")
        if s["stats_sample"] != 400:
            parts.append(f"--stats-sample {s['stats_sample']}")
        if s["overwrite"]:
            parts.append("--overwrite")
        cmd = " \\\n    ".join(parts)
        return (
            "# The paths below are LOCAL. Edit them to the server paths before\n"
            "# running -- the cluster cannot see your laptop's drives.\n"
            "ssh embl\n"
            "cd /g/aulehla/Tiago/embryo_crop\n"
            f"{cmd}\n\n"
            "# --workers comes from --cpus-per-task in cluster_job.sh, so it is\n"
            "# not passed here. One job, no array: the work is I/O-bound.\n")

    def _show_sbatch(self):
        w = tk.Toplevel(self.root)
        w.title("cluster command")
        w.geometry("880x340")
        ttk.Label(w, text="Copy this to the cluster. Paths are local — edit them.",
                  padding=8).pack(anchor="w")
        t = tk.Text(w, wrap="none", font=("Menlo", 11))
        t.pack(fill="both", expand=True, padx=8, pady=4)
        t.insert("1.0", self._sbatch_cmd())
        def copy():
            self.root.clipboard_clear()
            self.root.clipboard_append(t.get("1.0", "end-1c"))
        ttk.Button(w, text="Copy to clipboard", command=copy).pack(pady=6)

    def _save(self):
        os.makedirs(PRESET_DIR, exist_ok=True)
        p = filedialog.asksaveasfilename(
            initialdir=PRESET_DIR, defaultextension=".json",
            filetypes=[("preset", "*.json")])
        if p:
            json.dump(self._settings(), open(p, "w"), indent=2)
            messagebox.showinfo("saved", p)

    def _load(self):
        os.makedirs(PRESET_DIR, exist_ok=True)
        p = filedialog.askopenfilename(initialdir=PRESET_DIR,
                                       filetypes=[("preset", "*.json")])
        if not p:
            return
        s = json.load(open(p))
        self.raw.set(s.get("raw_dir", "")); self.out.set(s.get("out_dir", ""))
        self.plate.set(s.get("plate", ""))
        self.fov.set(str(s.get("fov_mm", acquifer.DEFAULT_FOV_MM)))
        self.outpx.set(str(s.get("output_px", acquifer.DEFAULT_OUTPUT_PX)))
        self.umpx.set(str(s.get("um_per_px_override") or ""))
        self.scaling.set(s.get("scaling", "plate"))
        self.lowp.set(str(s.get("low_pct", 1.0)))
        self.highp.set(str(s.get("high_pct", 99.9)))
        self.sample.set(str(s.get("stats_sample", 400)))
        self.flo.set(str(s.get("fixed_lo") or ""))
        self.fhi.set(str(s.get("fixed_hi") or ""))
        self.dirt.set(s.get("dir_template", "")); self.filet.set(s.get("file_template", ""))
        self.workers.set(str(s.get("workers", 8)))
        self.overwrite.set(bool(s.get("overwrite", False)))
        self.level.set("manual")
        if self.raw.get():
            self._probe()
        self._scaling_help()

    # -- queue -----------------------------------------------------------
    def _add(self):
        s = self._settings()
        if not s["raw_dir"] or not s["out_dir"]:
            messagebox.showerror("missing", "set a raw folder and an output folder")
            return
        self.jobs.append(s)
        self.qlist.insert("end",
                          f"{s['plate'] or os.path.basename(s['raw_dir'])}  "
                          f"[{s['scaling']}]  {s['workers']}w  -> {s['out_dir']}")

    def _rm(self):
        for i in reversed(self.qlist.curselection()):
            self.qlist.delete(i)
            del self.jobs[i]

    def _stop(self):
        self.stop_flag = True
        self.msgs.put(("log", "stop requested; finishing current frame ...\n"))

    def _run(self):
        if self.running or not self.jobs:
            return
        self.running = True
        self.stop_flag = False
        self.runbtn.config(state="disabled")
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        jobs = list(self.jobs)
        for n, s in enumerate(jobs, 1):
            if self.stop_flag:
                break
            self.msgs.put(("log", f"\n===== job {n}/{len(jobs)}: "
                                  f"{s.get('plate')} =====\n"))
            try:
                resources.apply_limits(nice_level=5)
                process.run(
                    process.Settings(**s),
                    log=lambda m: self.msgs.put(("log", str(m) + "\n")),
                    progress=lambda d, t, m: self.msgs.put(("prog", (d, t, m))),
                    should_stop=lambda: self.stop_flag)
            except Exception as e:
                self.msgs.put(("log", f"ERROR: {e}\n"))
        self.msgs.put(("done", None))

    def _pump(self):
        try:
            while True:
                kind, payload = self.msgs.get_nowait()
                if kind == "log":
                    self.log.insert("end", payload)
                    self.log.see("end")
                elif kind == "prog":
                    d, t, m = payload
                    self.bar["value"] = 100.0 * d / t if t else 0
                    self.status.config(text=f"{d:,}/{t:,}   {m}")
                elif kind == "done":
                    self.running = False
                    self.runbtn.config(state="normal")
                    self.status.config(text="queue finished")
        except queue.Empty:
            pass
        self.root.after(120, self._pump)


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
