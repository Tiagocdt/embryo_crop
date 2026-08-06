#!/usr/bin/env python3
"""Keep a run from eating the whole laptop.

The job is I/O-bound, not CPU-bound: it reads a frame, crops it, writes it.
So the knob that matters is HOW MANY FRAMES ARE IN FLIGHT AT ONCE. That one
number sets both the CPU used and the peak memory, and it is the only thing
worth limiting.

    peak memory  ~=  workers * bytes_per_frame * COPIES_IN_FLIGHT
    cpu          ~=  workers, capped by how fast the disk can feed them

Everything here is advisory and measured, so it degrades gracefully on a
machine it has never seen.
"""
from __future__ import annotations

import os
import resource
import subprocess

# A worker holds roughly: the raw frame, the crop, and one float32 working
# copy of the crop. Measured at ~3x the raw frame for 2048^2 uint16.
COPIES_IN_FLIGHT = 3


def machine():
    """(logical_cores, total_ram_bytes) without any third-party module."""
    cores = os.cpu_count() or 4
    ram = None
    try:
        ram = int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                                 capture_output=True, text=True,
                                 timeout=5).stdout.strip())
    except Exception:
        try:
            ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except Exception:
            ram = 8 * 1024 ** 3
    return cores, ram


def bytes_per_frame(shape, dtype="uint16"):
    n = shape[0] * shape[1] if shape else 2048 * 2048
    return n * (2 if "16" in str(dtype) else 1)


def recommend(frame_shape=None, dtype="uint16", cpu_fraction=0.6,
              memory_fraction=0.25):
    """Suggest a worker count that uses at most `cpu_fraction` of the cores
    and `memory_fraction` of RAM. Defaults leave the laptop usable."""
    cores, ram = machine()
    bpf = bytes_per_frame(frame_shape, dtype)
    by_cpu = max(1, int(cores * cpu_fraction))
    by_mem = max(1, int((ram * memory_fraction) / (bpf * COPIES_IN_FLIGHT)))
    workers = max(1, min(by_cpu, by_mem, 16))
    return {
        "workers": workers,
        "cores": cores,
        "ram_gb": round(ram / 1024 ** 3, 1),
        "bytes_per_frame": bpf,
        "limited_by": "cpu" if by_cpu <= by_mem else "memory",
        "est_peak_mb": round(workers * bpf * COPIES_IN_FLIGHT / 1024 ** 2, 0),
    }


def apply_limits(nice_level=5, memory_cap_mb=None):
    """Lower priority so the UI stays responsive, and optionally hard-cap
    address space. Returns a list of what actually took effect."""
    done = []
    try:
        os.nice(nice_level)
        done.append(f"nice +{nice_level}")
    except (OSError, PermissionError):
        done.append("nice: not permitted")
    if memory_cap_mb:
        cap = int(memory_cap_mb) * 1024 ** 2
        try:
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            new_hard = hard if hard != resource.RLIM_INFINITY else cap
            resource.setrlimit(resource.RLIMIT_AS, (cap, new_hard))
            done.append(f"memory cap {memory_cap_mb} MB")
        except (ValueError, OSError):
            # macOS often refuses RLIMIT_AS. The worker cap is the real limit.
            done.append("memory cap: not supported here (worker cap applies)")
    return done


def describe(frame_shape=None, dtype="uint16"):
    r = recommend(frame_shape, dtype)
    return (f"{r['cores']} cores · {r['ram_gb']} GB RAM · "
            f"{r['bytes_per_frame']/1024**2:.1f} MB per frame\n"
            f"recommended {r['workers']} workers "
            f"(limited by {r['limited_by']}), peak ~{r['est_peak_mb']:.0f} MB")


if __name__ == "__main__":
    import sys
    shape = None
    if len(sys.argv) > 1:
        try:
            from acquifer import index_folder
            shape = index_folder(sys.argv[1]).frame_shape
        except Exception as e:
            print(f"(could not index {sys.argv[1]}: {e})")
    print(describe(shape))
    for level in (("gentle", 0.35, 0.15), ("balanced", 0.6, 0.25),
                  ("all-out", 0.95, 0.5)):
        name, cf, mf = level
        r = recommend(shape, cpu_fraction=cf, memory_fraction=mf)
        print(f"  {name:<9} {r['workers']:>2} workers   "
              f"peak ~{r['est_peak_mb']:.0f} MB   (limited by {r['limited_by']})")
