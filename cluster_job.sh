#!/bin/bash
# =============================================================================
# Run embryo_crop on an HPC cluster, as ONE job.
#
#   sbatch cluster_job.sh <raw-dir> <out-dir> <plate-name> [extra process.py args]
#
# Why one job and not a job array
# -------------------------------
# The work is I/O-bound: read a frame, crop it, write it. Detection runs a
# handful of times per WELL, not per frame, so nothing here needs many machines.
# What it needs is many reads IN FLIGHT AT ONCE, and a thread pool inside a
# single job provides that just as well as N separate jobs -- with none of the
# array's costs (N scheduler queue waits, N Python interpreters, N re-reads of
# the file index, and a pile of near-empty tasks whenever the well count does
# not divide evenly).
#
# A job array is the right tool when tasks are CPU-bound and independent, or
# when one task cannot finish inside the wall-clock limit. Neither is true here.
# Scale --cpus-per-task and --workers together until the filesystem saturates;
# past that point, adding either stops helping.
#
# EDIT THESE for your site: the partition name and the module line below are
# EMBL-specific. `module load` must provide a Python with numpy >= 2.
# =============================================================================
#SBATCH --job-name=embryo_crop
#SBATCH --output=logs/embryo_crop_%j.out
#SBATCH --error=logs/embryo_crop_%j.out
#SBATCH --partition=htc-el8
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=08:00:00

set -euo pipefail

RAW="${1:?usage: sbatch cluster_job.sh <raw-dir> <out-dir> <plate> [args...]}"
OUT="${2:?usage: sbatch cluster_job.sh <raw-dir> <out-dir> <plate> [args...]}"
PLATE="${3:?usage: sbatch cluster_job.sh <raw-dir> <out-dir> <plate> [args...]}"
shift 3

# SLURM COPIES this script to a node-local spool dir before running it, so
# ${BASH_SOURCE[0]} points at the copy (e.g. /var/spool/.../slurm_script) and
# NOT at the checkout. Deriving the working directory from it lands you
# somewhere unwritable -- the symptom is a baffling
#   mkdir: cannot create directory 'logs': Permission denied
# even though the real directory is writable from the same node.
# $SLURM_SUBMIT_DIR is where sbatch was invoked; fall back to BASH_SOURCE only
# when running this script directly, outside SLURM.
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$SCRIPT_DIR"
mkdir -p logs

# --- site-specific: a Python with numpy >= 2 -------------------------------
# An EasyBuild module exports PYTHONPATH, which a venv does NOT isolate. If the
# module's numpy is older than the venv's, it SHADOWS it and tifffile dies with
#   TypeError: 'copy' is an invalid keyword argument for this function
# So either load a module that already has numpy >= 2 (as here), or unset
# PYTHONPATH after loading.
if command -v module >/dev/null 2>&1 || [ -f /etc/profile.d/lmod.sh ]; then
    source /etc/profile.d/lmod.sh 2>/dev/null || true
    module purge          >/dev/null 2>&1 || true
    module load SciPy-bundle/2026.05-gfbf-2026.1 >/dev/null 2>&1 || true
fi

PY="$SCRIPT_DIR/.venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3)"

echo "=============================================================="
echo "host        $(hostname)"
echo "job         ${SLURM_JOB_ID:-<no slurm>}   cpus=${SLURM_CPUS_PER_TASK:-?}"
echo "raw         $RAW"
echo "out         $OUT"
echo "plate       $PLATE"
echo "python      $PY"
"$PY" -c "import numpy,tifffile;print('numpy',numpy.__version__,'| tifffile',tifffile.__version__)"
echo "=============================================================="

# -u so progress appears in the log while the job runs instead of at the end.
exec "$PY" -u process.py "$RAW" "$OUT" \
    --plate "$PLATE" \
    --workers "${SLURM_CPUS_PER_TASK:-8}" \
    "$@"
