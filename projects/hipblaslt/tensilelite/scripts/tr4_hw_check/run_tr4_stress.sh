#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Build + run the standalone ds_load_tr4_b64 hardware stress test.
# See README.md for the full rationale and the (unverified) GPU-hang caveat.
#
# Usage:
#   ./run_tr4_stress.sh                 # use the DEFAULT_LOOPS / DEFAULT_ITERS below
#   ./run_tr4_stress.sh <loops> <iters> # override on the command line, e.g. 500 2000
#
# Env (auto-detected, override if needed):
#   ROCM   path to the ROCm SDK (dir containing bin/amdclang++ and lib/)
#   ARCH   GPU arch (default gfx1250)

set -u

# ---- tunables (edit these) ----------------------------------------------
DEFAULT_LOOPS=100   # number of kernel launches (stops early on first failure)
DEFAULT_ITERS=200   # tr4 full-64KB sweeps per launch (higher = longer soak)
# -------------------------------------------------------------------------

LOOPS="${1:-$DEFAULT_LOOPS}"
ITERS="${2:-$DEFAULT_ITERS}"
ARCH="${ARCH:-gfx1250}"

# --- locate ROCm ---------------------------------------------------------
if [ -z "${ROCM:-}" ]; then
  for cand in \
    /opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel \
    /opt/rocm \
    "$(command -v amdclang++ >/dev/null 2>&1 && dirname "$(dirname "$(command -v amdclang++)")")"; do
    if [ -n "$cand" ] && [ -x "$cand/bin/amdclang++" ]; then ROCM="$cand"; break; fi
  done
fi
if [ -z "${ROCM:-}" ] || [ ! -x "$ROCM/bin/amdclang++" ]; then
  echo "ERROR: could not find ROCm SDK. Set ROCM=/path/to/rocm (needs bin/amdclang++)." >&2
  exit 3
fi

CLANG="$ROCM/bin/amdclang++"
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/tr4_stress.hip"
BIN="$HERE/tr4_stress"

export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"
export HSA_ENABLE_SDMA=0   # required on gfx1250 (see project notes)

# --- build ---------------------------------------------------------------
echo "[build] $CLANG --offload-arch=$ARCH"
"$CLANG" -x hip --offload-arch="$ARCH" -O3 "$SRC" -o "$BIN" || { echo "BUILD FAILED"; exit 3; }

# sanity: confirm the instruction is actually emitted
n=$("$CLANG" -x hip --offload-arch="$ARCH" -O3 -S "$SRC" -o - 2>/dev/null | grep -c ds_load_tr4_b64)
echo "[build] ds_load_tr4_b64 in asm: $n (expect > 0)"

# --- run loop ------------------------------------------------------------
echo "[run] $LOOPS launches x iters=$ITERS ; logging rocm-smi around each"
for i in $(seq 1 "$LOOPS"); do
  smi=$("$ROCM/bin/rocm-smi" --showtemp --showclocks 2>/dev/null | grep -iE "junction|sclk" | tr '\n' ' ')
  out=$("$BIN" "$ITERS")
  rc=$?
  echo "launch $i: rc=$rc | $out | $smi"
  if [ "$rc" -ne 0 ]; then
    echo "=== STOP: launch $i returned $rc ==="
    [ "$rc" -eq 1 ] && echo "tr4 MISCOMPARE -> LDS transpose-read hardware confirmed flaky."
    [ "$rc" -eq 2 ] && echo "kernel exec error (could be a GPU hang; see README caveat)."
    exit "$rc"
  fi
done
echo "=== all $LOOPS launches PASSED (tr4 self-consistent) ==="
