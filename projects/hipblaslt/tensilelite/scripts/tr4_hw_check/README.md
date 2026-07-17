<!--
Copyright Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->
# tr4 hardware check (`ds_load_tr4_b64`)

Standalone micro-test to decide whether an intermittent, **reboot-clearable**
FP4 GEMM corruption on gfx1250 is caused by the LDS 4-bit transpose-read
instruction `ds_load_tr4_b64` itself — with **no** GEMM / WMMA / TDM /
global-store in the picture.

## Why this exists

On a machine in the "bad" state, FP4 `NN`/`TT` GEMM kernels intermittently
produced near-total, random output corruption that **disappeared after a reboot**
and could not be reproduced across machine-state changes (the exact same kernel
binary + problem size passed on one run and failed on another). Analysis of
which kernels are fragile pointed at one shared datapath:

| layout | A read        | B read        | fragile? |
|--------|---------------|---------------|----------|
| NN     | `ds_load_tr4` | `ds_load_b128`| **yes**  |
| TT     | `ds_load_b128`| `ds_load_tr4` | **yes**  |
| TN     | `ds_load_b128`| `ds_load_b128`| no       |
| NT     | `ds_load_tr4` | `ds_load_tr4` | (uses tr4)|

Only kernels that emit `ds_load_tr4_b64` are fragile; pure-`b128` (`TN`) kernels
never fail. Corruption is pre-store and random (so not `global_store`), and both
the BufferLoad and TDM load paths fail (so not `tdm_load`). That isolates the
suspect to the **LDS transpose-read datapath**.

A "same kernel, swap only the instruction" A/B test is **not possible for FP4**:
the 4-bit lane transpose has no VALU equivalent (`v_swap` works on 32-bit regs
and cannot reorder 8 packed 4-bit values across lanes), which is exactly why the
generator hard-rejects `FP4 requires LDSTrInst == True`. So this micro-test
exercises `ds_load_tr4_b64` in isolation instead.

## What it does

Fills a 64 KB `__shared__` block with a fixed pattern, then each lane reads its
slots with `ds_load_tr4_b64` and folds the results into a per-lane XOR
signature. The transpose read is a pure function of the fixed LDS contents, so
recomputing that signature many times must reproduce it bit-for-bit. Any
difference ⇒ tr4 returned different data for identical LDS contents ⇒ flaky
transpose-read hardware.

## Run

```bash
cd scripts/tr4_hw_check
./run_tr4_stress.sh            # build + 100 launches, stop on first failure
./run_tr4_stress.sh 500 2000   # 500 launches, iters=2000 each (longer soak)
```

Interpretation:
- **all launches PASS** → tr4 datapath is self-consistent on this machine right
  now (run it while the machine is in the *bad* state to be meaningful).
- **rc=1 (MISCOMPARE)** → `ds_load_tr4_b64` returned inconsistent data for
  identical LDS contents ⇒ **LDS transpose-read hardware confirmed flaky**.
- **rc=2 (exec error) / GPU hang** → see caveat below; do **not** treat as a tr4
  verdict yet.

Pair it with `rocm-smi` temp/clock logging (the script already prints a compact
line per launch) to correlate any failure with thermal/clock state.

## ⚠️ Unverified caveat — GPU hang during development

During development, every run of this test **hung the GPU** (host processes
stuck in `D` state at `amdgpu_mes_reg_write_reg_wait`, GPU pinned at 100%,
`kill -9` ineffective), even at 1 block / 64 lanes / a single `ds_load_tr4_b64`.

It is **not yet confirmed** whether the hang is:
1. our inline-asm missing a required hardware precondition that TensileLite's
   codegen normally sets up (e.g. `m0` configuration, the LDS base folded into
   the address VGPR rather than a bare byte offset, or a specific wave/exec
   state before the transpose read), **or**
2. the GPU already being in the bad/hung state when the test was run.

Because of (2), the development runs are inconclusive. Before trusting this test:

1. Run it on a **known-good** gfx1250 (freshly rebooted, idle). If it still
   hangs there, the problem is (1) — our asm — and the addressing needs fixing
   (compare the emitted `ds_load_tr4_b64 vDst, vAddr offset:N` against a real
   TensileLite kernel: TensileLite uses a precomputed `LocalReadAddr` VGPR that
   holds the actual LDS address plus an immediate `offset:`, not a bare byte
   offset in a plain VGPR).
2. If it runs cleanly on a good GPU and only hangs/miscompares on the bad one,
   that is the signal we want.

### Fixing the addressing (if the hang is our asm)

The most likely fix is to make the address operand a proper LDS address VGPR.
Options to try, in order:
- initialize a VGPR to the byte offset and ensure `m0` is set (some `ds_*`
  forms are `m0`-relative), or
- use the `offset:` immediate form: `ds_load_tr4_b64 %0, %1 offset:N` with `%1`
  holding a base address, mirroring TensileLite's emission, or
- generate a tiny kernel through TensileLite/rocisa and copy its exact
  `ds_load_tr4_b64` operand form.

## GEMM-level repro (safe; does not hang the GPU)

`fp4_tr4_layouts.yaml` runs FP4 `MT256x256x256 DU256` in `NN`/`TN`/`TT` through
normal (fully valid) TensileLite codegen, so it will **not** hang the GPU. On a
machine in the bad state, `NN` and `TT` (which use `ds_load_tr4`) intermittently
FAIL validation while `TN` (pure `b128`) always passes — the same signal, at the
GEMM level, without the inline-asm hang risk. Prefer this for day-to-day
checking; use the micro-test only once its addressing/hang caveat is resolved.

```bash
export HSA_ENABLE_SDMA=0
for i in $(seq 20); do \
  Tensile fp4_tr4_layouts.yaml /tmp/tr4_out || echo "run $i had failures"; \
done
```

## Files

- `fp4_tr4_layouts.yaml` — GEMM-level NN/TN/TT repro (safe, no GPU hang).
- `tr4_stress.hip` — the micro-test (inline-asm `ds_load_tr4_b64`; see caveat).
- `run_tr4_stress.sh` — build + looped run with per-launch `rocm-smi` logging.
- `README.md` — this file.
