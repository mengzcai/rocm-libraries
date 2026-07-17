// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Standalone ds_load_tr4_b64 hardware stress test for gfx1250.
//
// Purpose: isolate whether the intermittent, reboot-clearable FP4 corruption
// seen in NN/TT GEMM (kernels that use the LDS 4-bit transpose-read
// ds_load_tr4_b64) is caused by the tr4 datapath itself, with NO GEMM / WMMA /
// TDM / global-store involved. If this micro test reports FAIL on a machine in
// the bad state while doing nothing but ds_load_tr4_b64, that is direct
// evidence the LDS transpose-read hardware is the faulty unit (not codegen,
// not tdm_load, not global_store).
//
// Method (self-consistency): every wave fills its 64KB LDS block with a fixed
// pattern; each lane reads its slots with ds_load_tr4_b64 and folds them into
// a per-lane XOR signature. The transpose read is a pure function of the fixed
// LDS contents, so recomputing that signature many times MUST reproduce it
// bit-for-bit. Any difference => tr4 returned different data for identical LDS
// contents => flaky transpose-read. This matches the real failure signature
// (intermittent, near-total, RANDOM, reboot-clearable); a deterministic
// transpose miscompute is not what we chase (that would be codegen, ruled out,
// and would not clear on reboot).
//
// !!! IMPORTANT CAVEAT (unverified) !!!
// This test emits ds_load_tr4_b64 via inline asm. During development every run
// HUNG the GPU (processes stuck in D-state at amdgpu_mes_reg_write_reg_wait,
// GPU stuck at 100%), even at 1 block / 64 lanes / a single read. It is NOT yet
// confirmed whether that hang is (a) our inline asm missing a required hardware
// precondition (m0 setup / LDS base in the address register / wave state that
// TensileLite's codegen normally provides), or (b) the GPU already being in the
// bad state. DO NOT trust a "hang" as a tr4 verdict until this is resolved on a
// known-good GPU. See README.md in this directory for the debugging checklist.
//
// Build / run: see README.md and run_tr4_stress.sh in this directory.
//
// Exit 0 = all sweeps self-consistent. Exit 1 = miscompare (prints count +
// first mismatch iter/lane). Exit 2 = kernel launch/exec error.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>

typedef int v2i __attribute__((ext_vector_type(2)));

// Emit ds_load_tr4_b64 directly via inline asm. The clang builtin
// __builtin_amdgcn_ds_read_tr4_b64_v2i32 is gated on gfx950-insts and will not
// compile for gfx1250, but the raw instruction assembles fine for gfx1250
// (TensileLite emits it verbatim). addr is a 32-bit LDS byte offset.
__device__ __forceinline__ v2i tr4_read(uint32_t lds_byte_off)
{
    v2i out;
    asm volatile(
        "ds_load_tr4_b64 %0, %1\n\t"
        "s_wait_dscnt 0\n\t"
        : "=v"(out)
        : "v"(lds_byte_off)
        : "memory");
    return out;
}

// LDS bytes per block. 64KB matches the largest failing A-tile (MT512xDU256).
#define LDS_BYTES (64 * 1024)
#define THREADS 256

__global__ void tr4_stress(uint32_t* mismatch_count,
                           uint32_t* first_bad,
                           int iters)
{
    __shared__ v2i lds[LDS_BYTES / 8];

    const int tid = threadIdx.x;

    // Fill LDS with a deterministic, lane-distinct pattern (nonzero, spread
    // across all 4-bit nibbles so a transpose actually moves bits around).
    uint32_t* lds32 = (uint32_t*)lds;
    for (int i = tid; i < LDS_BYTES / 4; i += THREADS) {
        lds32[i] = 0x13579BDFu ^ (i * 0x9E3779B9u);
    }
    __syncthreads();

    const int nslots = LDS_BYTES / 8;

    // Each lane owns a fixed set of slots spanning the whole 64KB region
    // (lane tid handles slots tid, tid+THREADS, ...), so collectively the wave
    // covers every LDS bank/address each pass, but each lane only does
    // nslots/THREADS (=32) tr4 reads per pass. The per-lane XOR is a
    // deterministic fingerprint of the tr4 datapath.
    v2i sig = {0, 0};
    for (int s = tid; s < nslots; s += THREADS) {
        v2i r = tr4_read((uint32_t)(s * 8));
        sig.x ^= r.x;
        sig.y ^= r.y;
    }

    uint32_t local_bad = 0;
    uint32_t local_first = 0xFFFFFFFFu;

    // Stress passes: recompute the same fingerprint many times; every pass MUST
    // reproduce the reference bit-for-bit.
    for (int it = 0; it < iters; ++it) {
        v2i acc = {0, 0};
        for (int s = tid; s < nslots; s += THREADS) {
            v2i r = tr4_read((uint32_t)(s * 8));
            acc.x ^= r.x;
            acc.y ^= r.y;
        }
        if ((acc.x ^ sig.x) | (acc.y ^ sig.y)) {
            local_bad++;
            if (local_first == 0xFFFFFFFFu) local_first = (uint32_t)it;
        }
    }

    if (local_bad) {
        atomicAdd(mismatch_count, local_bad);
        atomicMin(first_bad, (local_first << 12) | (tid & 0xFFF));
    }
}

int main(int argc, char** argv)
{
    int iters  = (argc > 1) ? atoi(argv[1]) : 200;  // each iter = full 64KB sweep
    int blocks = (argc > 2) ? atoi(argv[2]) : 2048; // saturate the GPU

    uint32_t *d_mismatch, *d_first;
    hipMalloc(&d_mismatch, sizeof(uint32_t));
    hipMalloc(&d_first, sizeof(uint32_t));
    uint32_t zero = 0, initFirst = 0xFFFFFFFFu;
    hipMemcpy(d_mismatch, &zero, sizeof(uint32_t), hipMemcpyHostToDevice);
    hipMemcpy(d_first, &initFirst, sizeof(uint32_t), hipMemcpyHostToDevice);

    hipLaunchKernelGGL(tr4_stress, dim3(blocks), dim3(THREADS), 0, 0,
                       d_mismatch, d_first, iters);
    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        printf("KERNEL LAUNCH/EXEC ERROR: %s\n", hipGetErrorString(err));
        return 2;
    }

    uint32_t mismatch = 0, first = 0;
    hipMemcpy(&mismatch, d_mismatch, sizeof(uint32_t), hipMemcpyDeviceToHost);
    hipMemcpy(&first, d_first, sizeof(uint32_t), hipMemcpyDeviceToHost);

    printf("grid=%d x %d threads, iters=%d per lane, total tr4 reads=%.2e\n",
           blocks, THREADS, iters, (double)blocks * THREADS * iters);
    if (mismatch == 0) {
        printf("PASS: all ds_load_tr4_b64 reads self-consistent (0 miscompares)\n");
        return 0;
    } else {
        printf("FAIL: %u tr4 miscompares detected (first: iter=%u lane=%u)\n",
               mismatch, first >> 12, first & 0xFFF);
        printf("=> ds_load_tr4_b64 returned inconsistent data for identical LDS "
               "contents => LDS transpose-read hardware is flaky.\n");
        return 1;
    }
}
