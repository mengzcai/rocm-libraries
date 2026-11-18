# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""TailFolding: the K tail folded into the TDM unroll loop (gfx1250).

With TailFolding=1 the unroll loop runs ceil(K / DepthU) iterations and no tail
loop is emitted: every TDM address step also shrinks the descriptor's 32-bit K
dim, so TDM zero-fills the last k-tile beyond K. The GL2 prefetch stops one
k-tile earlier so it never reaches past K, and for K < DepthU its K index is
clamped to K - 1.
"""

import re

import pytest
import yaml

from config_harness import emit_kernels_from_config


pytestmark = pytest.mark.unit


def _config(tail_folding, extra=(), mi=(16, 16, 32, 1, 1, 1, 1, 2, 1), wg=(32, 1, 1)):
    # bf16 TN, TDM + SIA4 + PGR2 + PLR1, DepthU 128, cluster 2x2 (no StaggerU code).
    fork_parameters = [
        {"MatrixInstruction": [list(mi)]},
        {"WorkGroup": [list(wg)]},
        {"DepthU": [128]},
        {"LdsBlockSizePerPadA": [256]},
        {"LdsBlockSizePerPadB": [256]},
        {"PrefetchLocalRead": [1]},
        {"PrefetchGlobalRead": [2]},
        {"ScheduleIterAlg": [4]},
        {"TDMInst": [3]},
        {"TransposeLDS": [1]},
        {"LdsPadA": [8]},
        {"LdsPadB": [8]},
        {"GlobalReadVectorWidthA": [1]},
        {"GlobalReadVectorWidthB": [1]},
        {"VectorWidthA": [1]},
        {"VectorWidthB": [1]},
        {"LocalReadVectorWidth": [8]},
        {"StoreVectorWidth": [1]},
        {"SourceSwap": [True]},
        {"MIArchVgpr": [True]},
        {"InitCIterWmma": [1]},
        {"WavefrontSize": [32]},
        {"ExpandPointerSwap": [False]},
        {"ClusterDim": [[2, 2]]},
        {"TailFolding": [tail_folding]},
    ]
    fork_parameters = [p for p in fork_parameters if next(iter(p)) not in dict(extra)] + [{k: [v]} for k, v in extra]
    return {
        "GlobalParameters": {"Architecture": "gfx1250", "PrintLevel": 0},
        "BenchmarkProblems": [[
            {
                "OperationType": "GEMM",
                "DataType": "b",
                "DestDataType": "b",
                "ComputeDataType": "s",
                "HighPrecisionAccumulate": True,
                "TransposeA": True,
                "TransposeB": False,
                "UseBeta": True,
                "Batched": True,
                "SupportUserArgs": True,
            },
            {
                "InitialSolutionParameters": None,
                "BenchmarkCommonParameters": [{"KernelLanguage": ["Assembly"]}],
                "ForkParameters": fork_parameters,
                "BenchmarkFinalParameters": [{"ProblemSizes": [{"Exact": [720, 432, 1, 1248]}]}],
            },
        ]],
    }


def _emit(tmp_path, config, name):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(config))
    return emit_kernels_from_config(path, limit=1, arch="gfx1250")


def _emit_one(tmp_path, config, name):
    results = _emit(tmp_path, config, name)
    assert len(results) == 1
    _, assembly, error = results[0]
    assert error == 0
    return assembly


def _kernel_name(assembly):
    return re.search(r"^\.amdhsa_kernel (\S+)", assembly, re.M).group(1)


def _assert_shrink_pairs(assembly):
    """Every TDM address step (its carry half; the InitCIterWmma clone duplicates only the low
    add) also shrinks the 32-bit K dim, and nothing between the two halves clobbers SCC."""
    lines = assembly.split("\n")
    steps = assembly.count("// TDM increment hi (carry)")
    lo = [i for i, l in enumerate(lines)
          if re.match(r"s_sub_u32 s\[sgprtdmAGroup1\+1\], s\[sgprtdmAGroup1\+1\], 0x800000 .*TailFolding: K dim -= 128", l)]
    hi = [i for i, l in enumerate(lines)
          if re.match(r"s_subb_u32 s\[sgprtdmAGroup1\+2\], s\[sgprtdmAGroup1\+2\], 0 .*TailFolding: K dim \(high half\)", l)]
    assert steps and len(lo) == steps and len(hi) == steps, (steps, len(lo), len(hi))
    keep_scc = re.compile(r"^(s_mov|s_cselect|s_cmov|s_wait|s_set_vgpr_msb|s_nop|s_barrier|s_delay|s_setprio|v_|ds_|"
                          r"tensor_|global_|buffer_|s_prefetch|//|/\*|\.|label_|$)")
    for i in lo:
        j = i + 1
        while not lines[j].startswith("s_subb_u32 s[sgprtdmAGroup1+2]"):
            assert keep_scc.match(lines[j].strip()), "SCC clobbered between the K-dim halves: " + lines[j]
            j += 1


@pytest.mark.parametrize("hplr", [0, 2], ids=["noloadloop", "halfplr"])
def test_tail_folding_codegen(tmp_path, hplr):
    extra = [("HalfPLR", hplr)] if hplr else []
    kw = dict(mi=(16, 16, 32, 1, 1, 2, 2, 2, 2), wg=(32, 4, 1)) if hplr else {}
    assembly = _emit_one(tmp_path, _config(1, extra, **kw), "tf1_%u.yaml" % hplr)
    assert "_TF1_" in _kernel_name(assembly)
    assert "label_TailLoopBeginL" not in assembly
    assert re.search(r"s_and_b32 s\d+, s\[sgprSizesSum\+0\], 127 .*\n"
                     r"s_cselect_b32 s\d+, 1, 0 .*partial last k-tile\?\n"
                     r"s_add_u32 s\[sgprLoopCounterL\], s\[sgprLoopCounterL\], s\d+ .*ceil\(K / DepthU\)", assembly), \
        "the loop counter does not count the partial k-tile"
    _assert_shrink_pairs(assembly)


def test_tail_folding_zero_is_default(tmp_path):
    assembly = _emit_one(tmp_path, _config(0), "tf0.yaml")
    assert "_TF" not in _kernel_name(assembly)
    assert "TailFolding" not in assembly and "label_TailLoopBeginL" in assembly


def test_tail_folding_normalized_without_tail(tmp_path):
    assembly = _emit_one(tmp_path, _config(1, [("AssertSummationElementMultiple", 128)]), "tf1_asem.yaml")
    assert "_TF1_" not in _kernel_name(assembly) and "TailFolding" not in assembly


@pytest.mark.parametrize("pgl", [1, 2])
def test_tail_folding_gl2_prefetch_stops_before_k(tmp_path, pgl):
    # With a K tail the last k-tile is partial and the GL2 prefetch stops one k-tile earlier: the
    # thresholds are selected once before the loop from K % DepthU; the loop compares against an sgpr.
    base = _emit_one(tmp_path, _config(0, [("PrefetchGL2", pgl)]), "tf0_pgl%u.yaml" % pgl)
    fold = _emit_one(tmp_path, _config(1, [("PrefetchGL2", pgl)]), "tf1_pgl%u.yaml" % pgl)
    lits = sorted({int(v, 0) for v in re.findall(r"^s_cmp_le_u32 s\[sgprLoopCounterL\], (0x[0-9a-f]+|\d+) +// counterL<=PGR", base, re.M)})
    assert lits == list(range(2, 2 + pgl)) + [2 + pgl], lits
    assert "GL2FoldThreshold" not in base
    assert not re.search(r"^s_cmp_le_u32 s\[sgprLoopCounterL\], (0x[0-9a-f]+|\d+) +// counterL<=PGR", fold, re.M)
    assert re.search(r"^s_and_b32 s\[sgprGL2FoldThreshold\], s\[sgprSizesSum\+0\], 127 ", fold, re.M)
    assert re.search(r"^s_cselect_b32 s\[sgprGL2FoldThreshold\], %u, %u " % (3 + pgl, 2 + pgl), fold, re.M)
    pre = re.findall(r"^s_cmp_le_u32 s\[sgprLoopCounterL\], s\d+ +// counterL<=PGR(\+1)?\(\+1 if K tail\)", fold, re.M)
    assert len(pre) == pgl, pre
    inloop = lambda asm, pat: len(re.findall(pat, asm, re.M))
    assert inloop(fold, r"^s_cmp_le_u32 s\[sgprLoopCounterL\], s\[sgprGL2FoldThreshold\]") == \
        inloop(base, r"^s_cmp_le_u32 s\[sgprLoopCounterL\], (0x[0-9a-f]+|\d+) +// counterL<=PGR\+GL2$") > 0


@pytest.mark.parametrize("extra, why", [
    ([("ClusterDim", [1, 1])], "StaggerU"),
    ([("GlobalSplitU", 2)], "GlobalSplitU"),
    ([("PrefetchGlobalRead", 1), ("1LDSBuffer", 1)], "PrefetchGlobalRead"),
], ids=["stagger", "gsu2", "pgr1"])
def test_tail_folding_rejects(tmp_path, extra, why):
    assert _emit(tmp_path, _config(1, extra), "tf1_reject.yaml") == []


def test_tail_folding_gl2_prefetch_k_clamp(tmp_path):
    # K < DepthU still runs one folded iteration that prefetches k-tile 0: its K index is clamped to K - 1.
    base = _emit_one(tmp_path, _config(0, [("PrefetchGL2", 1)]), "tf0k.yaml")
    fold = _emit_one(tmp_path, _config(1, [("PrefetchGL2", 1)]), "tf1k.yaml")
    assert "TailFolding: past K?" not in base
    assert re.search(r"s_sub_u32 (s\d+), s\[sgprSizesSum\+0\], 1 .*TailFolding: last K index", fold)
    assert fold.count("TailFolding: past K?") == 2  # one prefetch address per tensor here
