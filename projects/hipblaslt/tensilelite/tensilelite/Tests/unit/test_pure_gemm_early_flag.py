# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# Codegen test: the pure-gemm flag is computed in the prologue.
#
# A CompactLoopStore kernel with bias + scaleAlphaVec + runtime activation loads
# the flag's kernargs (AddressBias, AddressScaleAlphaVec, ActivationType; Beta is
# preloaded) in the prologue and folds them into one whole-kernel sgpr,
# PureGemmFlag. A non-persistent kernel issues the loads at the start of the
# prologue, ahead of the first tensor load, or, when ClusterAIDRemap waits for
# MSG_RTN_GET_SE_AID_ID, right after that wait (kmcnt only waits for 0, so the
# wait would drain them and delay the first TDM). A persistent kernel issues them
# per tile inside the persistent loop. The epilogue only branches on the sgpr, so
# the pure path loads no store kernargs and has no s_wait_kmcnt.
################################################################################

import re

import pytest
import yaml

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_CONFIG = """
GlobalParameters: {MinimumRequiredVersion: 5.0.0, NumElementsToValidate: 0, KernelTime: false}
BenchmarkProblems:
- - {OperationType: GEMM, DataType: B, DestDataType: B, ComputeDataType: S, HighPrecisionAccumulate: true,
     TransposeA: true, TransposeB: false, UseBeta: true, Batched: true, UseBias: 1, BiasDataTypeList: [s, B],
     Activation: true, ActivationType: hipblaslt_all, UseScaleAlphaVec: 1}
  - InitialSolutionParameters: null
    BenchmarkCommonParameters:
    - KernelLanguage: [Assembly]
    ForkParameters:
    - {ExpandPointerSwap: [false]}
    - {GlobalReadVectorWidthA: [1]}
    - {GlobalReadVectorWidthB: [1]}
    - {InitCIterWmma: [1]}
    - {LocalReadVectorWidthA: [8]}
    - {LocalReadVectorWidthB: [8]}
    - {MXLoadInst: [TDM]}
    - {MaxLDS: [327680]}
    - {PrefetchGlobalRead: [2]}
    - {PrefetchLocalRead: [0]}
    - {ScheduleIterAlg: [4]}
    - {TDMInst: [3]}
    - {ThreadTile: [[1, 1]]}
    - {TransposeLDS: [1]}
    - {WavefrontSize: [32]}
    - {TileProcessingStrategy: [None]}
    - {GlobalSplitU: [1]}
    - {GlobalSplitUAlgorithm: [MultipleBuffer]}
    - {VectorWidthA: [1]}
    - {VectorWidthB: [2]}
    - {LdsBlockSizePerPadA: [256]}
    - {LdsBlockSizePerPadB: [256]}
    - {LdsPadA: [8]}
    - {LdsPadB: [8]}
    - {MIArchVgpr: [true]}
    - {ActivationFuncCall: [true]}
    - {StoreVectorWidth: [1]}
    - {AdaptiveGemm: [1]}
    - {CompactLoopStore: [true]}
    - {SourceSwap: [true]}
    - {WorkGroupMapping: [8]}
    - Groups:
      - - {DepthU: 32, MatrixInstruction: [16, 16, 32, 1, 1, 1, 2, 1, 4], WorkGroup: [32, 1, 1]}
    BenchmarkJoinParameters: null
    BenchmarkFinalParameters:
    - ProblemSizes:
      - Exact: [608, 432, 1, 32]
    - BiasTypeArgs: [s]
    - ActivationArgs:
      - - Enum: none
"""

_SCALAR_WRITE = re.compile(r"^(s_[a-z0-9_]+|v_readfirstlane_b32|v_readlane_b32)\s+s\[?(\d+)(?::(\d+))?\]?[, ]")
_NOT_A_WRITE = ("s_cmp", "s_bitcmp", "s_cbranch", "s_wait", "s_nop", "s_barrier", "s_branch", "s_setpc",
                "s_prefetch", "s_set_vgpr", "s_store", "s_atomic", "s_delay", "s_load", "s_buffer_load")


def _emit(tmp_path, persistent, cluster=False):
    config = yaml.safe_load(_CONFIG)
    params = config["BenchmarkProblems"][0][1]["ForkParameters"]
    if persistent:
        for parameter in params:
            if "TileProcessingStrategy" in parameter:
                parameter["TileProcessingStrategy"] = ["StreamK"]
        params.append({"WorkAssignment": ["DynamicWorkQueue"]})
    if cluster:
        params.append({"ClusterDim": [[2, 2]]})
        params.append({"ClusterAIDMapping": [-2]})
    path = tmp_path / ("early_flag_%s%s.yaml" % ("sk" if persistent else "dp", "_cluster" if cluster else ""))
    path.write_text(yaml.safe_dump(config))
    results = emit_kernels_from_config(path, limit=1, arch="gfx1250")
    assert len(results) == 1
    _, assembly, error = results[0]
    assert error == 0
    return assembly.split("\n")


def _flag_load_sites(lines):
    """[(first, last) line index] of each block of flag kernarg loads (ArgType check .. ArgsEnd label)."""
    sites, i = [], 0
    while True:
        try:
            start = next(j for j in range(i, len(lines)) if "label_PureGemmFlagExternalArgs" in lines[j] and "s_cbranch" in lines[j])
        except StopIteration:
            return sites
        end = next(j for j in range(start, len(lines)) if re.match(r"label_PureGemmFlagArgsEnd(_\d+)?:", lines[j]))
        loads = [j for j in range(start, end) if lines[j].startswith("s_load")]
        # AddressBias, AddressScaleAlphaVec, ActivationType for ArgType != 2 and == 2.
        assert len(loads) == 6, [lines[j] for j in loads]
        sites.append((start, end))
        i = end


def _flag_reg(lines):
    return int(next(re.match(r"\.set sgprPureGemmFlag, (\d+)", l).group(1) for l in lines
                    if l.startswith(".set sgprPureGemmFlag,")))


def _fold_end(lines, reg):
    fold = [i for i, l in enumerate(lines) if l.startswith("s_cselect_b32 s%d, s%d, 0" % (reg, reg))]
    assert fold, "the pure-gemm flag is never folded"
    wait = max(i for i in range(fold[0]) if "pure-gemm flag kernargs" in lines[i])
    assert lines[wait].startswith("s_wait_kmcnt 0"), "the fold does not wait for its kernargs"
    return fold[-1]


def _assert_flag_kept(lines, reg, start):
    """No instruction between the fold and the first pure branch rewrites the flag sgpr."""
    pure = next(i for i, l in enumerate(lines) if "branch to label_GW_Pure" in l)
    rewrites = []
    for i in range(start + 1, pure):
        m = _SCALAR_WRITE.match(lines[i])
        if not m or m.group(1).startswith(_NOT_A_WRITE):
            continue
        lo = int(m.group(2))
        hi = int(m.group(3)) if m.group(3) else lo
        if lo <= reg <= hi:
            rewrites.append((i + 1, lines[i]))
    assert not rewrites, "PureGemmFlag s%d is rewritten before the pure-gemm branch: %s" % (reg, rewrites[:3])


@pytest.mark.parametrize("cluster", [False, True], ids=["no_remap", "cluster_aid_remap"])
def test_non_persistent_flag_loads_never_delay_the_first_tensor_load(tmp_path, cluster):
    lines = _emit(tmp_path, persistent=False, cluster=cluster)
    sites = _flag_load_sites(lines)
    assert len(sites) == 1, sites
    first_tdm = next(i for i, l in enumerate(lines) if l.startswith("tensor_load_to_lds"))
    assert sites[0][1] < first_tdm, "the flag kernargs are not issued ahead of the first tensor load"
    se_aid = [i for i, l in enumerate(lines[:first_tdm]) if l.startswith("s_wait_kmcnt") and "SE_AID_ID" in l]
    if cluster:
        # The remap's kmcnt wait would drain loads issued before it and delay the first TDM.
        assert se_aid, "the kernel has no ClusterAIDRemap"
        assert sites[0][0] > se_aid[-1], "the flag kernargs are issued before the SE_AID_ID wait"
        end = next(i for i, l in enumerate(lines) if l.startswith("label_ClusterAIDRemapEnd"))
        assert sites[0][0] > end
    else:
        assert not se_aid
        sgprs_done = next(i for i, l in enumerate(lines) if l.startswith(".set sgprPureGemmFlag,"))
        assert sites[0][0] - sgprs_done < 8, "the flag kernargs are not issued at the start of the prologue"

    reg = _flag_reg(lines)
    fold = _fold_end(lines, reg)
    open_loop = next(i for i, l in enumerate(lines) if l.startswith("label_openLoopL:"))
    assert fold < open_loop, "the pure-gemm flag is not folded before the main loop"
    _assert_flag_kept(lines, reg, fold)

    # Pure path of the optimized NLL: the store-kernarg load is skipped and nothing waits for kernargs
    # before the branch to the pure pass.
    start = next(i for i, l in enumerate(lines) if l.startswith("label_Summation_End_OptNLL:"))
    pure = next(i for i in range(start, len(lines)) if "branch to label_GW_Pure" in lines[i])
    skip = False
    for i in range(start, pure):
        l = lines[i]
        if l.startswith("s_cbranch_scc1 label_PureGemmSkipStoreSgprLoad"):
            skip = True
        elif skip and re.match(r"label_PureGemmSkipStoreSgprLoad(_\d+)?:", l):
            skip = False
        elif not skip:
            assert not l.startswith(("s_wait_kmcnt", "s_load")), \
                "the pure path still loads or waits for store kernargs at line %d: %s" % (i + 1, l)
    assert any(l.startswith("s_cbranch_scc1 label_PureGemmSkipStoreSgprLoad") for l in lines[start:pure])


def test_persistent_flag_is_loaded_per_tile(tmp_path):
    lines = _emit(tmp_path, persistent=True)
    loop = next(i for i, l in enumerate(lines) if l.startswith("label_PersistentLoopStart:"))
    sites = _flag_load_sites(lines)
    assert len(sites) == 1, sites
    assert sites[0][0] > loop, "a persistent kernel must reload the flag kernargs per tile"
    reg = _flag_reg(lines)
    _assert_flag_kept(lines, reg, _fold_end(lines, reg))
