# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""FastPrologueDivide is a gfx1250-only option.

On gfx1250 (ISA 12.5.0) the prologue divisions use FastPrologueDivide=1 by
default: only the f32 reciprocal estimate runs on the VALU, the Newton step,
quotient and both corrections run on the SALU. Only the non-default
FastPrologueDivide=0 is named (FPD0). Every other ISA keeps the previous
divide and the kernel name it had before the option existed, whatever value
the option has.

The SALU helpers are emitted for gfx1250 and run through a small multi-lane
emulator of just the opcodes they use; quotient and remainder are compared
with Python integer division. ``v_rcp_iflag_f32`` is modeled with its 1 ulp
bound by trying both float32 neighbours of the exact reciprocal of the
converted divisor.
"""

import random
import shutil
import struct
from copy import deepcopy
from fractions import Fraction

import pytest

from rocisa import rocIsa
from rocisa.container import ContinuousRegister

from tensilelite.Common.GlobalParameters import defaultSolution
from tensilelite.Components.WorkGroupMappingAlgos import fastPrologueDivide, saluUInt32DivTail, \
    saluUInt32Reciprocals
from tensilelite.SolutionStructs.Naming import getKernelNameMin

pytestmark = pytest.mark.unit

GFX1250 = (12, 5, 0)
OTHER_ISAS = [(9, 4, 2), (9, 5, 0)]
U32 = 0xFFFFFFFF
LANES = 4


def _kernel(isa, fpd):
    """Build the smallest kernel dict that exercises Naming._getName."""
    kernel = deepcopy(defaultSolution)
    kernel["ProblemType"] = {
        "OperationIdentifier": "Cijk_Ailk_Bljk",
        "DataType": 0,
        "DestDataType": 0,
        "ComputeDataType": 0,
        "GroupedGemm": False,
        "UseBeta": False,
        "UseBias": 0,
    }
    kernel["MacroTile0"] = 64
    kernel["MacroTile1"] = 32
    kernel["DepthU"] = 256
    kernel["MatrixInstM"] = 16
    kernel["MatrixInstN"] = 16
    kernel["MatrixInstB"] = 1
    kernel["MatrixInstruction"] = [16, 16, 1, 1]
    kernel["MIWaveTile"] = [2, 2]
    kernel["WorkGroup"] = [32, 4, 2]
    kernel["ISA"] = isa
    kernel["WavefrontSize"] = 32 if isa == GFX1250 else 64
    kernel["FastPrologueDivide"] = fpd
    return kernel


@pytest.mark.parametrize("fpd", [0, 1])
@pytest.mark.parametrize("isa", OTHER_ISAS)
def test_other_isa_names_never_carry_fpd(isa, fpd):
    name = getKernelNameMin(_kernel(isa, fpd), False)

    assert "_FPD" not in name
    assert name == getKernelNameMin(_kernel(isa, 1), False)


def test_gfx1250_names_only_the_non_default():
    assert "_FPD" not in getKernelNameMin(_kernel(GFX1250, 1), False)
    assert "_FPD0_" in getKernelNameMin(_kernel(GFX1250, 0), False)


@pytest.mark.parametrize("isa,fpd,fast", [(GFX1250, 1, True), (GFX1250, 0, False)]
                         + [(isa, fpd, False) for isa in OTHER_ISAS for fpd in (0, 1)])
def test_fast_divide_only_on_gfx1250(isa, fpd, fast):
    assert fastPrologueDivide(_kernel(isa, fpd)) == fast


# ---------------------------------------------------------------------------
# Emulation of the emitted SALU helpers
# ---------------------------------------------------------------------------

def _f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _cvt_u32(bits):
    value = _f32(bits)
    if value != value or value <= 0:
        return 0
    return min(int(value), U32)


def _rcp(bits, roundUp):
    """Float32 neighbour of the exact 1/x below (roundUp=False) or above (roundUp=True)."""
    x = _f32(bits)
    r = _bits(1.0 / x)  # correctly rounded (struct packs with round-to-nearest-even)
    exact = Fraction(1) / Fraction(x)
    value = Fraction(_f32(r))
    if value > exact and not roundUp:
        r -= 1
    elif value < exact and roundUp:
        r += 1
    return r


def _run(asm, sgprs, roundUp):
    """Run the emitted text; VGPRs are LANES-wide vectors, SGPRs scalars; returns the SGPR file."""
    regs = {f"s{i}": v for i, v in sgprs.items()}
    vregs = {}
    scc = 0

    def s(operand):
        return regs[operand] if operand in regs else int(operand, 0) & U32

    def v(operand):
        if operand in vregs:
            return vregs[operand]
        return [s(operand)] * LANES

    lanewise = {
        "v_cvt_f32_u32": lambda a: _bits(float(a)),
        "v_rcp_iflag_f32": lambda a: _rcp(a, roundUp),
        "v_cvt_u32_f32": _cvt_u32,
        "v_mov_b32": lambda a: a,
        "v_mul_f32": lambda a, b: _bits(_f32(a) * _f32(b)),
    }
    for line in asm.splitlines():
        line = line.split("//")[0].strip()
        if not line or line.endswith(":"):
            continue
        op, _, rest = line.partition(" ")
        dst, *srcs = [x.strip() for x in rest.split(",")]
        assert "exec" not in (dst, *srcs), f"the SALU helpers must not write exec: {line}"
        if op in lanewise:
            vregs[dst] = [lanewise[op](*args) for args in zip(*map(v, srcs))]
        elif op == "v_writelane_b32":
            lanes = list(v(dst))
            lanes[s(srcs[1])] = s(srcs[0])
            vregs[dst] = lanes
        elif op == "v_readfirstlane_b32":
            regs[dst] = v(srcs[0])[0]
        elif op == "v_readlane_b32":
            regs[dst] = v(srcs[0])[s(srcs[1])]
        elif op == "s_mul_i32":
            regs[dst] = (s(srcs[0]) * s(srcs[1])) & U32
        elif op == "s_mul_hi_u32":
            regs[dst] = (s(srcs[0]) * s(srcs[1])) >> 32
        elif op == "s_add_u32":
            total = s(srcs[0]) + s(srcs[1])
            regs[dst], scc = total & U32, int(total > U32)
        elif op == "s_sub_u32":
            a, b = s(srcs[0]), s(srcs[1])
            regs[dst], scc = (a - b) & U32, int(b > a)
        elif op == "s_addc_u32":
            total = s(srcs[0]) + s(srcs[1]) + scc
            regs[dst], scc = total & U32, int(total > U32)
        elif op == "s_cmp_ge_u32":
            scc = int(s(dst) >= s(srcs[0]))
        elif op == "s_cselect_b32":
            regs[dst] = s(srcs[0]) if scc else s(srcs[1])
        else:
            raise AssertionError(f"emulator does not model: {line}")
    return regs


def _cases():
    # Below 2^24 v_cvt_f32_u32 is exact, so both rcp neighbours are modeled; real prologue
    # divisors (|WGM|, WGMXCC, CU_Count, chunk sizes, tile and cluster counts) are far below it.
    edges = [0, 1, 2, 3, 255, 256, 257, (1 << 24) - 1, 1 << 24, (1 << 24) + 1, (1 << 31) - 1,
             1 << 31, (1 << 31) + 1, U32 - 1, U32]
    divisors = [1, 2, 3, 7, 8, 13, 255, 256, 257, 4095, 4096, 4097, 65535, 65536, 75371,
                16781311, (1 << 24) - 3, (1 << 24) - 1]
    cases = [(n, d) for d in divisors for n in edges + [d - 1, d, d + 1, 2 * d - 1, 3 * d - 1]]
    rng = random.Random(15654)
    for _ in range(300):
        cases.append((rng.getrandbits(32), rng.getrandbits(rng.randint(1, 24)) or 1))
    return cases


@pytest.fixture(scope="module")
def _gfx1250():
    """Emit for gfx1250, then restore the kernel ISA that was set before (other tests in the
    same worker read the current ISA without setting it)."""
    ri = rocIsa.getInstance()
    previous = ri.getKernel()
    previous = (tuple(previous.isa), previous.wavefrontSize)
    ri.init(GFX1250, shutil.which("amdclang++") or "/usr/bin/amdclang++")
    ri.setKernel(GFX1250, 32)
    yield
    ri.setKernel(*previous)


@pytest.mark.parametrize("roundUp", [False, True])
@pytest.mark.parametrize("numDivisors", [1, 2, 3])
def test_salu_divide_matches_python(_gfx1250, roundUp, numDivisors):
    # Divisor i in s(20+i), dividend i in s(30+i); reciprocal i in s(40+i); s50 scratch.
    # Quotient i goes to s(40+i) itself (the reciprocal register, as the call sites do), remainder to s(60+i).
    divs, recs = [20 + i for i in range(numDivisors)], [40 + i for i in range(numDivisors)]
    asm = str(saluUInt32Reciprocals(divs, recs, 50, ContinuousRegister(idx=8, size=2)))
    for i in range(numDivisors):
        asm += str(saluUInt32DivTail(40 + i, 30 + i, 20 + i, 40 + i, 60 + i, 50))

    assert asm.count("v_rcp_iflag_f32") == 1
    assert "v_rcp_f64" not in asm and "v_cmpx" not in asm

    cases = _cases()
    wrong = []
    for k in range(0, len(cases), numDivisors):
        group = cases[k:k + numDivisors]
        if len(group) < numDivisors:
            break
        sgprs = {}
        for i, (n, d) in enumerate(group):
            sgprs[20 + i], sgprs[30 + i] = d, n
        regs = _run(asm, sgprs, roundUp)
        for i, (n, d) in enumerate(group):
            got = (regs[f"s{40 + i}"], regs[f"s{60 + i}"])
            if got != divmod(n, d):
                wrong.append(f"{n} / {d}: got {got}, want {divmod(n, d)}")
    assert not wrong, f"{len(wrong)} wrong, first: " + "; ".join(wrong[:5])


@pytest.mark.parametrize("roundUp", [False, True])
def test_salu_divide_shared_reciprocal(_gfx1250, roundUp):
    """One reciprocal, several tails (DefaultWGM / WGMXCC / K-Coherent form)."""
    asm = str(saluUInt32Reciprocals([20], [40], 50, ContinuousRegister(idx=8, size=2)))
    asm += str(saluUInt32DivTail(41, 30, 20, 40, 60, 50))
    asm += str(saluUInt32DivTail(40, 31, 20, 40, 61, 50))  # last tail overwrites the reciprocal
    wrong = []
    for n, d in _cases():
        regs = _run(asm, {20: d, 30: n, 31: U32 - n}, roundUp)
        got = (regs["s41"], regs["s60"], regs["s40"], regs["s61"])
        want = (*divmod(n, d), *divmod(U32 - n, d))
        if got != want:
            wrong.append(f"[{n}, {U32 - n}] / {d}: got {got}, want {want}")
    assert not wrong, f"{len(wrong)} wrong, first: " + "; ".join(wrong[:5])
