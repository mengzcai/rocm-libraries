# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""FastPrologueDivide is a gfx1250-only option.

On gfx1250 (ISA 12.5.0) the work-group remap prologue uses the exact 32-bit
divide by default, and only the non-default FastPrologueDivide=0 is named
(FPD0). Every other ISA keeps the f64-reciprocal divide and the kernel name it
had before the option existed, whatever value the option has.
"""

import shutil
from copy import deepcopy

import pytest

from rocisa import rocIsa
from rocisa.container import ContinuousRegister

from tensilelite.Common.GlobalParameters import defaultSolution
from tensilelite.Components.WorkGroupMappingAlgos import prologueDivideAndRemainder
from tensilelite.SolutionStructs.Naming import getKernelNameMin

pytestmark = pytest.mark.unit

GFX1250 = (12, 5, 0)
OTHER_ISAS = [(9, 4, 2), (9, 5, 0)]


def _kernel(isa, fpd):
    """Build the smallest kernel dict that exercises Naming._getName and the divide helper."""
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


def _divide_asm(isa, fpd):
    kernel = _kernel(isa, fpd)
    ri = rocIsa.getInstance()
    ri.init(isa, shutil.which("amdclang++") or "/usr/bin/amdclang++")
    ri.setKernel(isa, kernel["WavefrontSize"])
    return str(prologueDivideAndRemainder(kernel, qReg=4, dReg=12, divReg=13, rReg=14,
                                          tmpVgprRes=ContinuousRegister(idx=8, size=4)))


@pytest.mark.parametrize("isa,fpd,exactU32", [(GFX1250, 1, True), (GFX1250, 0, False)]
                         + [(isa, fpd, False) for isa in OTHER_ISAS for fpd in (0, 1)])
def test_prologue_divide_is_u32_only_on_gfx1250(isa, fpd, exactU32):
    asm = _divide_asm(isa, fpd)

    assert ("v_rcp_iflag_f32" in asm) == exactU32
    assert ("v_rcp_f64" in asm) == (not exactU32)
