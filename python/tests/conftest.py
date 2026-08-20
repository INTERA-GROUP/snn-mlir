# Copyright 2026 N Vision Systems And Technologies SL
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Fixtures: synthetic NIR graphs. No file I/O, no snn-opt needed for unit tests."""

import nir
import numpy as np
import pytest
from snn_mlir.nodes.cubali import CubaLIInfo
from snn_mlir.nodes.cubalif import CubaLIFInfo
from snn_mlir.nodes.li import LIInfo
from snn_mlir.nodes.lif import LIFInfo
from snn_mlir.nodes.linear import LinearInfo

# ── raw NodeInfo fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def linear_float():
    return LinearInfo(
        name="0",
        input_size=8,
        output_size=16,
        weights=np.random.uniform(-0.5, 0.5, (16, 8)).astype(np.float32),
    )


@pytest.fixture
def linear_float_bias():
    return LinearInfo(
        name="0",
        input_size=8,
        output_size=16,
        weights=np.random.uniform(-0.5, 0.5, (16, 8)).astype(np.float32),
        bias=np.random.uniform(-0.1, 0.1, (16,)).astype(np.float32),
    )


@pytest.fixture
def linear_quantized():
    info = LinearInfo(
        name="0",
        input_size=8,
        output_size=16,
        weights=np.random.uniform(-0.5, 0.5, (16, 8)).astype(np.float32),
    )
    info.quantize()
    return info


@pytest.fixture
def cubalif_float():
    return CubaLIFInfo(
        name="1",
        shape=(16,),
        cur_decay=0.9,
        vol_decay=0.95,
        threshold=1.0,
    )


@pytest.fixture
def cubalif_quantized():
    info = CubaLIFInfo(
        name="1",
        shape=(16,),
        cur_decay=0.9,
        vol_decay=0.95,
        threshold=1.0,
    )
    info.quantize()
    return info


@pytest.fixture
def cubali_float():
    return CubaLIInfo(name="2", shape=(16,), cur_decay=0.9, vol_decay=0.95)


@pytest.fixture
def cubali_quantized():
    info = CubaLIInfo(name="2", shape=(16,), cur_decay=0.9, vol_decay=0.95)
    info.quantize()
    return info


@pytest.fixture
def lif_float():
    return LIFInfo(name="2", shape=(16,), decay=0.9, threshold=1.0, v_reset=0.0)


@pytest.fixture
def lif_quantized():
    info = LIFInfo(name="2", shape=(16,), decay=0.9, threshold=1.0, v_reset=0.0)
    info.quantize()
    return info


@pytest.fixture
def li_float():
    return LIInfo(name="2", shape=(16,), decay=0.9)


@pytest.fixture
def li_quantized():
    info = LIInfo(name="2", shape=(16,), decay=0.9)
    info.quantize()
    return info


# ── NIR graph fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def nir_linear_cubalif():
    """Minimal 2-node NIR graph: Linear(8→16) → CubaLIF(16)."""
    W = np.random.uniform(-0.5, 0.5, (16, 8)).astype(np.float32)
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([8])}),
            "linear": nir.Linear(weight=W),
            "cubalif": nir.CubaLIF(
                tau_syn=np.full(16, 0.1),
                tau_mem=np.full(16, 0.05),
                r=np.full(16, 0.5),
                v_leak=np.zeros(16),
                v_threshold=np.ones(16),
                v_reset=np.zeros(16),
                input_type={"input": np.array([16])},
            ),
            "output": nir.Output(output_type={"output": np.array([16])}),
        },
        edges=[("input", "linear"), ("linear", "cubalif"), ("cubalif", "output")],
    )
