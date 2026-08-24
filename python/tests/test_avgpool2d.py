# Copyright 2026 N Vision Systems And Technologies SL
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""``nir.AvgPool2d`` → ``snn.avgpool2d``.

Mirrors sum pooling — no weights, no state, rank-changing — but averages each
window instead of summing it, which the lowering does with a trailing divide.
"""

import nir
import numpy as np
import pytest
from snn_mlir.nodes import NODE_PARSERS
from snn_mlir.nodes.avgpool2d import AvgPool2dInfo, parse_avgpool2d


def _pool_node(C=16, H=16, W=16, kernel=2, stride=2, padding=0):
    node = nir.AvgPool2d(
        kernel_size=np.array([kernel, kernel]),
        stride=np.array([stride, stride]),
        padding=np.array([padding, padding]),
    )
    node.input_type = {"input": np.array([C, H, W])}
    return node


# ── shape algebra ─────────────────────────────────────────────────────────────


def test_parse_reads_window_and_input_shape():
    info = parse_avgpool2d(_pool_node(C=16, H=16, W=16), "p0")
    assert isinstance(info, AvgPool2dInfo)
    assert info.kernel == (2, 2)
    assert info.stride == (2, 2)
    assert info.padding == (0, 0)
    assert info.in_shape == (16, 16, 16)


def test_out_shape_follows_the_pooling_formula():
    info = parse_avgpool2d(_pool_node(C=16, H=16, W=16), "p0")
    assert info.out_shape == (16, 8, 8)


def test_pooling_preserves_channels():
    info = parse_avgpool2d(_pool_node(C=8, H=8, W=8), "p0")
    assert info.out_shape[0] == 8


def test_adopt_in_shape_reflows_output_geometry():
    info = parse_avgpool2d(_pool_node(C=16, H=16, W=16), "p0")
    info.adopt_in_shape((8, 8, 8))
    assert info.in_shape == (8, 8, 8)
    assert info.out_shape == (8, 4, 4)


def test_scalar_window_fields_expand_to_pairs():
    node = nir.AvgPool2d(kernel_size=2, stride=2, padding=0)
    node.input_type = {"input": np.array([4, 6, 6])}
    info = parse_avgpool2d(node, "p0")
    assert info.kernel == (2, 2)
    assert info.out_shape == (4, 3, 3)


def test_rejects_non_rank3_input():
    node = _pool_node()
    node.input_type = {"input": np.array([16, 16])}
    with pytest.raises(ValueError, match="rank-3"):
        parse_avgpool2d(node, "p0")


# ── emission ──────────────────────────────────────────────────────────────────


def test_registered():
    assert NODE_PARSERS[nir.AvgPool2d] is parse_avgpool2d


def test_is_neither_synapse_nor_neuron():
    info = parse_avgpool2d(_pool_node(), "p0")
    assert not info.is_synapse
    assert not info.is_neuron


def test_emits_snn_avgpool2d_with_window_attrs():
    info = parse_avgpool2d(_pool_node(C=16, H=16, W=16, kernel=2, stride=2), "p0")
    lines, out_var = info.emit_mlir("%input", is_last=False, quantize=False)
    text = "\n".join(lines)
    assert "snn.avgpool2d" in text
    assert "kernel = array<i64: 2, 2>" in text
    assert "stride = array<i64: 2, 2>" in text
    assert "memref<16x16x16xf32> -> memref<16x8x8xf32>" in text
    assert out_var == "%pool_p0"


def test_carries_no_weight_globals():
    info = parse_avgpool2d(_pool_node(), "p0")
    assert info.weight_globals(quantize=False) == []


def test_quantized_path_is_not_implemented_yet():
    info = parse_avgpool2d(_pool_node(), "p0")
    with pytest.raises(NotImplementedError):
        info.emit_mlir("%input", is_last=False, quantize=True)
