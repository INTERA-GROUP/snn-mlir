# Copyright 2026 N Vision Systems And Technologies SL
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""``nir.SumPool2d`` → ``snn.sumpool2d``.

A rank-changing layer that carries no weights and no state: it reduces each
spatial window to its sum, per channel. Its output geometry depends on the
input spatial size, so — like a conv — the shape it produces is computed, not
fixed by any parameter.
"""

import nir
import numpy as np
import pytest
from snn_mlir.nodes import NODE_PARSERS
from snn_mlir.nodes.sumpool2d import SumPool2dInfo, parse_sumpool2d


def _pool_node(C=16, H=16, W=16, kernel=2, stride=2, padding=0):
    node = nir.SumPool2d(
        kernel_size=np.array([kernel, kernel]),
        stride=np.array([stride, stride]),
        padding=np.array([padding, padding]),
    )
    node.input_type = {"input": np.array([C, H, W])}
    return node


# ── shape algebra ─────────────────────────────────────────────────────────────


def test_parse_reads_window_and_input_shape():
    info = parse_sumpool2d(_pool_node(C=16, H=16, W=16), "p0")
    assert isinstance(info, SumPool2dInfo)
    assert info.kernel == (2, 2)
    assert info.stride == (2, 2)
    assert info.padding == (0, 0)
    assert info.in_shape == (16, 16, 16)


def test_out_shape_follows_the_pooling_formula():
    # (16 + 0 - 2)/2 + 1 = 8, non-overlapping 2x2 — the nmnistcnn pools.
    info = parse_sumpool2d(_pool_node(C=16, H=16, W=16), "p0")
    assert info.out_shape == (16, 8, 8)


def test_pooling_preserves_channels():
    info = parse_sumpool2d(_pool_node(C=8, H=8, W=8), "p0")
    assert info.out_shape[0] == 8


def test_adopt_in_shape_reflows_output_geometry():
    info = parse_sumpool2d(_pool_node(C=16, H=16, W=16), "p0")
    info.adopt_in_shape((8, 8, 8))
    assert info.in_shape == (8, 8, 8)
    assert info.out_shape == (8, 4, 4)


def test_scalar_window_fields_expand_to_pairs():
    node = nir.SumPool2d(kernel_size=2, stride=2, padding=0)
    node.input_type = {"input": np.array([4, 6, 6])}
    info = parse_sumpool2d(node, "p0")
    assert info.kernel == (2, 2)
    assert info.out_shape == (4, 3, 3)


def test_rejects_non_rank3_input():
    node = _pool_node()
    node.input_type = {"input": np.array([16, 16])}
    with pytest.raises(ValueError, match="rank-3"):
        parse_sumpool2d(node, "p0")


# ── emission ──────────────────────────────────────────────────────────────────


def test_registered():
    assert NODE_PARSERS[nir.SumPool2d] is parse_sumpool2d


def test_is_neither_synapse_nor_neuron():
    info = parse_sumpool2d(_pool_node(), "p0")
    assert not info.is_synapse
    assert not info.is_neuron


def test_emits_snn_sumpool2d_with_window_attrs():
    info = parse_sumpool2d(_pool_node(C=16, H=16, W=16, kernel=2, stride=2), "p0")
    lines, out_var = info.emit_mlir("%input", is_last=False, quantize=False)
    text = "\n".join(lines)
    assert "snn.sumpool2d" in text
    assert "kernel = array<i64: 2, 2>" in text
    assert "stride = array<i64: 2, 2>" in text
    assert "memref<16x16x16xf32> -> memref<16x8x8xf32>" in text
    assert out_var == "%pool_p0"


def test_carries_no_weight_globals():
    info = parse_sumpool2d(_pool_node(), "p0")
    assert info.weight_globals(quantize=False) == []


def test_quantized_path_is_not_implemented_yet():
    info = parse_sumpool2d(_pool_node(), "p0")
    with pytest.raises(NotImplementedError):
        info.emit_mlir("%input", is_last=False, quantize=True)
