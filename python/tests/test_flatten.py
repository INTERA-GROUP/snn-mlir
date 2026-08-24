# Copyright 2026 N Vision Systems And Technologies SL
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""``nir.Flatten`` → ``memref.collapse_shape``.

The rank boundary between the convolutional body and the dense tail. It carries
no weights, no state and does no arithmetic — it reinterprets one contiguous
buffer at a lower rank — so it lowers to a plain ``memref.collapse_shape`` and
is element-type agnostic (it needs no float/quant guard).
"""

import nir
from snn_mlir.nodes import NODE_PARSERS
from snn_mlir.nodes.flatten import FlattenInfo, parse_flatten


def _parse(start_dim=0, end_dim=-1, name="f0"):
    return parse_flatten(nir.Flatten(start_dim=start_dim, end_dim=end_dim), name)


# ── shape algebra ─────────────────────────────────────────────────────────────


def test_unfilled_shape_reads_as_unconstrained():
    info = _parse()
    assert isinstance(info, FlattenInfo)
    assert info.in_shape is None
    assert info.out_shape is None


def test_full_collapse_to_rank1():
    # start=0, end=-1: the nmnistcnn Flatten, (8, 4, 4) -> (128).
    info = _parse(start_dim=0, end_dim=-1)
    info.adopt_in_shape((8, 4, 4))
    assert info.in_shape == (8, 4, 4)
    assert info.out_shape == (128,)


def test_partial_collapse_keeps_outer_dims():
    # Fold only the trailing two dims: (2, 3, 4) -> (2, 12).
    info = _parse(start_dim=1, end_dim=-1)
    info.adopt_in_shape((2, 3, 4))
    assert info.out_shape == (2, 12)


def test_positive_end_dim():
    info = _parse(start_dim=0, end_dim=1)
    info.adopt_in_shape((2, 3, 4))
    assert info.out_shape == (6, 4)


# ── emission ──────────────────────────────────────────────────────────────────


def test_registered():
    assert NODE_PARSERS[nir.Flatten] is parse_flatten


def test_is_neither_synapse_nor_neuron():
    info = _parse()
    assert not info.is_synapse
    assert not info.is_neuron


def test_emits_collapse_shape_full_reassociation():
    info = _parse(start_dim=0, end_dim=-1, name="f8")
    info.adopt_in_shape((8, 4, 4))
    lines, out_var = info.emit_mlir("%pool", is_last=False, quantize=False)
    text = "\n".join(lines)
    assert "memref.collapse_shape %pool [[0, 1, 2]]" in text
    assert "memref<8x4x4xf32> into memref<128xf32>" in text
    assert out_var == "%flat_f8"


def test_emission_is_element_type_agnostic():
    """Flatten works on spikes too: quantized mode reinterprets an i8 buffer."""
    info = _parse(start_dim=0, end_dim=-1)
    info.adopt_in_shape((8, 4, 4))
    lines, _ = info.emit_mlir("%pool", is_last=False, quantize=True)
    text = "\n".join(lines)
    assert "memref<8x4x4xi8> into memref<128xi8>" in text


def test_partial_collapse_reassociation_groups():
    info = _parse(start_dim=1, end_dim=-1)
    info.adopt_in_shape((2, 3, 4))
    lines, _ = info.emit_mlir("%x", is_last=False, quantize=False)
    text = "\n".join(lines)
    assert "memref.collapse_shape %x [[0], [1, 2]]" in text
    assert "memref<2x3x4xf32> into memref<2x12xf32>" in text
