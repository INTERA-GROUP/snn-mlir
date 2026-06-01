# Copyright 2026 Sensing & Control Systems, S.L.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Verify rescale injection fires for any is_synapse→is_neuron edge."""

import numpy as np
from snn_mlir._graph import insert_rescale_nodes
from snn_mlir.nodes._rescale import RescaleInfo
from snn_mlir.nodes.cubali import CubaLIInfo
from snn_mlir.nodes.cubalif import CubaLIFInfo
from snn_mlir.nodes.li import LIInfo
from snn_mlir.nodes.lif import LIFInfo
from snn_mlir.nodes.linear import LinearInfo


def _make_linear(name="0", size_in=8, size_out=16):
    info = LinearInfo(
        name=name,
        input_size=size_in,
        output_size=size_out,
        weights=np.random.uniform(-0.5, 0.5, (size_out, size_in)).astype(np.float32),
    )
    info.quantize()
    return info


def _make_cubalif(name="1", size=16):
    info = CubaLIFInfo(name=name, size=size, cur_decay=0.9, vol_decay=0.95, threshold=1.0)
    info.quantize()
    return info


def _make_lif(name="1", size=16):
    info = LIFInfo(name=name, size=size, decay=0.9, threshold=1.0)
    info.quantize()
    return info


def _make_li(name="1", size=16):
    info = LIInfo(name=name, size=size, decay=0.9)
    info.quantize()
    return info


def _make_cubali(name="1", size=16):
    info = CubaLIInfo(name=name, size=size, cur_decay=0.9, vol_decay=0.95)
    info.quantize()
    return info


def test_rescale_injected_for_cubalif():
    layers = [_make_linear(), _make_cubalif()]
    result = insert_rescale_nodes(layers)
    assert len(result) == 3
    assert isinstance(result[1], RescaleInfo)


def test_rescale_injected_for_lif():
    layers = [_make_linear(), _make_lif()]
    result = insert_rescale_nodes(layers)
    assert isinstance(result[1], RescaleInfo)


def test_rescale_injected_for_li():
    layers = [_make_linear(), _make_li()]
    result = insert_rescale_nodes(layers)
    assert isinstance(result[1], RescaleInfo)


def test_rescale_injected_for_cubali():
    layers = [_make_linear(), _make_cubali()]
    result = insert_rescale_nodes(layers)
    assert isinstance(result[1], RescaleInfo)


def test_rescale_scales_match():
    lin = _make_linear()
    neu = _make_cubalif()
    result = insert_rescale_nodes([lin, neu])
    rescale = result[1]
    assert rescale._w_scale == lin.w_scale
    assert rescale._d_scale == neu.d_scale


def test_rescale_mlir_contains_snn_rescale():
    rescale = RescaleInfo(name="0", size=16, _w_scale=7, _d_scale=12)
    lines, out_var = rescale.emit_mlir("%synapse_0", False, True)
    text = "\n".join(lines)
    assert "snn.rescale" in text
    assert "w_scale = 7" in text
    assert "d_scale = 12" in text
    assert out_var == "%rescaled_0"


def test_two_layer_chain_has_two_rescales():
    layers = [
        _make_linear("0", 8, 16),
        _make_cubalif("1", 16),
        _make_linear("2", 16, 8),
        _make_lif("3", 8),
    ]
    result = insert_rescale_nodes(layers)
    rescales = [n for n in result if isinstance(n, RescaleInfo)]
    assert len(rescales) == 2
