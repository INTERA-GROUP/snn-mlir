# Copyright 2026 Sensing & Control Systems, S.L.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Integration tests: NIR graph → MLIR string structure."""

from snn_mlir._emit import generate_mlir
from snn_mlir._graph import insert_rescale_nodes, parse_graph, quantize_layers
from snn_mlir.nodes.cubalif import CubaLIFInfo
from snn_mlir.nodes.linear import LinearInfo


def test_parse_graph_returns_two_layers(nir_linear_cubalif):
    layers = parse_graph(nir_linear_cubalif)
    assert len(layers) == 2
    assert isinstance(layers[0], LinearInfo)
    assert isinstance(layers[1], CubaLIFInfo)


def test_float_mlir_has_func_and_return(nir_linear_cubalif):
    layers = parse_graph(nir_linear_cubalif)
    mlir = generate_mlir(layers, quantize=False)
    assert "func.func @snn_forward_step" in mlir
    assert "snn.linear" in mlir
    assert "snn.cubalif" in mlir
    assert "return" in mlir
    assert "snn.rescale" not in mlir


def test_quantized_mlir_has_rescale(nir_linear_cubalif):
    layers = parse_graph(nir_linear_cubalif)
    quantize_layers(layers)
    emit_layers = insert_rescale_nodes(layers)
    mlir = generate_mlir(emit_layers, quantize=True)
    assert "snn.rescale" in mlir
    assert "snn.linear" in mlir
    assert "snn.cubalif" in mlir
    assert "w_scale" in mlir
    assert "d_scale" in mlir


def test_quantized_mlir_input_is_i8(nir_linear_cubalif):
    layers = parse_graph(nir_linear_cubalif)
    quantize_layers(layers)
    emit_layers = insert_rescale_nodes(layers)
    mlir = generate_mlir(emit_layers, quantize=True)
    assert "memref<8xi8>" in mlir


def test_float_mlir_output_is_f32(nir_linear_cubalif):
    layers = parse_graph(nir_linear_cubalif)
    mlir = generate_mlir(layers, quantize=False)
    assert "memref<16xf32>" in mlir


def test_emit_c_interface_attribute(nir_linear_cubalif):
    layers = parse_graph(nir_linear_cubalif)
    mlir = generate_mlir(layers, quantize=False)
    assert "llvm.emit_c_interface" in mlir
