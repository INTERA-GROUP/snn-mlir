# Copyright 2026 Sensing & Control Systems, S.L.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
def test_float_mlir_contains_snn_linear(linear_float):
    lines, out_var = linear_float.emit_mlir("%input", False, False)
    text = "\n".join(lines)
    assert "snn.linear" in text
    assert "memref<16x8xf32>" in text
    assert out_var == "%synapse_0"


def test_float_mlir_no_w_scale(linear_float):
    lines, _ = linear_float.emit_mlir("%input", False, False)
    assert "w_scale" not in "\n".join(lines)


def test_quantized_mlir_contains_w_scale(linear_quantized):
    lines, _ = linear_quantized.emit_mlir("%input", False, True)
    text = "\n".join(lines)
    assert "w_scale" in text
    assert "i8" in text
    assert "i32" in text


def test_quantized_weights_in_range(linear_quantized):
    assert linear_quantized.int8_weights is not None
    assert linear_quantized.int8_weights.min() >= -128
    assert linear_quantized.int8_weights.max() <= 127


def test_bias_appears_in_float_mlir(linear_float_bias):
    lines, _ = linear_float_bias.emit_mlir("%input", False, False)
    text = "\n".join(lines)
    assert "bias(" in text


def test_is_synapse_trait(linear_float):
    assert linear_float.is_synapse is True
    assert linear_float.is_neuron is False


def test_weight_func_args_float(linear_float):
    args = linear_float.weight_func_args(False)
    assert len(args) == 1
    assert "f32" in args[0][1]


def test_weight_func_args_with_bias(linear_float_bias):
    args = linear_float_bias.weight_func_args(False)
    assert len(args) == 2
