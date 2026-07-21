# Copyright 2026 N Vision Systems And Technologies SL
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
def test_float_mlir_contains_snn_linear(linear_float):
    lines, out_var = linear_float.emit_mlir("%input", False, False)
    text = "\n".join(lines)
    assert "snn.linear" in text
    assert "memref<16x8xf32>" in text
    assert out_var == "%synapse_0"


def test_float_mlir_reads_weight_global(linear_float):
    lines, _ = linear_float.emit_mlir("%input", False, False)
    text = "\n".join(lines)
    assert "memref.get_global @w_0 : memref<16x8xf32>" in text


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


def test_weight_globals_float(linear_float):
    globals_ = linear_float.weight_globals(False)
    assert len(globals_) == 1
    assert 'memref.global "private" constant @w_0' in globals_[0]
    assert "memref<16x8xf32>" in globals_[0]
    assert "dense<" in globals_[0]


def test_weight_globals_with_bias(linear_float_bias):
    globals_ = linear_float_bias.weight_globals(False)
    assert len(globals_) == 2
    assert "@w_0" in globals_[0]
    assert "@b_0" in globals_[1]


def test_weight_globals_quantized(linear_quantized):
    globals_ = linear_quantized.weight_globals(True)
    assert "memref<16x8xi8>" in globals_[0]
    assert "dense<" in globals_[0]


def test_quantize_clamps_w_scale_to_d_scale():
    # Tiny weights push floor(log2(127/max|w|)) above the neuron's Q12 scale,
    # which would make the rescale shift (d_scale - w_scale) negative. The clamp
    # caps w_scale at d_scale so the shift stays non-negative.
    import numpy as np
    from snn_mlir.nodes.linear import _D_SCALE, LinearInfo

    info = LinearInfo(
        name="0", input_size=4, output_size=4, weights=np.full((4, 4), 1e-3, dtype=np.float32)
    )
    info.quantize()
    assert info.w_scale == _D_SCALE


def test_quantize_leaves_normal_w_scale_unclamped():
    import numpy as np
    from snn_mlir.nodes.linear import _D_SCALE, LinearInfo

    info = LinearInfo(
        name="0", input_size=4, output_size=4, weights=np.full((4, 4), 0.5, dtype=np.float32)
    )
    info.quantize()
    assert info.w_scale < _D_SCALE
