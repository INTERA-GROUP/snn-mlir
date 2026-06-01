# Copyright 2026 Sensing & Control Systems, S.L.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from snn_mlir.nodes.li import LIInfo


def test_is_neuron_trait(li_float):
    assert li_float.is_neuron is True
    assert li_float.is_synapse is False


def test_float_mlir_contains_snn_li(li_float):
    lines, out_var = li_float.emit_mlir("%in", True, False)
    text = "\n".join(lines)
    assert "snn.li" in text
    assert "decay_float" in text
    assert out_var == "%output"


def test_quantized_mlir_uses_q12(li_quantized):
    lines, _ = li_quantized.emit_mlir("%in", True, True)
    text = "\n".join(lines)
    assert "d_scale = 12" in text
    assert "decay_int" in text


def test_q12_decay_scaling():
    info = LIInfo(name="2", size=8, decay=0.9)
    info.quantize()
    # 0.9 * 4096 ≈ 3686
    assert 3600 < info.decay_scaled < 3800


def test_output_type_quantized(li_quantized):
    assert li_quantized.output_element_type(True) == "i32"


def test_single_state_arg(li_float):
    args = li_float.state_func_args(False)
    assert len(args) == 1
    assert args[0][0] == "%voltage_2"


def test_intermediate_node_allocates(li_float):
    lines, out_var = li_float.emit_mlir("%in", False, False)
    assert "memref.alloca" in "\n".join(lines)
    assert out_var == "%voltage_out_2"
