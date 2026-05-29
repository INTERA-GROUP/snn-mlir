# SPDX-License-Identifier: Apache-2.0
from snn_mlir.nodes.cubalif import CubaLIFInfo


def test_float_mlir_contains_snn_cubalif(cubalif_float):
    lines, out_var = cubalif_float.emit_mlir("%rescaled_0", True, False)
    text = "\n".join(lines)
    assert "snn.cubalif" in text
    assert "cur_decay_float" in text
    assert out_var == "%output"


def test_quantized_mlir_uses_q12(cubalif_quantized):
    lines, _ = cubalif_quantized.emit_mlir("%rescaled_0", True, True)
    text = "\n".join(lines)
    assert "d_scale = 12" in text
    assert "cur_decay_int" in text
    assert "threshold_int" in text


def test_q12_decay_range():
    info = CubaLIFInfo(name="1", size=8, cur_decay=0.9, vol_decay=0.95, threshold=1.0)
    info.quantize()
    # Q12: value * 4096; 0.9 * 4096 ≈ 3686
    assert 3600 < info.cur_decay_scaled < 3800
    # 1.0 * 4096 = 4096
    assert info.threshold_scaled == 4096


def test_is_neuron_trait(cubalif_float):
    assert cubalif_float.is_neuron is True
    assert cubalif_float.is_synapse is False


def test_state_func_args_float(cubalif_float):
    args = cubalif_float.state_func_args(False)
    assert len(args) == 2
    names = [a[0] for a in args]
    assert "%current_1" in names
    assert "%voltage_1" in names


def test_output_type_quantized(cubalif_quantized):
    assert cubalif_quantized.output_element_type(True) == "i8"


def test_intermediate_node_allocates(cubalif_float):
    lines, out_var = cubalif_float.emit_mlir("%in", False, False)
    assert "memref.alloca" in "\n".join(lines)
    assert out_var == "%spikes_1"
