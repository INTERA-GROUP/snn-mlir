# Copyright 2026 N Vision Systems And Technologies SL
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import nir
import numpy as np
import pytest
from snn_mlir.nodes.lif import LIFInfo, parse_lif


def test_is_neuron_trait(lif_float):
    assert lif_float.is_neuron is True
    assert lif_float.is_synapse is False


def test_float_mlir_contains_snn_lif(lif_float):
    lines, out_var = lif_float.emit_mlir("%in", True, False)
    text = "\n".join(lines)
    assert "snn.lif" in text
    assert "decay_float" in text
    assert "threshold_float" in text
    assert "v_reset_float" in text
    assert out_var == "%output"


def test_quantized_mlir_uses_q12(lif_quantized):
    lines, _ = lif_quantized.emit_mlir("%in", True, True)
    text = "\n".join(lines)
    assert "d_scale = 12" in text
    assert "decay_int" in text
    assert "threshold_int" in text
    assert "v_reset_int" in text


def test_q12_scaling():
    info = LIFInfo(name="2", shape=(8,), decay=0.9, threshold=1.0, v_reset=0.0)
    info.quantize()
    # 0.9 * 4096 ≈ 3686
    assert 3600 < info.decay_scaled < 3800
    # 1.0 * 4096 = 4096
    assert info.threshold_scaled == 4096
    # 0.0 * 4096 = 0
    assert info.v_reset_scaled == 0


def test_output_type_quantized(lif_quantized):
    assert lif_quantized.output_element_type(True) == "i8"


def test_single_state_arg(lif_float):
    args = lif_float.state_func_args(False)
    assert len(args) == 1
    assert args[0][0] == "%voltage_2"


def test_intermediate_node_allocates(lif_float):
    lines, out_var = lif_float.emit_mlir("%in", False, False)
    assert "memref.alloca" in "\n".join(lines)
    assert out_var == "%spikes_2"


def _lif_node(v_reset):
    return nir.LIF(
        tau=np.full(4, 2.0),
        r=np.full(4, 4.0),
        v_leak=np.zeros(4),
        v_threshold=np.ones(4),
        v_reset=np.full(4, v_reset),
        input_type={"input": np.array([4])},
    )


def test_parse_lif_accepts_zero_v_reset():
    info = parse_lif(_lif_node(0.0), "l0")
    assert info.size == 4


def test_parse_lif_rejects_nonzero_v_reset():
    # A nonzero reset is not yet supported by the quantized lowering; it must be
    # rejected loudly rather than silently produce a divergent network.
    with pytest.raises(ValueError, match="v_reset != 0 not supported"):
        parse_lif(_lif_node(0.3), "l0")
