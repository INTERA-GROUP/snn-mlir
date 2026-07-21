# Copyright 2026 N Vision Systems And Technologies SL
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import nir
import numpy as np
import pytest
from snn_mlir.nodes.li import LIInfo, parse_li


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


def test_parse_li_from_real_nir_node():
    # Regression: parse_li used to read node.w_in, which nir.LI has no such field,
    # so snn.li was unreachable from any real graph. It must parse like a LIF
    # without firing: decay derived from tau/r, not from v_leak.
    node = nir.LI(
        tau=np.full(4, 2.0),
        r=np.full(4, 4.0),
        v_leak=np.zeros(4),
        input_type={"input": np.array([4])},
    )
    info = parse_li(node, "li0")
    assert info.size == 4
    dt = 2.0 / 4.0
    assert info.decay == pytest.approx(1 - dt / 2.0)


def test_parse_li_rejects_nonzero_v_leak():
    node = nir.LI(
        tau=np.full(4, 2.0),
        r=np.full(4, 4.0),
        v_leak=np.full(4, 0.3),
        input_type={"input": np.array([4])},
    )
    with pytest.raises(ValueError, match="v_leak must be 0"):
        parse_li(node, "li0")
