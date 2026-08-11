# Copyright 2026 N Vision Systems And Technologies SL
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import nir
import numpy as np
import pytest
from snn_mlir.nodes.cubali import CubaLIInfo, parse_cubali


def test_is_neuron_trait(cubali_float):
    assert cubali_float.is_neuron is True
    assert cubali_float.is_synapse is False


def test_float_mlir_contains_snn_cubali(cubali_float):
    lines, out_var = cubali_float.emit_mlir("%in", True, False)
    text = "\n".join(lines)
    assert "snn.cubali" in text
    assert "cur_decay_float" in text
    assert "vol_decay_float" in text
    assert out_var == "%output"


def test_quantized_mlir_uses_q12(cubali_quantized):
    lines, _ = cubali_quantized.emit_mlir("%in", True, True)
    text = "\n".join(lines)
    assert "d_scale = 12" in text
    assert "cur_decay_int" in text
    assert "vol_decay_int" in text


def test_q12_decay_range():
    info = CubaLIInfo(name="2", size=8, cur_decay=0.9, vol_decay=0.95)
    info.quantize()
    # 0.9 * 4096 ≈ 3686
    assert 3600 < info.cur_decay_scaled < 3800
    # 0.95 * 4096 ≈ 3891
    assert 3800 < info.vol_decay_scaled < 4000


def test_output_type_quantized(cubali_quantized):
    assert cubali_quantized.output_element_type(True) == "i32"


def test_two_state_args(cubali_float):
    args = cubali_float.state_func_args(False)
    assert len(args) == 2
    names = [a[0] for a in args]
    assert "%current_2" in names
    assert "%voltage_2" in names


def test_intermediate_node_allocates(cubali_float):
    lines, out_var = cubali_float.emit_mlir("%in", False, False)
    assert "memref.alloca" in "\n".join(lines)
    assert out_var == "%voltage_out_2"


# ── the discrete-convention (k = 1) guard ─────────────────────────────────────
#
# dt = tau_mem/r = 0.1 = tau_syn, so with the default w_in = 1 the input gain
# k = w_in*dt/tau_syn is exactly 1 — the discrete convention.


def _cubali_node() -> nir.CubaLI:
    return nir.CubaLI(
        tau_syn=np.full(4, 0.1),
        tau_mem=np.full(4, 0.05),
        r=np.full(4, 0.5),
        v_leak=np.zeros(4),
        input_type={"input": np.array([4])},
    )


def test_parse_cubali_accepts_discrete_convention():
    info = parse_cubali(_cubali_node(), "v0")
    assert info.size == 4


def test_parse_cubali_rejects_k_not_one():
    node = _cubali_node()
    node.w_in = np.full(4, 2.5)  # k = 2.5: continuous-style export
    with pytest.raises(ValueError, match=r"CubaLI 'v0'.*w_in\*dt/tau_syn = 2\.5 != 1"):
        parse_cubali(node, "v0")
