# Copyright 2026 N Vision Systems And Technologies SL
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""``nir.IF`` / ``nir.I`` — the non-leaky pair.

These reuse ``snn.lif`` / ``snn.li``: they are separate PARSERS, not separate
ops, exactly as ``parse_linear``/``parse_affine`` both build ``snn.linear``.
What they cannot be is the leaky parser with the decay forced to 1, and the
first test here is what pins that.
"""

import nir
import numpy as np
import pytest
from snn_mlir.nodes import NODE_PARSERS
from snn_mlir.nodes.li import LIInfo, parse_i
from snn_mlir.nodes.lif import LIFInfo, parse_if, parse_lif


def _if_node(r=1.0, v_threshold=1.0, v_reset=0.0, shape=(4,)):
    return nir.IF(
        r=np.full(shape, r),
        v_threshold=np.full(shape, v_threshold),
        v_reset=np.full(shape, v_reset),
        input_type={"input": np.array(shape)},
    )


def _i_node(r=1.0, shape=(4,)):
    return nir.I(r=np.full(shape, r), input_type={"input": np.array(shape)})


# ── the reason these are separate parsers ─────────────────────────────────────


def test_if_carries_no_tau_so_the_leaky_parser_cannot_serve_it():
    """The concrete reason parse_if exists.

    ``nir.IF`` has no ``tau`` and no ``v_leak`` field, so ``parse_lif`` — which
    computes ``dt = tau/r`` and checks ``v_leak`` — cannot run on one at all.
    "IF is the leaky case with decay 1.0" is a statement about the dynamics,
    never an implementation strategy.
    """
    node = _if_node()
    assert not hasattr(node, "tau")
    assert not hasattr(node, "v_leak")
    with pytest.raises(AttributeError):
        parse_lif(node, "n0")


def test_i_carries_only_r():
    assert {f for f in vars(_i_node())} >= {"r"}
    assert not hasattr(_i_node(), "tau")


# ── decay is 1 by definition ──────────────────────────────────────────────────


def test_parse_if_decay_is_exactly_one():
    info = parse_if(_if_node(), "n0")
    assert isinstance(info, LIFInfo)
    assert info.decay == 1.0
    assert info.threshold == 1.0
    assert info.size == 4


def test_parse_i_decay_is_exactly_one():
    info = parse_i(_i_node(), "n0")
    assert isinstance(info, LIInfo)
    assert info.decay == 1.0
    assert info.size == 4


def test_if_quantized_decay_is_unity_in_q12():
    info = parse_if(_if_node(), "n0")
    info.quantize()
    assert info.decay_scaled == 1 << 12  # exactly 1.0, no leak


# ── the r guard: present on IF/I, ABSENT on LIF/LI ────────────────────────────


@pytest.mark.parametrize("r", [0.5, 2.0, 4.0])
def test_parse_if_rejects_input_gain(r):
    with pytest.raises(ValueError, match="input gain r"):
        parse_if(_if_node(r=r), "n0")


@pytest.mark.parametrize("r", [0.5, 2.0, 4.0])
def test_parse_i_rejects_input_gain(r):
    with pytest.raises(ValueError, match="input gain r"):
        parse_i(_i_node(r=r), "n0")


@pytest.mark.parametrize("r", [2.0, 4.0, 8.0])
def test_parse_lif_still_accepts_any_r(r):
    """``r != 1`` must NOT be guarded on the leaky nodes.

    There ``r = tau/dt`` carries the timestep, so it cancels out of the input
    gain and survives only in ``decay = 1 - 1/r``. Guarding it would reject
    every correctly-exported LIF model in existence.
    """
    node = nir.LIF(
        tau=np.full(4, 2.0),
        r=np.full(4, r),
        v_leak=np.zeros(4),
        v_threshold=np.ones(4),
        v_reset=np.zeros(4),
        input_type={"input": np.array([4])},
    )
    info = parse_lif(node, "n0")
    assert info.decay == pytest.approx(1 - 1 / r)


# ── the remaining guards mirror parse_lif ─────────────────────────────────────


def test_parse_if_rejects_nonzero_v_reset():
    with pytest.raises(ValueError, match="v_reset != 0 not supported"):
        parse_if(_if_node(v_reset=0.3), "n0")


def test_parse_if_rejects_nonuniform_threshold():
    node = _if_node()
    node.v_threshold = np.array([1.0, 1.0, 2.0, 1.0])
    with pytest.raises(ValueError, match="v_threshold must be uniform"):
        parse_if(node, "n0")


def test_parse_if_rejects_nonuniform_r():
    node = _if_node()
    node.r = np.array([1.0, 1.0, 1.5, 1.0])
    with pytest.raises(ValueError, match="r must be uniform"):
        parse_if(node, "n0")


# ── registration ──────────────────────────────────────────────────────────────


def test_both_are_registered():
    assert NODE_PARSERS[nir.IF] is parse_if
    assert NODE_PARSERS[nir.I] is parse_i


def test_if_emits_snn_lif_and_i_emits_snn_li():
    """The dialect stays at four neuron ops — no snn.if, no snn.i."""
    lines, _ = parse_if(_if_node(), "n0").emit_mlir("%in", True, False)
    assert "snn.lif" in "\n".join(lines)
    lines, _ = parse_i(_i_node(), "n0").emit_mlir("%in", True, False)
    assert "snn.li " in "\n".join(lines)


# ── rank: an IF is where a feature map meets a neuron ─────────────────────────


def test_parse_if_keeps_the_full_feature_map_shape():
    """The bug this pins: reading ``input_type["input"][0]`` off a (16,16,16)
    feature map yields 16 neurons where the model has 4096."""
    info = parse_if(_if_node(shape=(16, 16, 16)), "n0")
    assert info.shape == (16, 16, 16)
    assert info.size == 16 * 16 * 16


def test_parse_i_keeps_the_full_feature_map_shape():
    info = parse_i(_i_node(shape=(8, 4, 4)), "n0")
    assert info.shape == (8, 4, 4)
    assert info.size == 128
