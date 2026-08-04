# Copyright 2026 N Vision Systems And Technologies SL
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Tests for the non-throwing suitability check.

The contract under test is that ``check`` is *total* (never raises for a graph,
always terminates) and *complete* (reports every node, not just the first bad
one) — and that its verdicts agree with what ``parse_graph`` actually does.
That last property is what keeps the two from drifting apart, so several tests
assert the two against each other rather than against a hardcoded expectation.
"""

import nir
import numpy as np
import pytest
from snn_mlir import check, parse_graph


def _linear(out_size: int = 16, in_size: int = 8) -> nir.Linear:
    return nir.Linear(weight=np.random.uniform(-0.5, 0.5, (out_size, in_size)).astype(np.float32))


def _cubalif(size: int = 16, v_reset: float = 0.0) -> nir.CubaLIF:
    return nir.CubaLIF(
        tau_syn=np.full(size, 0.1),
        tau_mem=np.full(size, 0.05),
        r=np.full(size, 0.5),
        v_leak=np.zeros(size),
        v_threshold=np.ones(size),
        v_reset=np.full(size, v_reset),
        input_type={"input": np.array([size])},
    )


def _graph(nodes: dict, edges: list) -> nir.NIRGraph:
    return nir.NIRGraph(nodes=nodes, edges=edges)


# ── the healthy case ──────────────────────────────────────────────────────────


def test_linear_chain_is_supported(nir_linear_cubalif):
    report = check(nir_linear_cubalif)
    assert report.ok
    assert report.errors == []
    assert [n.name for n in report.nodes] == ["input", "linear", "cubalif", "output"]


def test_terminals_are_not_unsupported(nir_linear_cubalif):
    """Input/Output have no parser but are not a defect — they are the walk's ends."""
    roles = {n.name: n.role for n in check(nir_linear_cubalif).nodes}
    assert roles["input"] == "terminal"
    assert roles["output"] == "terminal"
    assert all(n.ok for n in check(nir_linear_cubalif).nodes)


def test_roles_come_from_node_traits(nir_linear_cubalif):
    roles = {n.name: n.role for n in check(nir_linear_cubalif).nodes}
    assert roles["linear"] == "synapse"
    assert roles["cubalif"] == "neuron"


def test_accepts_a_path(tmp_path, nir_linear_cubalif):
    f = tmp_path / "model.nir"
    nir.write(str(f), nir_linear_cubalif)
    assert check(f).ok


# ── node-level rejections ─────────────────────────────────────────────────────


def test_unsupported_node_type_is_reported_per_node():
    g = _graph(
        {
            # Conv1d infers a [channels, length] input; NIRGraph type-checks the
            # edge on construction, so the terminal has to match its rank.
            "input": nir.Input(input_type={"input": np.array([1, 8])}),
            "conv": nir.Conv1d(
                input_shape=8,
                weight=np.zeros((4, 1, 3), dtype=np.float32),
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=np.zeros(4, dtype=np.float32),
            ),
            "output": nir.Output(output_type={"output": np.array([4, 6])}),
        },
        [("input", "conv"), ("conv", "output")],
    )
    report = check(g)
    assert not report.ok
    (finding,) = [f for f in report.findings if f.kind == "unsupported_type"]
    assert finding.node == "conv"
    assert "Conv1d" in finding.message


def test_v_reset_rejection_carries_the_parsers_own_message():
    g = _graph(
        {
            "input": nir.Input(input_type={"input": np.array([8])}),
            "lin": _linear(),
            "neuron": _cubalif(v_reset=1.0),
            "output": nir.Output(output_type={"output": np.array([16])}),
        },
        [("input", "lin"), ("lin", "neuron"), ("neuron", "output")],
    )
    report = check(g)
    assert not report.ok

    (finding,) = [f for f in report.findings if f.kind == "unsupported_parameter"]
    assert finding.node == "neuron"
    # Verbatim from the parser, not a copy maintained here.
    with pytest.raises(ValueError) as exc:
        parse_graph(g)
    assert finding.message == str(exc.value)


def test_every_bad_node_is_reported_not_just_the_first():
    """parse_graph stops at the first rejection; check must not."""
    g = _graph(
        {
            "input": nir.Input(input_type={"input": np.array([8])}),
            "lin": _linear(),
            "n1": _cubalif(v_reset=1.0),
            "lin2": _linear(16, 16),
            "n2": _cubalif(v_reset=2.0),
            "output": nir.Output(output_type={"output": np.array([16])}),
        },
        [("input", "lin"), ("lin", "n1"), ("n1", "lin2"), ("lin2", "n2"), ("n2", "output")],
    )
    report = check(g)
    assert not report.ok
    assert {f.node for f in report.errors} == {"n1", "n2"}


def test_non_uniform_parameters_are_rejected_per_node():
    neuron = _cubalif()
    neuron.v_threshold = np.linspace(0.5, 1.5, 16)  # per-neuron thresholds
    g = _graph(
        {
            "input": nir.Input(input_type={"input": np.array([8])}),
            "lin": _linear(),
            "neuron": neuron,
            "output": nir.Output(output_type={"output": np.array([16])}),
        },
        [("input", "lin"), ("lin", "neuron"), ("neuron", "output")],
    )
    report = check(g)
    assert not report.ok
    assert any("uniform" in f.message for f in report.errors)


# ── topology ──────────────────────────────────────────────────────────────────


def test_recurrent_graph_is_rejected_though_every_node_is_fine():
    """The case node-level checking alone cannot catch.

    A recurrent edge gives a node two successors. Each node still parses, so a
    per-node check reports all-green for a model that cannot be converted.
    """
    g = _graph(
        {
            "input": nir.Input(input_type={"input": np.array([8])}),
            "lin": _linear(),
            "neuron": _cubalif(),
            "rec": _linear(16, 16),
            "output": nir.Output(output_type={"output": np.array([16])}),
        },
        [
            ("input", "lin"),
            ("lin", "neuron"),
            ("neuron", "rec"),
            ("rec", "neuron"),
            ("neuron", "output"),
        ],
    )
    report = check(g)

    assert all(n.ok for n in report.nodes)  # every node individually fine
    assert not report.ok  # the graph is not

    (finding,) = [f for f in report.graph if f.kind == "nonlinear_topology"]
    assert finding.node == "neuron"
    with pytest.raises(ValueError) as exc:
        parse_graph(g)
    assert finding.message == str(exc.value)


def test_cycle_terminates():
    """parse_graph would spin forever here; check must return."""
    g = _graph(
        {
            "input": nir.Input(input_type={"input": np.array([8])}),
            "a": _linear(8, 8),
            "b": _linear(8, 8),
            "output": nir.Output(output_type={"output": np.array([8])}),
        },
        [("input", "a"), ("a", "b"), ("b", "a")],
    )
    report = check(g)
    assert not report.ok
    assert [f.kind for f in report.graph] == ["cycle"]


def test_dead_end_is_reported():
    g = _graph(
        {
            "input": nir.Input(input_type={"input": np.array([8])}),
            "lin": _linear(),
            "output": nir.Output(output_type={"output": np.array([16])}),
        },
        [("input", "lin")],
    )
    report = check(g)
    assert not report.ok
    assert [f.kind for f in report.graph] == ["dead_end"]


def test_missing_terminal_is_reported():
    g = _graph({"lin": _linear()}, [])
    report = check(g)
    assert not report.ok
    assert {f.kind for f in report.graph} == {"missing_terminal"}


def test_unreachable_node_is_a_warning_not_an_error():
    """It converts — but not all of it does, which is worth saying.

    NIRGraph synthesizes an Input/Output pair around the disconnected node, so
    this also pins that only the node the user actually wrote is reported.
    """
    g = _graph(
        {
            "input": nir.Input(input_type={"input": np.array([8])}),
            "lin": _linear(),
            "neuron": _cubalif(),
            "orphan": _linear(4, 4),
            "output": nir.Output(output_type={"output": np.array([16])}),
        },
        [("input", "lin"), ("lin", "neuron"), ("neuron", "output")],
    )
    report = check(g)
    assert report.ok  # still convertible
    (finding,) = report.warnings
    assert finding.kind == "unreachable"
    assert finding.node == "orphan"
    parse_graph(g)  # and it really does convert


# ── contract ──────────────────────────────────────────────────────────────────


def test_order_is_the_data_flow_path(nir_linear_cubalif):
    assert check(nir_linear_cubalif).order == ("input", "linear", "cubalif", "output")


def test_order_is_empty_when_the_walk_cannot_complete():
    """No path means no order — callers must not read a partial walk as one."""
    g = _graph(
        {
            "input": nir.Input(input_type={"input": np.array([8])}),
            "lin": _linear(),
            "output": nir.Output(output_type={"output": np.array([16])}),
        },
        [("input", "lin")],
    )
    assert check(g).order == ()


def test_report_is_json_serializable(nir_linear_cubalif):
    import json

    payload = json.dumps(check(nir_linear_cubalif).as_dict())
    assert json.loads(payload)["ok"] is True


def test_check_agrees_with_parse_graph(nir_linear_cubalif):
    """The property that matters: ok <=> parse_graph succeeds."""
    assert check(nir_linear_cubalif).ok
    parse_graph(nir_linear_cubalif)  # does not raise
