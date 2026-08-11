# Copyright 2026 N Vision Systems And Technologies SL
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Graph-structure tests: cycle breaking, GraphInfo, edge-based rescales.

The model shape throughout is the canonical SNN self-recurrence (the brailernn
topology): fc1 → neuron ⇄ w_rec, then fc2 → a second neuron.
"""

import nir
import numpy as np
import pytest
from snn_mlir import GraphInfo, parse_graph, quantize_layers
from snn_mlir._graph import insert_rescale_nodes
from snn_mlir.nodes import RescaleInfo
from snn_mlir.nodes.cubalif import CubaLIFInfo
from snn_mlir.nodes.linear import LinearInfo


def _linear(out_size: int, in_size: int, scale: float = 0.5) -> nir.Linear:
    w = np.random.uniform(-scale, scale, (out_size, in_size)).astype(np.float32)
    w.flat[0] = scale  # pin max|w| so the natural w_scale is deterministic
    return nir.Linear(weight=w)


def _cubalif(size: int) -> nir.CubaLIF:
    return nir.CubaLIF(
        tau_syn=np.full(size, 0.1),
        tau_mem=np.full(size, 0.05),
        r=np.full(size, 0.5),
        v_leak=np.zeros(size),
        v_threshold=np.ones(size),
        v_reset=np.zeros(size),
        input_type={"input": np.array([size])},
    )


def _recurrent_graph(fc1_scale: float = 0.5, rec_scale: float = 0.5) -> nir.NIRGraph:
    """input → fc1 → lif1 ⇄ w_rec, lif1 → fc2 → lif2 → output."""
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([12])}),
            "fc1": _linear(40, 12, fc1_scale),
            "lif1": _cubalif(40),
            "w_rec": _linear(40, 40, rec_scale),
            "fc2": _linear(7, 40),
            "lif2": _cubalif(7),
            "output": nir.Output(output_type={"output": np.array([7])}),
        },
        edges=[
            ("input", "fc1"),
            ("fc1", "lif1"),
            ("lif1", "w_rec"),
            ("w_rec", "lif1"),
            ("lif1", "fc2"),
            ("fc2", "lif2"),
            ("lif2", "output"),
        ],
    )


# ── cycle breaking ────────────────────────────────────────────────────────────


def test_recurrent_graph_parses_with_broken_edge():
    graph = parse_graph(_recurrent_graph())
    assert isinstance(graph, GraphInfo)
    assert graph.recurrent_edges == [("lif1", "w_rec")]
    # The neuron→synapse edge is gone from the forward edges; the
    # synapse→neuron edge (the merge input) stays.
    assert ("lif1", "w_rec") not in graph.edges
    assert ("w_rec", "lif1") in graph.edges


def test_recurrent_synapse_runs_first():
    """The recurrent synapse reads last timestep's spikes, so it starts the
    timestep; everything else keeps dataflow order."""
    graph = parse_graph(_recurrent_graph())
    assert graph.order == ["w_rec", "fc1", "lif1", "fc2", "lif2"]


def test_fan_in_is_visible_on_the_merge_neuron():
    graph = parse_graph(_recurrent_graph())
    assert sorted(graph.predecessors("lif1")) == ["fc1", "w_rec"]
    assert graph.successors("lif1") == ["fc2"]


def test_self_loop_without_synapse_is_rejected():
    """A neuron feeding itself directly has no synapse edge to break."""
    g = nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([12])}),
            "fc1": _linear(40, 12),
            "lif1": _cubalif(40),
            "output": nir.Output(output_type={"output": np.array([40])}),
        },
        edges=[
            ("input", "fc1"),
            ("fc1", "lif1"),
            ("lif1", "lif1"),
            ("lif1", "output"),
        ],
    )
    with pytest.raises(ValueError, match="cycle"):
        parse_graph(g)


def test_fan_in_from_a_non_synapse_is_rejected():
    """Merge form requires every branch to be a synapse output."""
    g = nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([40])}),
            "fc1": _linear(40, 40),
            "lif0": _cubalif(40),
            "lif1": _cubalif(40),
            "w_rec": _linear(40, 40),
            "output": nir.Output(output_type={"output": np.array([40])}),
        },
        edges=[
            ("input", "fc1"),
            ("fc1", "lif0"),
            ("lif0", "lif1"),  # neuron output feeding the merge directly
            ("lif1", "w_rec"),
            ("w_rec", "lif1"),
            ("lif1", "output"),
        ],
    )
    with pytest.raises(ValueError, match="fan-in"):
        parse_graph(g)


# ── GraphInfo list-compatibility (the linear-chain regression) ────────────────


def test_graphinfo_behaves_like_the_old_layer_list(nir_linear_cubalif):
    graph = parse_graph(nir_linear_cubalif)
    assert len(graph) == 2
    assert isinstance(graph[0], LinearInfo)
    assert isinstance(graph[1], CubaLIFInfo)
    assert [type(x) for x in reversed(graph)] == [CubaLIFInfo, LinearInfo]
    assert isinstance(graph[0:1], list)  # slicing, used by downstream viz code
    assert graph.recurrent_edges == []
    assert graph.order == ["linear", "cubalif"]


# ── edge-based rescale insertion + the shared fan-in w_scale rule ─────────────


def test_fan_in_neuron_gets_one_rescale_per_edge():
    graph = parse_graph(_recurrent_graph())
    quantize_layers(graph)
    spliced = insert_rescale_nodes(graph)
    rescales = [n for n in spliced if isinstance(n, RescaleInfo)]
    assert len(rescales) == 3  # fc1→lif1, w_rec→lif1, fc2→lif2
    assert sorted(spliced.predecessors("lif1")) == ["fc1/rescale", "w_rec/rescale"]


def test_synapses_feeding_the_same_neuron_share_one_w_scale():
    """Different natural scales must collapse to the minimum, and the weights
    must be re-rounded at the shared scale — otherwise the merge branches
    arrive in different Q-formats and no backend could fuse them exactly."""
    graph = parse_graph(_recurrent_graph(fc1_scale=0.5, rec_scale=0.12))
    quantize_layers(graph)
    fc1, w_rec = graph.nodes["fc1"], graph.nodes["w_rec"]
    assert fc1.w_scale != w_rec.w_scale  # natural scales differ

    spliced = insert_rescale_nodes(graph)
    shared = min(7, w_rec.w_scale)  # fc1's natural scale is 7 (max|w| = 0.5)
    assert fc1.w_scale == w_rec.w_scale == shared
    # Weights were re-rounded at the shared scale, not just relabeled.
    assert int(fc1.int8_weights.flat[0]) == round(0.5 * 2**shared)
    # Both merge rescales now shift by the same amount.
    r1 = spliced.nodes["fc1/rescale"]
    r2 = spliced.nodes["w_rec/rescale"]
    assert (r1._w_scale, r1._d_scale) == (r2._w_scale, r2._d_scale)


def test_linear_chain_rescales_are_unchanged_by_the_edge_form(nir_linear_cubalif):
    graph = parse_graph(nir_linear_cubalif)
    quantize_layers(graph)
    spliced = insert_rescale_nodes(graph)
    assert [type(x) for x in spliced] == [LinearInfo, RescaleInfo, CubaLIFInfo]
    assert spliced[1]._w_scale == graph.nodes["linear"].w_scale
