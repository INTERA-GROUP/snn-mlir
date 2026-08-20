# Copyright 2026 N Vision Systems And Technologies SL
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Shape plumbing: what each layer reads and writes, and who decides it.

Every layer carries ``in_shape``/``out_shape``; ``parse_graph`` flows them along
the chain and cross-checks against NIR's declared types. The propagated shape is
authoritative — NIR's is a cross-check — so a disagreement warns rather than
raises.
"""

import nir
import numpy as np
import pytest
from snn_mlir._graph import insert_rescale_nodes, parse_graph, quantize_layers
from snn_mlir.nodes._base import nir_shape
from snn_mlir.nodes.cubalif import CubaLIFInfo
from snn_mlir.nodes.lif import LIFInfo

# ── nir_shape: the whole entry, not element [0] ───────────────────────────────


def test_nir_shape_keeps_every_dimension():
    assert nir_shape({"input": np.array([16, 16, 16])}, "input", node="n") == (16, 16, 16)


def test_nir_shape_is_plain_ints():
    shape = nir_shape({"input": np.array([8])}, "input", node="n")
    assert shape == (8,)
    assert all(type(d) is int for d in shape)


def test_nir_shape_refuses_a_missing_entry():
    """SumPool2d/AvgPool2d null their own types in __post_init__, so a
    Python-constructed pool node genuinely has no shape. Refusing beats
    defaulting to zero."""
    with pytest.raises(ValueError, match="declares no input shape"):
        nir_shape(None, "input", node="pool")
    with pytest.raises(ValueError, match="declares no input shape"):
        nir_shape({}, "input", node="pool")


# ── per-layer shape traits ────────────────────────────────────────────────────


def test_linear_shapes_come_from_the_weight_matrix(linear_float):
    assert linear_float.in_shape == (8,)
    assert linear_float.out_shape == (16,)


def test_neuron_is_shape_preserving(cubalif_float):
    assert cubalif_float.in_shape == cubalif_float.out_shape == (16,)


def test_size_is_derived_not_stored():
    info = CubaLIFInfo(name="n", shape=(8, 4, 4), cur_decay=0.9, vol_decay=0.95, threshold=1.0)
    assert info.size == 128
    assert info.state_size == 128


def test_adopting_a_shape_moves_the_size_with_it():
    info = LIFInfo(name="n", shape=(16,), decay=1.0, threshold=1.0)
    info.adopt_in_shape((16, 16, 16))
    assert info.shape == (16, 16, 16)
    assert info.size == 4096


def test_a_synapse_does_not_absorb_a_shape(linear_float):
    """`snn.linear` is strictly rank-1: the weight matrix decides both ends, so
    adopt_in_shape is the base class no-op and a mismatch is reported instead."""
    linear_float.adopt_in_shape((2, 4))
    assert linear_float.in_shape == (8,)


# ── propagation along a real graph ────────────────────────────────────────────


def test_shapes_flow_through_the_chain(nir_linear_cubalif):
    graph = parse_graph(nir_linear_cubalif)
    assert graph.nodes["linear"].in_shape == (8,)
    assert graph.nodes["cubalif"].in_shape == (16,)


def test_rescale_inherits_the_synapse_shape(nir_linear_cubalif):
    graph = parse_graph(nir_linear_cubalif)
    quantize_layers(graph)
    emit = insert_rescale_nodes(graph)
    rescale = emit.nodes["linear/rescale"]
    assert rescale.shape == (16,)
    assert rescale.size == 16


def _chain(out_features=16, neurons=16):
    """Linear(8→out_features) → LIF(neurons).

    NIR derives a neuron's ``input_type`` from its own parameter arrays in
    ``__post_init__`` — passing a contradictory one is simply overwritten — so
    the only way to build a genuine disagreement is to size the neuron
    differently from the synapse feeding it, which is exactly the real-world
    failure this warning is for.
    """
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([8])}),
            "linear": nir.Linear(weight=np.zeros((out_features, 8), dtype=np.float32)),
            "lif": nir.LIF(
                tau=np.full(neurons, 2.0),
                r=np.full(neurons, 4.0),
                v_leak=np.zeros(neurons),
                v_threshold=np.ones(neurons),
                v_reset=np.zeros(neurons),
            ),
            "output": nir.Output(output_type={"output": np.array([neurons])}),
        },
        edges=[("input", "linear"), ("linear", "lif"), ("lif", "output")],
    )


def test_a_disagreeing_neuron_warns_and_takes_the_propagated_shape():
    """A stale declared shape loses to the one the arithmetic requires.

    ``NIRGraph.__post_init__`` runs type inference and refuses to build a graph
    whose declared types contradict each other, so this state cannot be
    constructed — it can only be *read from a file*, whose types NIR restores
    verbatim without re-inferring. That is not a hypothetical: NIR's own
    ``Conv2d`` inference uses the kernel height for both spatial dimensions, so
    a non-square kernel writes a wrong-but-internally-consistent output shape
    into the file. Mutating after construction is how the test reaches the state
    a file read produces.
    """
    graph = _chain()
    graph.nodes["lif"].input_type = {"input": np.array([32])}
    with pytest.warns(UserWarning, match="predecessor produces"):
        parsed = parse_graph(graph)
    # The synapse produces 16, so 16 is what the neuron must be.
    assert parsed.nodes["lif"].shape == (16,)


def test_an_agreeing_chain_is_silent(nir_linear_cubalif):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        parse_graph(nir_linear_cubalif)


def test_a_stale_declared_output_type_warns():
    graph = _chain()
    graph.nodes["linear"].output_type = {"output": np.array([99])}
    with pytest.warns(UserWarning, match="NIR declares output_type"):
        parse_graph(graph)
