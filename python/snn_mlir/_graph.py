# Copyright 2026 Sensing & Control Systems, S.L.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import nir

from .nodes import NODE_PARSERS, NodeInfo, RescaleInfo


def parse_graph(graph: nir.NIRGraph) -> list[NodeInfo]:
    """Walk NIR edges from 'input' to 'output'; return ordered layer list."""
    adj: dict[str, list[str]] = {}
    for src, dst in graph.edges:
        adj.setdefault(src, []).append(dst)

    ordered: list[str] = []
    current = "input"
    while current != "output":
        nexts = adj.get(current, [])
        if len(nexts) != 1:
            raise ValueError(
                f"Node '{current}' has {len(nexts)} successors — "
                "only linear-chain graphs are supported.",
            )
        current = nexts[0]
        if current != "output":
            ordered.append(current)

    layers: list[NodeInfo] = []
    for name in ordered:
        node = graph.nodes[name]
        parser = NODE_PARSERS.get(type(node))
        if parser is None:
            raise NotImplementedError(
                f"Node '{name}' has unsupported type: {type(node).__name__}",
            )
        layers.append(parser(node, name))
    return layers


def quantize_layers(layers: list[NodeInfo]) -> None:
    """Compute each layer's quantization parameters in-place."""
    for layer in layers:
        layer.quantize()


def insert_rescale_nodes(layers: list[NodeInfo]) -> list[NodeInfo]:
    """Insert synthetic RescaleInfo between each synapse→neuron edge.

    Uses is_synapse / is_neuron traits so any future synapse type (Conv2d) and
    any neuron type (LIF, LI, CubaLI, CubaLIF) are handled automatically.
    """
    result: list[NodeInfo] = []
    for i, layer in enumerate(layers):
        result.append(layer)
        if layer.is_synapse:
            next_neuron = next(
                (nb for nb in layers[i + 1 :] if nb.is_neuron),
                None,
            )
            if next_neuron is not None:
                result.append(
                    RescaleInfo(
                        name=layer.name,
                        size=layer.weight_shape[0],
                        _w_scale=layer.w_scale,
                        _d_scale=next_neuron.d_scale,
                    )
                )
    return result
