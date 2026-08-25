# Copyright 2026 N Vision Systems And Technologies SL
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""MLIR text emission: one ``func.func @snn_forward_step`` per model.

The emitter walks the graph's topological order carrying a ``var_map`` from
node name to the MLIR value holding that node's output, so dataflow — not list
adjacency — decides what feeds what. Two constructs exist only at this level
(they are graph structure, not NIR nodes, so no ``NodeInfo`` emits them):

* **Fan-in merge** — a neuron fed by several synapses receives their
  elementwise sum (``arith.addf`` in float, ``arith.addi`` in Q12 after the
  per-edge rescales), computed by a ``linalg.generic`` just before the neuron.
* **Recurrent state** — for every broken neuron→synapse edge, the synapse
  reads the neuron's *previous-timestep* spikes from a dedicated function
  argument, and the function ends by ``memref.copy``-ing the neuron's fresh
  spikes into that buffer for the next call.

Function-argument order — the positional C ABI contract with any caller
(``snn_data.h``/``main.c`` generators must mirror it exactly):

1. ``%input``
2. for each layer in topological order: its state buffers
   (``state_func_args``), and — immediately after them, when the layer is a
   recurrent neuron — its previous-spikes buffer ``%prev_spikes_<name>``
   (spike-typed: ``i8`` quantized, ``f32`` float)
3. ``%output``
"""

from ._graph import GraphInfo, as_graph_info
from .nodes import NeuronInfo, NodeInfo
from .nodes._base import memref_type


def generate_mlir(layers: "list[NodeInfo] | GraphInfo", quantize: bool) -> str:
    """Produce a complete MLIR module string for one forward timestep.

    Accepts a :class:`GraphInfo` (possibly with recurrent edges and fan-in) or
    the old plain layer list, which is treated as a linear chain. The returned
    string is ready to pipe into snn-opt.
    """
    graph = as_graph_info(layers)
    scalar_t = "i8" if quantize else "f32"

    # ── network I/O shapes (rank-1 for a dense entry, rank-N for a conv one) ──
    input_shape = graph.input_shape

    last_neuron = next((layer for layer in reversed(graph) if layer.is_neuron), None)
    if last_neuron is None:
        raise ValueError("No neuron layer found in graph.")

    recurrent_neurons = [u for u, _ in graph.recurrent_edges]

    # ── module-level weight constants ─────────────────────────────────────────
    globals_: list[str] = []
    for layer in graph:
        globals_.extend(layer.weight_globals(quantize))

    # ── function arguments (the positional C ABI — see module docstring) ──────
    args: list[tuple[str, str]] = [
        ("%input", memref_type(input_shape, scalar_t)),
    ]
    for name in graph.order:
        layer = graph.nodes[name]
        args.extend(layer.state_func_args(quantize))
        if name in recurrent_neurons:
            spike_t = layer.output_element_type(quantize)
            args.append(
                (f"%prev_spikes_{name}", f"memref<{layer.state_size}x{spike_t}>"),
            )

    out_elem = last_neuron.output_element_type(quantize)
    args.append(("%output", memref_type(last_neuron.out_shape, out_elem)))

    # ── function body ─────────────────────────────────────────────────────────
    body: list[str] = []
    var_map: dict[str, str] = {}
    rec_source_of = {synapse: neuron for neuron, synapse in graph.recurrent_edges}
    for i, name in enumerate(graph.order):
        layer = graph.nodes[name]
        preds = graph.predecessors(name)

        if name in rec_source_of:
            # A recurrent synapse has no forward predecessor: its input is the
            # previous timestep's spikes of the neuron it feeds back into.
            input_var = f"%prev_spikes_{rec_source_of[name]}"
        elif preds == ["input"]:
            input_var = "%input"
        elif len(preds) == 1:
            input_var = var_map[preds[0]]
        else:
            # A fan-in merge is only defined in front of a neuron (it sizes the
            # merged accumulator to the neuron's state).
            if not isinstance(layer, NeuronInfo):
                raise ValueError(f"Node '{name}': fan-in merge requires a neuron consumer.")
            merge_lines, input_var = _emit_merge(
                name,
                [var_map[p] for p in preds],
                layer.state_size,
                quantize,
            )
            body.extend(merge_lines)

        is_last = i == len(graph.order) - 1
        lines, out_var = layer.emit_mlir(input_var, is_last, quantize)
        body.extend(lines)
        var_map[name] = out_var

    for neuron, _synapse in graph.recurrent_edges:
        layer = graph.nodes[neuron]
        spike_t = layer.output_element_type(quantize)
        memref_t = f"memref<{layer.state_size}x{spike_t}>"
        body += [
            "",
            f"    // --- Recurrent state {neuron}: spikes -> next timestep ---",
            f"    memref.copy {var_map[neuron]}, %prev_spikes_{neuron} : {memref_t} to {memref_t}",
        ]

    # ── assemble ──────────────────────────────────────────────────────────────
    args_str = ",\n    ".join(f"{name} : {typ}" for name, typ in args)
    body_str = "\n".join(body)
    globals_str = ("\n".join(globals_) + "\n") if globals_ else ""

    return (
        "module {\n"
        f"{globals_str}"
        "  func.func @snn_forward_step(\n"
        f"    {args_str}\n"
        "  ) attributes { llvm.emit_c_interface } {\n"
        f"{body_str}\n"
        "    return\n"
        "  }\n"
        "}\n"
    )


def _emit_merge(
    neuron_name: str,
    input_vars: list[str],
    size: int,
    quantize: bool,
) -> tuple[list[str], str]:
    """Elementwise sum of the fan-in branches, just before the neuron.

    In quantized mode the branches are the per-edge rescale outputs, already in
    the same Q12 format (their synapses share one ``w_scale``), so an ``addi``
    is exact.
    """
    elem_t = "i32" if quantize else "f32"
    add_op = "arith.addi" if quantize else "arith.addf"
    memref_t = f"memref<{size}x{elem_t}>"
    out_var = f"%merged_{neuron_name}"

    n = len(input_vars)
    identity = "affine_map<(d0) -> (d0)>"
    maps = ", ".join([identity] * (n + 1))
    ins = ", ".join(input_vars)
    ins_types = ", ".join([memref_t] * n)
    block_args = ", ".join([f"%in{k}: {elem_t}" for k in range(n)] + [f"%acc: {elem_t}"])

    lines = [
        "",
        f"    // --- Merge {neuron_name}: {n}-way fan-in add ---",
        f"    {out_var} = memref.alloca() : {memref_t}",
        f"    linalg.generic {{indexing_maps = [{maps}],"
        f' iterator_types = ["parallel"]}}'
        f" ins({ins} : {ins_types}) outs({out_var} : {memref_t}) {{",
        f"    ^bb0({block_args}):",
    ]
    acc = "%in0"
    for k in range(1, n):
        sum_var = f"%sum{k}"
        lines.append(f"      {sum_var} = {add_op} {acc}, %in{k} : {elem_t}")
        acc = sum_var
    lines += [
        f"      linalg.yield {acc} : {elem_t}",
        "    }",
    ]
    return lines, out_var
