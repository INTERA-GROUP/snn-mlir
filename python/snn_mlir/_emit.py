# SPDX-License-Identifier: Apache-2.0
from .nodes import NodeInfo


def generate_mlir(layers: list[NodeInfo], quantize: bool) -> str:
    """Produce a complete MLIR module string for one forward timestep.

    The returned string is ready to pipe into snn-opt.
    """
    scalar_t = "i8" if quantize else "f32"

    # ── infer network I/O sizes ───────────────────────────────────────────────
    first_synapse = next((layer for layer in layers if layer.is_synapse), None)
    if first_synapse is None:
        raise ValueError("No synapse (weight) layer found in graph.")
    input_size = first_synapse.weight_shape[1]

    last_neuron = next((layer for layer in reversed(layers) if layer.is_neuron), None)
    if last_neuron is None:
        raise ValueError("No neuron layer found in graph.")

    # ── function arguments ────────────────────────────────────────────────────
    args: list[tuple[str, str]] = [
        ("%input", f"memref<{input_size}x{scalar_t}>"),
    ]
    for layer in layers:
        args.extend(layer.weight_func_args(quantize))
    for layer in layers:
        args.extend(layer.state_func_args(quantize))

    out_elem = last_neuron.output_element_type(quantize)
    args.append(("%output", f"memref<{last_neuron.state_size}x{out_elem}>"))

    # ── function body ─────────────────────────────────────────────────────────
    body: list[str] = []
    current_var = "%input"
    for i, layer in enumerate(layers):
        is_last = i == len(layers) - 1
        lines, current_var = layer.emit_mlir(current_var, is_last, quantize)
        body.extend(lines)

    # ── assemble ──────────────────────────────────────────────────────────────
    args_str = ",\n    ".join(f"{name} : {typ}" for name, typ in args)
    body_str = "\n".join(body)

    return (
        "module {\n"
        "  func.func @snn_forward_step(\n"
        f"    {args_str}\n"
        "  ) attributes { llvm.emit_c_interface } {\n"
        f"{body_str}\n"
        "    return\n"
        "  }\n"
        "}\n"
    )
