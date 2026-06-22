# Copyright 2026 N Vision Systems And Technologies SL
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from pathlib import Path

import nir as _nir

from . import _emit, _graph
from .nodes import NodeInfo


def parse_graph(source: "_nir.NIRGraph | str | Path") -> list[NodeInfo]:
    """Walk a NIR graph and return its ordered list of layers.

    This is the structured entry point: it stops after parsing so callers can
    inspect, quantize, or feed the :class:`~snn_mlir.nodes.NodeInfo` objects to
    their own code generation before (or instead of) emitting MLIR text.

    Args:
        source: A nir.NIRGraph object, or a path to a .nir file.

    Returns:
        An ordered list of NodeInfo layers (no quantization applied).

    """
    if isinstance(source, (str, Path)):
        source = _nir.read(str(source))
    return _graph.parse_graph(source)


def quantize_layers(layers: list[NodeInfo]) -> None:
    """Compute each layer's quantization parameters in-place.

    Mutates the layers, so call it at most once per list (quantization is not
    idempotent — re-running it would re-scale already-quantized weights).
    """
    _graph.quantize_layers(layers)


def mlir_from_layers(layers: list[NodeInfo], *, quantize: bool = False) -> str:
    """Emit SNN dialect MLIR text from a pre-parsed list of layers.

    Args:
        layers:   Layers from :func:`parse_graph` (pass them through
                  :func:`quantize_layers` first if ``quantize=True``).
        quantize: If True, inserts ``snn.rescale`` nodes and emits the quantized
                  (int8 / Q12) module. Must match how ``layers`` were quantized.

    Returns:
        A string containing the complete MLIR module.

    """
    emit_layers = _graph.insert_rescale_nodes(layers) if quantize else layers
    return _emit.generate_mlir(emit_layers, quantize)


def to_mlir(
    source: "_nir.NIRGraph | str | Path",
    *,
    quantize: bool = False,
) -> str:
    """Convert a NIR graph to SNN dialect MLIR text.

    Convenience wrapper composing :func:`parse_graph`, :func:`quantize_layers`,
    and :func:`mlir_from_layers` for the common one-shot case.

    Args:
        source:   A nir.NIRGraph object, or a path to a .nir file.
        quantize: If True, emit int8 weights and Q12 fixed-point neuron state.
                  If False (default), emit float32.

    Returns:
        A string containing the complete MLIR module, ready to pipe into snn-opt.

    Example::

        import snn_mlir
        mlir = snn_mlir.to_mlir("network.nir", quantize=True)
        with open("network.mlir", "w") as f:
            f.write(mlir)

    """
    layers = parse_graph(source)
    if quantize:
        quantize_layers(layers)
    return mlir_from_layers(layers, quantize=quantize)


def export(
    source: "_nir.NIRGraph | str | Path",
    output_path: "str | Path",
    *,
    quantize: bool = False,
) -> None:
    """Convert a NIR graph and write the result to a .mlir file.

    Thin wrapper around :func:`to_mlir` for convenience.

    Args:
        source:      A nir.NIRGraph object, or a path to a .nir file.
        output_path: Destination .mlir file path.
        quantize:    Passed through to :func:`to_mlir`.

    """
    mlir = to_mlir(source, quantize=quantize)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(mlir)
