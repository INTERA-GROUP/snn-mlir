# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import nir as _nir

from . import _emit, _graph


def to_mlir(
    source: "_nir.NIRGraph | str | Path",
    *,
    quantize: bool = False,
) -> str:
    """Convert a NIR graph to SNN dialect MLIR text.

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
    if isinstance(source, (str, Path)):
        source = _nir.read(str(source))

    layers = _graph.parse_graph(source)
    if quantize:
        _graph.quantize_layers(layers)
    emit_layers = _graph.insert_rescale_nodes(layers) if quantize else layers
    return _emit.generate_mlir(emit_layers, quantize)


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
