# Copyright 2026 N Vision Systems And Technologies SL
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from ._api import export, mlir_from_layers, parse_graph, quantize_layers, to_mlir
from .nodes import NODE_PARSERS, NodeInfo

try:
    __version__ = _version("snn-mlir")
except PackageNotFoundError:  # not installed (e.g. running from a bare checkout)
    __version__ = "0.0.0+unknown"

__all__ = [
    "NODE_PARSERS",
    "NodeInfo",
    "__version__",
    "export",
    "mlir_from_layers",
    "parse_graph",
    "quantize_layers",
    "to_mlir",
]
