# Copyright 2026 N Vision Systems And Technologies SL
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from ._api import export, mlir_from_layers, parse_graph, quantize_layers, to_mlir
from ._check import Finding, NodeReport, Report, check
from ._cname import c_identifier, ensure_unique_c_names
from ._codegen import codegen_folder
from ._graph import GraphInfo
from ._run import run_folder, toolchain_available
from .nodes import NODE_PARSERS, NodeInfo

try:
    __version__ = _version("snn-mlir")
except PackageNotFoundError:  # not installed (e.g. running from a bare checkout)
    __version__ = "0.0.0+unknown"

__all__ = [
    "NODE_PARSERS",
    "Finding",
    "GraphInfo",
    "NodeInfo",
    "NodeReport",
    "Report",
    "__version__",
    "c_identifier",
    "check",
    "codegen_folder",
    "ensure_unique_c_names",
    "export",
    "mlir_from_layers",
    "parse_graph",
    "quantize_layers",
    "run_folder",
    "to_mlir",
    "toolchain_available",
]
