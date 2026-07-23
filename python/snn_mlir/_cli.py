# Copyright 2026 N Vision Systems And Technologies SL
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Command-line interface for snn-mlir.

Three verbs take a NIR model from graph to running CPU reference without
writing any Python:

    snn-mlir export <model.nir> [-o out.mlir] [-q]   NIR -> SNN-dialect MLIR
    snn-mlir codegen <folder> [-q]                   folder -> build/ C sources
    snn-mlir run <folder> [-q]                       compile + execute -> results.csv

``export`` and ``codegen`` are wired; ``run`` announces itself and exits
non-zero until its milestone lands.
"""

import argparse
import sys
from pathlib import Path

from . import __version__, export


def _cmd_export(args: argparse.Namespace) -> int:
    src = Path(args.model)
    if not src.is_file():
        raise FileNotFoundError(f"NIR file not found: {src}")
    out = Path(args.output) if args.output else src.with_suffix(".mlir")
    export(src, out, quantize=args.quantize)
    print(f"wrote {out}")
    return 0


def _cmd_codegen(args: argparse.Namespace) -> int:
    from ._codegen import codegen_folder

    build = codegen_folder(args.folder, quantize=args.quantize)
    print(f"wrote {build}/ (network.mlir, snn_data.h, main.c, input.h)")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    raise NotImplementedError("`snn-mlir run` is not available yet")


def _add_quantize(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "-q",
        "--quantize",
        action="store_true",
        help="emit int8 weights and Q12 fixed-point neuron state (default: float32)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="snn-mlir",
        description="NIR-to-MLIR frontend and CPU reference for the snn-mlir SNN dialect.",
    )
    parser.add_argument("--version", action="version", version=f"snn-mlir {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>", required=True)

    p_export = sub.add_parser(
        "export",
        help="convert a .nir file to SNN-dialect MLIR",
        description="Convert a NIR graph to SNN-dialect MLIR text, written beside the "
        "input as <stem>.mlir unless -o is given.",
    )
    p_export.add_argument("model", help="path to the input .nir file")
    p_export.add_argument(
        "-o",
        "--output",
        help="destination .mlir path (default: <model>.mlir beside the input)",
    )
    _add_quantize(p_export)
    p_export.set_defaults(func=_cmd_export)

    p_codegen = sub.add_parser(
        "codegen",
        help="generate build/ C sources + MLIR from a model folder (no toolchain)",
        description="Generate a build/ folder (network.mlir, snn_data.h, main.c, "
        "input.h) from a folder holding one .nir and an input.csv.",
    )
    p_codegen.add_argument("folder", help="model folder (one .nir + input.csv)")
    _add_quantize(p_codegen)
    p_codegen.set_defaults(func=_cmd_codegen)

    p_run = sub.add_parser(
        "run",
        help="compile and execute a model folder on the CPU -> results.csv",
        description="Compile the CPU reference for a model folder and run it, writing "
        "results.csv. Requires the snn-opt toolchain.",
    )
    p_run.add_argument("folder", help="model folder (one .nir + input.csv)")
    p_run.add_argument(
        "--platform",
        default="linux",
        choices=["linux"],
        help="target platform (default: linux)",
    )
    _add_quantize(p_run)
    p_run.set_defaults(func=_cmd_run)

    return parser


def main(argv: "list[str] | None" = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (NotImplementedError, ValueError, FileNotFoundError) as exc:
        print(f"snn-mlir: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
