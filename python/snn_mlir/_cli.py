# Copyright 2026 N Vision Systems And Technologies SL
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Command-line interface for snn-mlir.

Four verbs take a NIR model from graph to running CPU reference without
writing any Python:

    snn-mlir check <model.nir> [--json]              is this model supported?
    snn-mlir export <model.nir> [-o out.mlir] [-q]   NIR -> SNN-dialect MLIR
    snn-mlir codegen <folder> [-q]                   folder -> build/ C sources
    snn-mlir run <folder> [-q]                       compile + execute -> results.csv

All four verbs are wired. ``run`` needs a from-source toolchain (snn-opt +
the matching LLVM tools + a C compiler) and gates on it with a clear error.
``check`` is the one that needs nothing but the file — it answers up front what
the other three would otherwise fail on halfway.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import __version__, export


def _cmd_check(args: argparse.Namespace) -> int:
    from ._check import check

    src = Path(args.model)
    if not src.is_file():
        raise FileNotFoundError(f"NIR file not found: {src}")
    report = check(src)

    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
        return 0 if report.ok else 1

    print(f"{src.name} — {len(report.nodes)} nodes")
    print()
    # Widths from the data: node names are user-chosen and range from "0" to
    # "lif1.w_rec", so a fixed column is either ragged or truncating.
    w_name = max((len(n.name) for n in report.nodes), default=4)
    w_type = max((len(n.type) for n in report.nodes), default=4)
    # Data-flow order where the walk found one, so the table reads like the model
    # rather than like the file's node dict. Anything off that path (unreachable
    # nodes, or every node when the walk failed) keeps its original order at the end.
    rank = {name: i for i, name in enumerate(report.order)}
    for n in sorted(report.nodes, key=lambda n: rank.get(n.name, len(rank))):
        mark = "ok  " if n.ok else "FAIL"
        print(f"  {mark}  {n.name:<{w_name}}  {n.type:<{w_type}}  {n.role}")
        for f in n.findings:
            print(f"        {f.message}")

    if report.graph:
        print()
        for f in report.graph:
            print(f"  {f.severity.upper():<7} {f.message}")

    print()
    if report.ok:
        n_warn = len(report.warnings)
        suffix = f" ({n_warn} warning{'s' if n_warn != 1 else ''})" if n_warn else ""
        print(f"supported: this graph can be converted by snn-mlir{suffix}.")
        return 0
    n_err = len(report.errors)
    print(f"not supported: {n_err} error{'s' if n_err != 1 else ''}.")
    return 1


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
    from ._run import run_folder

    results = run_folder(args.folder, quantize=args.quantize, platform=args.platform)
    print(f"wrote {results}")
    return 0


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

    p_check = sub.add_parser(
        "check",
        help="report whether a .nir model is supported, and what blocks it",
        description="Check a NIR graph against the front-end's rules without converting "
        "it. Every node is reported, not just the first problem. Exits 1 if the model "
        "cannot be converted.",
    )
    p_check.add_argument("model", help="path to the input .nir file")
    p_check.add_argument(
        "--json",
        action="store_true",
        help="emit the full report as JSON instead of a table",
    )
    p_check.set_defaults(func=_cmd_check)

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
    except subprocess.CalledProcessError as exc:
        tool = Path(exc.cmd[0] if isinstance(exc.cmd, list) else exc.cmd).name
        print(f"snn-mlir: error: {tool} failed (exit {exc.returncode})", file=sys.stderr)
        if exc.stderr:
            sys.stderr.buffer.write(exc.stderr)
        return 1
    except (NotImplementedError, ValueError, FileNotFoundError) as exc:
        print(f"snn-mlir: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
