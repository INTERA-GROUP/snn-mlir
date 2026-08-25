# Copyright 2026 N Vision Systems And Technologies SL
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Compile and run the CPU reference for a model folder.

``run_folder`` drives the whole path: ``codegen`` the folder, lower
``network.mlir`` to LLVM IR, compile it to an object with ``llc``, link the
generated ``main.c`` against it with the system C compiler, execute, and write
the per-timestep output to ``results.csv``. No comparison is done — engineers
diff ``results.csv`` against whatever reference they like.

The lowering mirrors ``pipelines/lower_cpu_linux.sh``. It needs a from-source
toolchain (``snn-opt`` + the matching LLVM tools + a C compiler);
``resolve_toolchain`` gates on that up front with an actionable error.
"""

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ._codegen import codegen_folder

# MLIR lowering passes, kept in sync with pipelines/lower_cpu_linux.sh.
_MLIR_OPT_PASSES = [
    "-convert-linalg-to-affine-loops",
    "-affine-loop-invariant-code-motion",
    "-affine-scalrep",
    "-cse",
    "-canonicalize",
    "-lower-affine",
    "-convert-scf-to-cf",
    "-convert-func-to-llvm",
    "-convert-cf-to-llvm",
    "-finalize-memref-to-llvm",
    "-convert-arith-to-llvm",
    "-reconcile-unrealized-casts",
    "-canonicalize",
]


@dataclass
class Toolchain:
    snn_opt: Path
    mlir_opt: Path
    mlir_translate: Path
    llc: Path
    cc: Path


def _find(name: str) -> "Path | None":
    found = shutil.which(name)
    return Path(found) if found else None


def _repo_root() -> Path:
    """The snn-mlir checkout root (…/snn-mlir), when running from source."""
    return Path(__file__).resolve().parents[2]


def _cmake_cache_mlir_dir() -> "str | None":
    """MLIR_DIR recorded in build/CMakeCache.txt when snn-opt was configured.

    This is the SAME LLVM/MLIR that built snn-opt, so the resolved mlir-opt / llc
    versions match the tool and the IR it emits — no separate env var needed.
    """
    cache = _repo_root() / "build" / "CMakeCache.txt"
    if not cache.is_file():
        return None
    for line in cache.read_text().splitlines():
        # e.g. "MLIR_DIR:PATH=/x/lib/cmake/mlir" or "MLIR_DIR:UNINITIALIZED=…".
        key, sep, value = line.partition("=")
        if sep and key.split(":", 1)[0] == "MLIR_DIR" and value.strip():
            return value.strip()
    return None


def _llvm_bin() -> "Path | None":
    """LLVM tools directory: MLIR_DIR env, else the build/CMakeCache.txt record."""
    mlir_dir = os.environ.get("MLIR_DIR") or _cmake_cache_mlir_dir()
    if not mlir_dir:
        return None
    bin_dir = (Path(mlir_dir) / ".." / ".." / ".." / "bin").resolve()
    return bin_dir if bin_dir.is_dir() else None


def _repo_snn_opt() -> "Path | None":
    """snn-opt from this repo's own build/bin (scripts/build_snn_dialect.sh output)."""
    candidate = _repo_root() / "build" / "bin" / "snn-opt"
    return candidate if candidate.exists() else None


def _resolve_optional() -> "dict[str, Path | None]":
    """Resolve each tool to a path or None (no raising) — the basis for the gate."""
    llvm_bin = _llvm_bin()

    def llvm_tool(name: str) -> "Path | None":
        if llvm_bin is not None:
            candidate = llvm_bin / name
            if candidate.exists():
                return candidate
        return _find(name)

    # snn-opt: SNN_OPT override (a full path), else the repo's own build/bin
    # (scripts/build_snn_dialect.sh installs it there), else PATH.
    override = os.environ.get("SNN_OPT")
    if override:
        p = Path(override)
        snn_opt = p if p.exists() else None
    else:
        snn_opt = _repo_snn_opt() or _find("snn-opt")

    # C compiler: CC override (name or path), else the usual suspects.
    cc_name = os.environ.get("CC")
    cc = _find(cc_name) if cc_name else (_find("cc") or _find("clang") or _find("gcc"))

    return {
        "snn_opt": snn_opt,
        "mlir_opt": llvm_tool("mlir-opt"),
        "mlir_translate": llvm_tool("mlir-translate"),
        "llc": llvm_tool("llc"),
        "cc": cc,
    }


def toolchain_available() -> bool:
    """True iff every tool ``run`` needs is resolvable (used to skip tests)."""
    return all(v is not None for v in _resolve_optional().values())


def resolve_toolchain() -> Toolchain:
    """Resolve the full toolchain or raise a single actionable error."""
    tools = _resolve_optional()
    problems = []
    if tools["snn_opt"] is None:
        problems.append(
            "  snn-opt: build it (scripts/build_snn_dialect.sh) — it is auto-detected "
            "at build/bin/snn-opt; or add it to PATH, or set SNN_OPT to the binary"
        )
    if any(tools[k] is None for k in ("mlir_opt", "mlir_translate", "llc")):
        problems.append(
            "  mlir-opt / mlir-translate / llc: auto-detected from build/CMakeCache.txt "
            "when snn-opt was built in-repo; otherwise set MLIR_DIR to your LLVM build's "
            "lib/cmake/mlir, or add the LLVM tools to PATH"
        )
    if tools["cc"] is None:
        problems.append("  C compiler: install clang or gcc, or set CC")
    if problems:
        raise FileNotFoundError("required toolchain is incomplete:\n" + "\n".join(problems))

    def _got(key: str) -> Path:
        path = tools[key]
        assert path is not None  # guaranteed non-None by the checks above
        return path

    return Toolchain(
        snn_opt=_got("snn_opt"),
        mlir_opt=_got("mlir_opt"),
        mlir_translate=_got("mlir_translate"),
        llc=_got("llc"),
        cc=_got("cc"),
    )


def run_folder(
    folder: "str | Path",
    *,
    quantize: bool = False,
    platform: str = "linux",
) -> Path:
    """Codegen, compile, and execute a model folder; write ``build/results.csv``.

    Returns the path to ``results.csv``.
    """
    if platform != "linux":
        raise ValueError(f"unsupported platform: {platform!r} (only 'linux' for now)")

    tools = resolve_toolchain()
    build = codegen_folder(folder, quantize=quantize)

    mlir = build / "network.mlir"
    ll = build / "network.ll"
    obj = build / "network.o"
    exe = build / "snn_exe"
    results = build / "results.csv"

    _lower(tools, mlir, ll)
    subprocess.run(
        [str(tools.llc), "--relocation-model=pic", "-filetype=obj", str(ll), "-o", str(obj)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [str(tools.cc), str(build / "main.c"), str(obj), "-o", str(exe), "-lm"],
        check=True,
        capture_output=True,
    )
    proc = subprocess.run([str(exe)], check=True, capture_output=True)
    _write_results(proc.stdout, results)
    return results


def _lower(tools: Toolchain, mlir: Path, ll: Path) -> None:
    """snn-opt --convert-snn-to-linalg | mlir-opt <passes> | mlir-translate --mlir-to-llvmir."""
    p1 = subprocess.run(
        [str(tools.snn_opt), str(mlir), "--convert-snn-to-linalg"],
        check=True,
        capture_output=True,
    )
    p2 = subprocess.run(
        [str(tools.mlir_opt), *_MLIR_OPT_PASSES],
        input=p1.stdout,
        check=True,
        capture_output=True,
    )
    p3 = subprocess.run(
        [str(tools.mlir_translate), "--mlir-to-llvmir"],
        input=p2.stdout,
        check=True,
        capture_output=True,
    )
    ll.write_bytes(p3.stdout)


def _write_results(stdout: bytes, results: Path) -> None:
    """Extract the CSV_START..CSV_END block the driver prints into results.csv."""
    lines = stdout.decode(errors="replace").splitlines()
    try:
        start = lines.index("CSV_START")
        end = lines.index("CSV_END")
    except ValueError as exc:
        raise ValueError("executable did not emit CSV_START/CSV_END markers") from exc
    rows = lines[start + 1 : end]
    results.write_text("\n".join(rows) + "\n")
