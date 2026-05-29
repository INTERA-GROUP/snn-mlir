#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SNN dialect → LLVM IR (.ll) on x86_64 Linux.
# Pipe all MLIR passes inline; no intermediate files.
#
# Prerequisites:
#   - snn-opt built via scripts/build_snn_dialect.sh  (→ build/bin/snn-opt)
#   - MLIR_DIR set to the directory containing MLIRConfig.cmake
#   - e.g.: export MLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir
#
# Usage:
#   bash pipelines/lower_cpu_linux.sh examples/snn_oxford/build/network.mlir
set -euo pipefail

REPO_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )
SNN_OPT="$REPO_ROOT/build/bin/snn-opt"

if [ -z "$MLIR_DIR" ]; then
    echo "Error: MLIR_DIR is not set."
    echo "  export MLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir"
    exit 1
fi
LLVM_BIN="$MLIR_DIR/../../../bin"

INPUT="${1:?Usage: pipelines/lower_cpu_linux.sh <input.mlir>}"
OUT_LL="$(dirname "$INPUT")/$(basename "$INPUT" .mlir).ll"

"$SNN_OPT" "$INPUT" --convert-snn-to-linalg \
    | "$LLVM_BIN/mlir-opt" \
        -convert-linalg-to-affine-loops \
        -affine-loop-invariant-code-motion \
        -affine-scalrep \
        -cse \
        -canonicalize \
        -lower-affine \
        -convert-scf-to-cf \
        -convert-func-to-llvm \
        -convert-cf-to-llvm \
        -finalize-memref-to-llvm \
        -convert-arith-to-llvm \
        -reconcile-unrealized-casts \
        -canonicalize \
    | "$LLVM_BIN/mlir-translate" --mlir-to-llvmir \
    > "$OUT_LL"

echo "LLVM IR written to: $OUT_LL"
