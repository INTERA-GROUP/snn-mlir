#!/usr/bin/env bash
# Copyright 2026 N Vision Systems And Technologies SL
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"

# MLIR_DIR must point to the directory containing MLIRConfig.cmake.
# Example: $HOME/mlir-install/lib/cmake/mlir
if [ -z "${MLIR_DIR:-}" ]; then
    echo "ERROR: MLIR_DIR is not set."
    echo ""
    echo "Set it to the directory containing MLIRConfig.cmake, e.g.:"
    echo "  MLIR_DIR=\$HOME/mlir-install/lib/cmake/mlir $0"
    echo ""
    echo "If you need to build LLVM/MLIR from scratch, see the README."
    exit 1
fi

if [ ! -f "$MLIR_DIR/MLIRConfig.cmake" ]; then
    echo "ERROR: MLIRConfig.cmake not found in MLIR_DIR=$MLIR_DIR"
    echo "Check that MLIR was built or installed correctly."
    exit 1
fi

# Derive the LLVM bin dir from MLIR_DIR (lib/cmake/mlir -> root -> bin).
LLVM_ROOT="$(cd "$MLIR_DIR/../../.." && pwd)"
LIT_PATH="${LLVM_EXTERNAL_LIT:-$LLVM_ROOT/bin/llvm-lit}"

echo "Building snn-opt in $BUILD_DIR ..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake -G Ninja "$REPO_ROOT" \
    -DMLIR_DIR="$MLIR_DIR" \
    -DLLVM_EXTERNAL_LIT="$LIT_PATH"

NPROC=$(nproc 2>/dev/null || echo 4)
ninja -j"$NPROC" snn-opt

echo ""
echo "Build successful. Tool available at:"
echo "  $BUILD_DIR/bin/snn-opt"
echo ""
echo "To run all tests:"
echo "  ninja -C $BUILD_DIR check-snn"
