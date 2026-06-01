# Copyright 2026 Sensing & Control Systems, S.L.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Export the SNN Oxford NIR model to MLIR + C runtime files.

Network: Linear(200→256) → CubaLIF(256) → Linear(256→200) → CubaLIF(200)
Source:  LAVA-DL, exported as network.nir

Usage:
    python run.py              # float32
    python run.py --quantize   # int8 weights, Q12 neuron state
"""

import argparse
import sys
from pathlib import Path

import snn_mlir

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))  # make examples/_codegen.py importable
import _codegen  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quantize", action="store_true", help="int8 weights + Q12 fixed-point neuron state"
    )
    parser.add_argument("--n-steps", type=int, default=100)
    args = parser.parse_args()

    build = HERE / "build"

    snn_mlir.export(
        HERE / "network.nir",
        build / "network.mlir",
        quantize=args.quantize,
    )
    _codegen.export(
        HERE / "network.nir",
        build,
        quantize=args.quantize,
        n_steps=args.n_steps,
        index_bits=64,
        input_file=HERE / "input.h",
    )

    mode = "quantized" if args.quantize else "float32"
    print(f"Done ({mode}). Files in {build}:")
    for f in sorted(build.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
