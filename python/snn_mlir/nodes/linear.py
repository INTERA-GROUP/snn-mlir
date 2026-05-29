# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

import nir
import numpy as np

from ._base import NodeInfo

__all__ = ["LinearInfo", "parse_affine", "parse_linear"]


@dataclass
class LinearInfo(NodeInfo):
    name: str
    input_size: int
    output_size: int
    weights: np.ndarray
    bias: np.ndarray | None = None
    _w_scale: int | None = field(default=None, init=False)
    _quantized_weights: np.ndarray | None = field(default=None, init=False)
    _quantized_bias: np.ndarray | None = field(default=None, init=False)

    # ── classification traits ─────────────────────────────────────────────────

    @property
    def is_synapse(self) -> bool:
        return True

    # ── weight traits ─────────────────────────────────────────────────────────

    @property
    def weight_shape(self) -> tuple[int, int]:
        return (self.output_size, self.input_size)

    @property
    def float_weights(self) -> np.ndarray:
        return self.weights

    @property
    def float_bias(self) -> np.ndarray | None:
        return self.bias

    @property
    def int8_weights(self) -> np.ndarray | None:
        return self._quantized_weights

    @property
    def int32_bias(self) -> np.ndarray | None:
        return self._quantized_bias

    @property
    def w_scale(self) -> int | None:
        return self._w_scale

    # ── quantization ──────────────────────────────────────────────────────────

    def quantize(self) -> None:
        """Compute int8 quantization in place: w_scale, quantized weights/bias."""
        w = self.weights
        min_w, max_w = float(np.min(w)), float(np.max(w))
        qmin, qmax = -128, 127
        ratio = min(
            abs(qmax / max_w) if max_w != 0 else float("inf"),
            abs(qmin / min_w) if min_w != 0 else float("inf"),
        )
        self._w_scale = int(np.floor(np.log2(ratio)))
        self._quantized_weights = np.round(w * (2**self._w_scale)).astype(np.int8)
        if self.bias is not None:
            self._quantized_bias = np.round(
                self.bias * (2**self._w_scale),
            ).astype(np.int32)

    # ── MLIR function arg contributions ──────────────────────────────────────

    def weight_func_args(self, quantize: bool) -> list[tuple[str, str]]:
        w_t = "i8" if quantize else "f32"
        b_t = "i32" if quantize else "f32"
        args = [(f"%w_{self.name}", f"memref<{self.output_size}x{self.input_size}x{w_t}>")]
        if self.bias is not None:
            args.append((f"%b_{self.name}", f"memref<{self.output_size}x{b_t}>"))
        return args

    # ── MLIR body emission ────────────────────────────────────────────────────

    def emit_mlir(
        self,
        input_var: str,
        is_last: bool,
        quantize: bool,
    ) -> tuple[list[str], str]:
        out_var = f"%synapse_{self.name}"
        bias_var = f"%b_{self.name}" if self.bias is not None else None
        if quantize:
            lines = _emit_linear_int(self, input_var, out_var, f"%w_{self.name}", bias_var)
        else:
            lines = _emit_linear_float(self, input_var, out_var, f"%w_{self.name}", bias_var)
        return lines, out_var


def parse_linear(node: nir.Linear, name: str) -> LinearInfo:
    return LinearInfo(
        name=name,
        input_size=int(node.weight.shape[1]),
        output_size=int(node.weight.shape[0]),
        weights=np.array(node.weight, dtype=np.float32),
    )


def parse_affine(node: nir.Affine, name: str) -> LinearInfo:
    return LinearInfo(
        name=name,
        input_size=int(node.weight.shape[1]),
        output_size=int(node.weight.shape[0]),
        weights=np.array(node.weight, dtype=np.float32),
        bias=np.array(node.bias, dtype=np.float32),
    )


# ── MLIR emission helpers ──────────────────────────────────────────────────────


def _emit_linear_float(
    info: LinearInfo,
    input_var: str,
    output_var: str,
    weight_var: str,
    bias_var: str | None,
) -> list[str]:
    o, i = info.output_size, info.input_size
    bias_part = f" bias({bias_var} : memref<{o}xf32>)" if bias_var else ""
    return [
        "",
        f"    // --- Linear {info.name}: ({i}) -> ({o}) ---",
        f"    {output_var} = memref.alloca() : memref<{o}xf32>",
        f"    snn.linear ins({input_var}, {weight_var}){bias_part} out({output_var})"
        f" : memref<{i}xf32>, memref<{o}x{i}xf32> -> memref<{o}xf32>",
    ]


def _emit_linear_int(
    info: LinearInfo,
    input_var: str,
    output_var: str,
    weight_var: str,
    bias_var: str | None,
) -> list[str]:
    o, i = info.output_size, info.input_size
    bias_part = f" bias({bias_var} : memref<{o}xi32>)" if bias_var else ""
    return [
        "",
        f"    // --- Linear {info.name}: ({i}) -> ({o}), int8 weights ---",
        f"    {output_var} = memref.alloca() : memref<{o}xi32>",
        f"    snn.linear ins({input_var}, {weight_var}){bias_part} out({output_var})"
        f" {{w_scale = {info._w_scale} : i64}}"
        f" : memref<{i}xi8>, memref<{o}x{i}xi8> -> memref<{o}xi32>",
    ]
