# Copyright 2026 N Vision Systems And Technologies SL
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from dataclasses import dataclass, field

import nir
import numpy as np

from ._base import SynapseInfo

__all__ = ["LinearInfo", "parse_affine", "parse_linear"]

# The downstream neuron's Q-format scale (see e.g. nodes/lif.py::_D_SCALE). A
# layer's rescale shift is `d_scale - w_scale`, so w_scale must not exceed
# d_scale or the shift goes negative, which is unrepresentable on hardware whose
# rescale is left-shift-only. More weight precision than the neuron's fixed-point
# state can hold buys nothing anyway. d_scale is 12 for every neuron today; if it
# ever varies per neuron, this clamp must move to where the synapse->neuron edge
# is known (insert_rescale_nodes), since quantize() cannot see its neuron.
_D_SCALE = 12


@dataclass
class LinearInfo(SynapseInfo):
    input_size: int
    output_size: int
    weights: np.ndarray
    bias: np.ndarray | None = None
    _w_scale: int | None = field(default=None, init=False)
    _quantized_weights: np.ndarray | None = field(default=None, init=False)
    _quantized_bias: np.ndarray | None = field(default=None, init=False)

    # ── shape traits ──────────────────────────────────────────────────────────
    #
    # `snn.linear` is strictly rank-1 (see docs/python/nir-mapping.md); the
    # rank-changing synapses are Conv2d, pooling and Flatten.

    @property
    def in_shape(self) -> tuple[int, ...]:
        return (self.input_size,)

    @property
    def out_shape(self) -> tuple[int, ...]:
        return (self.output_size,)

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
        self.requantize(min(int(np.floor(np.log2(ratio))), _D_SCALE))

    def requantize(self, w_scale: int) -> None:
        """Re-quantize weights (and bias) at an externally chosen ``w_scale``.

        Recomputes from the float weights, so calling it repeatedly is safe.
        Only ever called with a scale at or below the natural one ``quantize``
        picked (synapses feeding the same neuron share the minimum of their
        scales), so the int8 range cannot overflow.
        """
        self._w_scale = w_scale
        self._quantized_weights = np.round(self.weights * (2**w_scale)).astype(np.int8)
        if self.bias is not None:
            self._quantized_bias = np.round(
                self.bias * (2**w_scale),
            ).astype(np.int32)

    # ── MLIR module-level constants ───────────────────────────────────────────

    def weight_globals(self, quantize: bool) -> list[str]:
        """Emit module-level ``memref.global`` constants for weights (and bias).

        Weights are baked into the IR as private constant globals rather than
        passed as function arguments, so the compiled module is self-contained
        and a backend can inspect or transform the weight data directly.
        """
        o, i = self.output_size, self.input_size
        if quantize:
            w_t, w_lit = "i8", _dense_int(self.int8_weights)
        else:
            w_t, w_lit = "f32", _dense_float(self.weights)
        lines = [
            f'  memref.global "private" constant @w_{self.name}'
            f" : memref<{o}x{i}x{w_t}> = dense<{w_lit}>",
        ]
        if self.bias is not None:
            if quantize:
                b_t, b_lit = "i32", _dense_int(self.int32_bias)
            else:
                b_t, b_lit = "f32", _dense_float(self.bias)
            lines.append(
                f'  memref.global "private" constant @b_{self.name}'
                f" : memref<{o}x{b_t}> = dense<{b_lit}>",
            )
        return lines

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
    lines = [
        "",
        f"    // --- Linear {info.name}: ({i}) -> ({o}) ---",
        f"    {weight_var} = memref.get_global @w_{info.name} : memref<{o}x{i}xf32>",
    ]
    bias_part = ""
    if bias_var:
        lines.append(f"    {bias_var} = memref.get_global @b_{info.name} : memref<{o}xf32>")
        bias_part = f" bias({bias_var} : memref<{o}xf32>)"
    lines += [
        f"    {output_var} = memref.alloca() : memref<{o}xf32>",
        f"    snn.linear ins({input_var}, {weight_var}){bias_part} out({output_var})"
        f" : memref<{i}xf32>, memref<{o}x{i}xf32> -> memref<{o}xf32>",
    ]
    return lines


def _emit_linear_int(
    info: LinearInfo,
    input_var: str,
    output_var: str,
    weight_var: str,
    bias_var: str | None,
) -> list[str]:
    o, i = info.output_size, info.input_size
    lines = [
        "",
        f"    // --- Linear {info.name}: ({i}) -> ({o}), int8 weights ---",
        f"    {weight_var} = memref.get_global @w_{info.name} : memref<{o}x{i}xi8>",
    ]
    bias_part = ""
    if bias_var:
        lines.append(f"    {bias_var} = memref.get_global @b_{info.name} : memref<{o}xi32>")
        bias_part = f" bias({bias_var} : memref<{o}xi32>)"
    lines += [
        f"    {output_var} = memref.alloca() : memref<{o}xi32>",
        f"    snn.linear ins({input_var}, {weight_var}){bias_part} out({output_var})"
        f" {{w_scale = {info._w_scale} : i64}}"
        f" : memref<{i}xi8>, memref<{o}x{i}xi8> -> memref<{o}xi32>",
    ]
    return lines


# ── dense literal helpers ───────────────────────────────────────────────────────


def _dense_int(arr: np.ndarray | None) -> str:
    """Render an int numpy array as a nested MLIR ``dense`` element literal.

    ``None`` means the quantized weights/bias were never computed — call
    ``quantize()`` before emitting the int8 module.
    """
    if arr is None:
        raise ValueError("quantized data is missing; call quantize() first")
    if arr.ndim == 1:
        return "[" + ", ".join(str(int(v)) for v in arr) + "]"
    return "[" + ", ".join(_dense_int(row) for row in arr) + "]"


def _dense_float(arr: np.ndarray) -> str:
    """Render a float numpy array as a nested MLIR ``dense`` element literal."""
    if arr.ndim == 1:
        return "[" + ", ".join(f"{float(v):.8e}" for v in arr) + "]"
    return "[" + ", ".join(_dense_float(row) for row in arr) + "]"
