# Copyright 2026 N Vision Systems And Technologies SL
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from dataclasses import dataclass, field

import nir
import numpy as np

from ._base import SynapseInfo, memref_type, nir_shape

__all__ = ["ConvInfo", "parse_conv2d"]

# See nodes/linear.py::_D_SCALE — the neuron Q-format scale that clamps w_scale.
_D_SCALE = 12


@dataclass
class ConvInfo(SynapseInfo):
    """A 2-D convolution synapse: ``snn.conv2d``.

    Activations are rank-3 feature maps ``(C, H, W)``; weights are rank-4
    ``(O, C, Kh, Kw)``. Unlike a dense layer, a conv's output spatial size
    depends on the *input* spatial size, so ``in_spatial`` is carried through
    from the predecessor (``adopt_in_shape``), not fixed by the weights alone.
    """

    in_channels: int
    out_channels: int
    kernel: tuple[int, int]  # (kh, kw)
    stride: tuple[int, int]  # (sh, sw)
    padding: tuple[int, int]  # (ph, pw)
    weights: np.ndarray  # (O, C, Kh, Kw)
    in_spatial: tuple[int, int]  # (H, W), refined by adopt_in_shape
    bias: np.ndarray | None = None
    _w_scale: int | None = field(default=None, init=False)
    _quantized_weights: np.ndarray | None = field(default=None, init=False)
    _quantized_bias: np.ndarray | None = field(default=None, init=False)

    # ── shape traits ──────────────────────────────────────────────────────────

    @property
    def in_shape(self) -> tuple[int, ...]:
        return (self.in_channels, *self.in_spatial)

    @property
    def out_shape(self) -> tuple[int, ...]:
        h, w = self.in_spatial
        kh, kw = self.kernel
        sh, sw = self.stride
        ph, pw = self.padding
        h_out = (h + 2 * ph - kh) // sh + 1
        w_out = (w + 2 * pw - kw) // sw + 1
        return (self.out_channels, h_out, w_out)

    def adopt_in_shape(self, shape: tuple[int, ...]) -> None:
        # A conv's output geometry follows its input's spatial dims, so — unlike
        # a dense synapse — it must record what its predecessor produced. The
        # channel count is fixed by the weights; only H, W flow in.
        self.in_spatial = tuple(shape[1:])

    # ── weight traits ─────────────────────────────────────────────────────────

    @property
    def weight_shape(self) -> tuple[int, ...]:
        return (self.out_channels, self.in_channels, *self.kernel)

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

    # ── MLIR module-level constants ───────────────────────────────────────────

    def weight_globals(self, quantize: bool) -> list[str]:
        if quantize:
            raise NotImplementedError(
                "quantized snn.conv2d is not implemented yet (float lane only)",
            )
        w_ty = memref_type(self.weight_shape, "f32")
        lines = [
            f'  memref.global "private" constant @w_{self.name}'
            f" : {w_ty} = dense<{_dense_float(self.weights)}>",
        ]
        if self.bias is not None:
            b_ty = memref_type((self.out_channels,), "f32")
            lines.append(
                f'  memref.global "private" constant @b_{self.name}'
                f" : {b_ty} = dense<{_dense_float(self.bias)}>",
            )
        return lines

    # ── MLIR body emission ────────────────────────────────────────────────────

    def emit_mlir(
        self,
        input_var: str,
        is_last: bool,
        quantize: bool,
    ) -> tuple[list[str], str]:
        if quantize:
            raise NotImplementedError(
                "quantized snn.conv2d is not implemented yet (float lane only)",
            )
        out_var = f"%synapse_{self.name}"
        in_ty = memref_type(self.in_shape, "f32")
        w_ty = memref_type(self.weight_shape, "f32")
        out_ty = memref_type(self.out_shape, "f32")
        sh, sw = self.stride
        ph, pw = self.padding
        c, h, w = self.in_shape
        o = self.out_channels
        lines = [
            "",
            f"    // --- Conv2d {self.name}: ({c},{h},{w}) -> {self.out_shape} ---",
            f"    %w_{self.name} = memref.get_global @w_{self.name} : {w_ty}",
        ]
        bias_part = ""
        if self.bias is not None:
            b_ty = memref_type((o,), "f32")
            lines.append(
                f"    %b_{self.name} = memref.get_global @b_{self.name} : {b_ty}",
            )
            bias_part = f" bias(%b_{self.name} : {b_ty})"
        lines += [
            f"    {out_var} = memref.alloca() : {out_ty}",
            f"    snn.conv2d ins({input_var}, %w_{self.name}){bias_part}"
            f" out({out_var})"
            f" {{stride = array<i64: {sh}, {sw}>,"
            f" padding = array<i64: {ph}, {pw}>}}"
            f" : {in_ty}, {w_ty} -> {out_ty}",
        ]
        return lines, out_var


def parse_conv2d(node: nir.Conv2d, name: str) -> ConvInfo:
    """``nir.Conv2d`` → :class:`ConvInfo`.

    Only the supported layout is accepted: ``dilation == 1`` and ``groups == 1``
    (a loud guard, like the ``k != 1`` check on CubaLIF). NIR weights are
    ``(O, C, Kh, Kw)``; the input shape ``(C, H, W)`` comes from the node's own
    ``input_type`` and is later cross-checked against what the predecessor
    actually produces (``_propagate_shapes``).
    """
    dilation = np.atleast_1d(node.dilation)
    if not np.all(dilation == 1):
        raise ValueError(
            f"Conv2d '{name}': dilation = {tuple(int(d) for d in dilation)} != 1 is not "
            "supported.",
        )
    groups = int(getattr(node, "groups", 1))
    if groups != 1:
        raise ValueError(f"Conv2d '{name}': groups = {groups} != 1 is not supported.")

    weights = np.array(node.weight, dtype=np.float32)  # (O, C, Kh, Kw)
    out_channels, in_channels, kh, kw = (int(d) for d in weights.shape)
    stride = _pair(node.stride)
    padding = _pair(node.padding)

    in_shape = nir_shape(node.input_type, "input", node=name)  # (C, H, W)
    if len(in_shape) != 3:
        raise ValueError(
            f"Conv2d '{name}': expected a rank-3 (C, H, W) input shape, got {in_shape}.",
        )

    bias = None if node.bias is None else np.array(node.bias, dtype=np.float32)
    return ConvInfo(
        name=name,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel=(kh, kw),
        stride=stride,
        padding=padding,
        weights=weights,
        in_spatial=(in_shape[1], in_shape[2]),
        bias=bias,
    )


def _pair(value: object) -> tuple[int, int]:
    """A scalar or length-2 NIR field as a ``(vertical, horizontal)`` pair."""
    arr = np.atleast_1d(value)
    if arr.size == 1:
        return (int(arr.flat[0]), int(arr.flat[0]))
    return (int(arr.flat[0]), int(arr.flat[1]))


def _dense_float(arr: np.ndarray) -> str:
    """Render a float numpy array as a nested MLIR ``dense`` element literal."""
    if arr.ndim == 1:
        return "[" + ", ".join(f"{float(v):.8e}" for v in arr) + "]"
    return "[" + ", ".join(_dense_float(row) for row in arr) + "]"
