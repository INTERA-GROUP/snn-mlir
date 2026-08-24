# Copyright 2026 N Vision Systems And Technologies SL
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from dataclasses import dataclass, field

import nir
import numpy as np

from ._base import SynapseInfo, memref_type, nir_shape
from .conv import _dense_float

__all__ = ["Conv1dInfo", "parse_conv1d"]

# See nodes/linear.py::_D_SCALE — the neuron Q-format scale that clamps w_scale.
_D_SCALE = 12


@dataclass
class Conv1dInfo(SynapseInfo):
    """A 1-D convolution synapse: ``snn.conv1d``.

    Activations are rank-2 feature maps ``(C, L)``; weights are rank-3
    ``(O, C, K)``. Like :class:`~snn_mlir.nodes.conv.ConvInfo`, a conv's output
    spatial size depends on the *input* spatial size, so ``in_len`` is carried
    through from the predecessor (``adopt_in_shape``), not fixed by the weights.
    """

    in_channels: int
    out_channels: int
    kernel: int
    stride: int
    padding: int
    weights: np.ndarray  # (O, C, K)
    in_len: int  # L, refined by adopt_in_shape
    bias: np.ndarray | None = None
    _w_scale: int | None = field(default=None, init=False)
    _quantized_weights: np.ndarray | None = field(default=None, init=False)
    _quantized_bias: np.ndarray | None = field(default=None, init=False)

    # ── shape traits ──────────────────────────────────────────────────────────

    @property
    def in_shape(self) -> tuple[int, ...]:
        return (self.in_channels, self.in_len)

    @property
    def out_shape(self) -> tuple[int, ...]:
        l_out = (self.in_len + 2 * self.padding - self.kernel) // self.stride + 1
        return (self.out_channels, l_out)

    def adopt_in_shape(self, shape: tuple[int, ...]) -> None:
        # A conv's output geometry follows its input's spatial dim, so — unlike
        # a dense synapse — it must record what its predecessor produced. The
        # channel count is fixed by the weights; only L flows in.
        self.in_len = shape[1]

    # ── weight traits ─────────────────────────────────────────────────────────

    @property
    def weight_shape(self) -> tuple[int, ...]:
        return (self.out_channels, self.in_channels, self.kernel)

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
                "quantized snn.conv1d is not implemented yet (float lane only)",
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
                "quantized snn.conv1d is not implemented yet (float lane only)",
            )
        out_var = f"%synapse_{self.name}"
        in_ty = memref_type(self.in_shape, "f32")
        w_ty = memref_type(self.weight_shape, "f32")
        out_ty = memref_type(self.out_shape, "f32")
        c, ln = self.in_shape
        o = self.out_channels
        lines = [
            "",
            f"    // --- Conv1d {self.name}: ({c},{ln}) -> {self.out_shape} ---",
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
            f"    snn.conv1d ins({input_var}, %w_{self.name}){bias_part}"
            f" out({out_var})"
            f" {{stride = {self.stride} : i64,"
            f" padding = {self.padding} : i64}}"
            f" : {in_ty}, {w_ty} -> {out_ty}",
        ]
        return lines, out_var


def parse_conv1d(node: nir.Conv1d, name: str) -> Conv1dInfo:
    """``nir.Conv1d`` → :class:`Conv1dInfo`.

    Only the supported layout is accepted: ``dilation == 1`` and ``groups == 1``
    (a loud guard, like the ``k != 1`` check on CubaLIF). NIR weights are
    ``(O, C, K)``; the input shape ``(C, L)`` comes from the node's own
    ``input_type`` and is later cross-checked against what the predecessor
    actually produces (``_propagate_shapes``).
    """
    dilation = np.atleast_1d(node.dilation)
    if not np.all(dilation == 1):
        raise ValueError(
            f"Conv1d '{name}': dilation = {tuple(int(d) for d in dilation)} != 1 is not supported.",
        )
    groups = int(getattr(node, "groups", 1))
    if groups != 1:
        raise ValueError(f"Conv1d '{name}': groups = {groups} != 1 is not supported.")

    weights = np.array(node.weight, dtype=np.float32)  # (O, C, K)
    out_channels, in_channels, k = (int(d) for d in weights.shape)
    stride = _scalar(node.stride)
    padding = _scalar(node.padding)

    in_shape = nir_shape(node.input_type, "input", node=name)  # (C, L)
    if len(in_shape) != 2:
        raise ValueError(
            f"Conv1d '{name}': expected a rank-2 (C, L) input shape, got {in_shape}.",
        )

    bias = None if node.bias is None else np.array(node.bias, dtype=np.float32)
    return Conv1dInfo(
        name=name,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel=k,
        stride=stride,
        padding=padding,
        weights=weights,
        in_len=in_shape[1],
        bias=bias,
    )


def _scalar(value: object) -> int:
    """A scalar or length-1 NIR spatial field as a plain int."""
    return int(np.atleast_1d(value).flat[0])
