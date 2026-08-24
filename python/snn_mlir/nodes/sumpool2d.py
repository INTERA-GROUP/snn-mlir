# Copyright 2026 N Vision Systems And Technologies SL
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from dataclasses import dataclass

import nir
import numpy as np

from ._base import NodeInfo, memref_type, nir_shape

__all__ = ["SumPool2dInfo", "parse_sumpool2d"]


@dataclass
class SumPool2dInfo(NodeInfo):
    """A 2-D sum-pooling layer: ``snn.sumpool2d``.

    Activations are rank-3 feature maps ``(C, H, W)``; the channel count is
    preserved and the spatial dims shrink by the pooling formula. It carries no
    weights and no state, so it is neither a synapse nor a neuron — a pure
    rank-preserving, shape-shrinking reshape that rides the graph's generic
    shape propagation. Its output geometry depends on the *input* spatial size,
    so ``in_shape`` is carried through from the predecessor (``adopt_in_shape``).
    """

    kernel: tuple[int, int]  # (kh, kw)
    stride: tuple[int, int]  # (sh, sw)
    padding: tuple[int, int]  # (ph, pw)
    in_shape_: tuple[int, ...]  # (C, H, W), refined by adopt_in_shape

    # ── shape traits ──────────────────────────────────────────────────────────

    @property
    def in_shape(self) -> tuple[int, ...]:
        return self.in_shape_

    @property
    def out_shape(self) -> tuple[int, ...]:
        c, h, w = self.in_shape_
        kh, kw = self.kernel
        sh, sw = self.stride
        ph, pw = self.padding
        h_out = (h + 2 * ph - kh) // sh + 1
        w_out = (w + 2 * pw - kw) // sw + 1
        return (c, h_out, w_out)

    def adopt_in_shape(self, shape: tuple[int, ...]) -> None:
        # Pooling preserves channels and reshapes only H, W, so its whole output
        # geometry follows its predecessor's feature map — record it verbatim.
        self.in_shape_ = shape

    # ── MLIR body emission ────────────────────────────────────────────────────

    def emit_mlir(
        self,
        input_var: str,
        is_last: bool,
        quantize: bool,
    ) -> tuple[list[str], str]:
        if quantize:
            raise NotImplementedError(
                "quantized snn.sumpool2d is not implemented yet (float lane only)",
            )
        out_var = f"%pool_{self.name}"
        in_ty = memref_type(self.in_shape, "f32")
        out_ty = memref_type(self.out_shape, "f32")
        kh, kw = self.kernel
        sh, sw = self.stride
        ph, pw = self.padding
        lines = [
            "",
            f"    // --- SumPool2d {self.name}: {self.in_shape} -> {self.out_shape} ---",
            f"    {out_var} = memref.alloca() : {out_ty}",
            f"    snn.sumpool2d ins({input_var}) out({out_var})"
            f" {{kernel = array<i64: {kh}, {kw}>,"
            f" stride = array<i64: {sh}, {sw}>,"
            f" padding = array<i64: {ph}, {pw}>}}"
            f" : {in_ty} -> {out_ty}",
        ]
        return lines, out_var


def parse_sumpool2d(node: nir.SumPool2d, name: str) -> SumPool2dInfo:
    """``nir.SumPool2d`` → :class:`SumPool2dInfo`.

    NIR carries ``kernel_size``, ``stride`` and ``padding`` (scalars or length-2
    pairs). The input shape ``(C, H, W)`` comes from the node's own
    ``input_type`` and is later cross-checked against what the predecessor
    actually produces (``_propagate_shapes``).
    """
    kernel = _pair(node.kernel_size)
    stride = _pair(node.stride)
    padding = _pair(node.padding)

    in_shape = nir_shape(node.input_type, "input", node=name)  # (C, H, W)
    if len(in_shape) != 3:
        raise ValueError(
            f"SumPool2d '{name}': expected a rank-3 (C, H, W) input shape, got {in_shape}.",
        )
    return SumPool2dInfo(
        name=name,
        kernel=kernel,
        stride=stride,
        padding=padding,
        in_shape_=in_shape,
    )


def _pair(value: object) -> tuple[int, int]:
    """A scalar or length-2 NIR field as a ``(vertical, horizontal)`` pair."""
    arr = np.atleast_1d(value)
    if arr.size == 1:
        return (int(arr.flat[0]), int(arr.flat[0]))
    return (int(arr.flat[0]), int(arr.flat[1]))
