# Copyright 2026 N Vision Systems And Technologies SL
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from dataclasses import dataclass

import nir
import numpy as np
import numpy.typing as npt

from ._base import NodeInfo, memref_type, nir_shape

__all__ = ["AvgPool2dInfo", "parse_avgpool2d"]


@dataclass
class AvgPool2dInfo(NodeInfo):
    """A 2-D average-pooling layer: ``snn.avgpool2d``.

    Same shape behaviour as ``SumPool2dInfo`` (rank-3 ``(C, H, W)`` in and out,
    channels preserved, spatial dims shrunk by the pooling formula, no weights or
    state). A separate node because averaging divides each window by its count;
    output geometry follows the input, so ``in_shape`` is carried through by
    ``adopt_in_shape``.
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
        self.in_shape_ = shape

    # ── MLIR body emission ────────────────────────────────────────────────────

    def emit_mlir(
        self,
        input_var: str,
        is_last: bool,
        quantize: bool,
    ) -> tuple[list[str], str]:
        out_var = f"%pool_{self.name}"
        elem = "i8" if quantize else "f32"  # truncating integer mean: i8 -> i8
        in_ty = memref_type(self.in_shape, elem)
        out_ty = memref_type(self.out_shape, elem)
        kh, kw = self.kernel
        sh, sw = self.stride
        ph, pw = self.padding
        lines = [
            "",
            f"    // --- AvgPool2d {self.name}: {self.in_shape} -> {self.out_shape} ---",
            f"    {out_var} = memref.alloca() : {out_ty}",
            f"    snn.avgpool2d ins({input_var}) out({out_var})"
            f" {{kernel = array<i64: {kh}, {kw}>,"
            f" stride = array<i64: {sh}, {sw}>,"
            f" padding = array<i64: {ph}, {pw}>}}"
            f" : {in_ty} -> {out_ty}",
        ]
        return lines, out_var


def parse_avgpool2d(node: nir.AvgPool2d, name: str) -> AvgPool2dInfo:
    """``nir.AvgPool2d`` → :class:`AvgPool2dInfo`.

    NIR carries ``kernel_size``, ``stride`` and ``padding`` (scalars or length-2
    pairs). The input shape ``(C, H, W)`` comes from the node's ``input_type``
    and is cross-checked against the predecessor's output (``_propagate_shapes``).
    """
    kernel = _pair(node.kernel_size)
    stride = _pair(node.stride)
    padding = _pair(node.padding)

    in_shape = nir_shape(node.input_type, "input", node=name)  # (C, H, W)
    if len(in_shape) != 3:
        raise ValueError(
            f"AvgPool2d '{name}': expected a rank-3 (C, H, W) input shape, got {in_shape}.",
        )
    return AvgPool2dInfo(
        name=name,
        kernel=kernel,
        stride=stride,
        padding=padding,
        in_shape_=in_shape,
    )


def _pair(value: npt.ArrayLike) -> tuple[int, int]:
    """A scalar or length-2 NIR field as a ``(vertical, horizontal)`` pair."""
    arr = np.atleast_1d(value)
    if arr.size == 1:
        return (int(arr.flat[0]), int(arr.flat[0]))
    return (int(arr.flat[0]), int(arr.flat[1]))
