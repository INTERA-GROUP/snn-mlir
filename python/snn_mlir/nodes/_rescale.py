# Copyright 2026 N Vision Systems And Technologies SL
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import math
from dataclasses import dataclass

from ._base import NodeInfo, memref_type


@dataclass
class RescaleInfo(NodeInfo):
    """Synthetic node inserted between a synapse and a neuron in quantized mode.

    Not present in the NIR graph — created by _graph.insert_rescale_nodes()
    once quantization parameters are known for both the preceding synapse layer
    (w_scale) and the following neuron layer (d_scale).
    """

    shape: tuple[int, ...]
    _w_scale: int
    _d_scale: int

    # ── shape traits ──────────────────────────────────────────────────────────
    #
    # A rescale is a per-element shift, so it is shape-preserving like a neuron
    # but is neither synapse nor neuron; it carries its own small shape block.

    @property
    def size(self) -> int:
        """Flat element count — what the emitters and the C ABI measure in."""
        return math.prod(self.shape)

    @property
    def in_shape(self) -> tuple[int, ...]:
        return self.shape

    @property
    def out_shape(self) -> tuple[int, ...]:
        return self.shape

    def adopt_in_shape(self, shape: tuple[int, ...]) -> None:
        self.shape = shape

    @property
    def w_scale(self) -> int:
        return self._w_scale

    @property
    def d_scale(self) -> int:
        return self._d_scale

    def emit_mlir(
        self,
        input_var: str,
        is_last: bool,
        quantize: bool,
    ) -> tuple[list[str], str]:
        out_var = f"%rescaled_{self.name}"
        shift = self._d_scale - self._w_scale
        memref_t = memref_type(self.shape, "i32")
        return [
            "",
            f"    // --- Rescale {self.name}: (2^{self._w_scale}) -> i32 (2^{self._d_scale}), shift {shift} ---",  # noqa: E501
            f"    {out_var} = memref.alloca() : {memref_t}",
            f"    snn.rescale ins({input_var}) out({out_var})"
            f" {{w_scale = {self._w_scale} : i64, d_scale = {self._d_scale} : i64}}"
            f" : {memref_t} -> {memref_t}",
        ], out_var
