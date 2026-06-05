# Copyright 2026 N Vision Systems And Technologies SL
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from dataclasses import dataclass

from ._base import NodeInfo


@dataclass
class RescaleInfo(NodeInfo):
    """Synthetic node inserted between a synapse and a neuron in quantized mode.

    Not present in the NIR graph — created by _graph.insert_rescale_nodes()
    once quantization parameters are known for both the preceding synapse layer
    (w_scale) and the following neuron layer (d_scale).
    """

    name: str
    size: int
    _w_scale: int
    _d_scale: int

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
        return [
            "",
            f"    // --- Rescale {self.name}: (2^{self._w_scale}) -> i32 (2^{self._d_scale}), shift {shift} ---",  # noqa: E501
            f"    {out_var} = memref.alloca() : memref<{self.size}xi32>",
            f"    snn.rescale ins({input_var}) out({out_var})"
            f" {{w_scale = {self._w_scale} : i64, d_scale = {self._d_scale} : i64}}"
            f" : memref<{self.size}xi32> -> memref<{self.size}xi32>",
        ], out_var
