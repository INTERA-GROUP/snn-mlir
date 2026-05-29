# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

import nir
import numpy as np

from ._base import NodeInfo

__all__ = ["LIFInfo", "parse_lif"]

_D_SCALE = 12


@dataclass
class LIFInfo(NodeInfo):
    name: str
    size: int
    decay: float
    threshold: float
    v_reset: float = 0.0
    decay_scaled: int | None = field(default=None, init=False)
    threshold_scaled: int | None = field(default=None, init=False)
    v_reset_scaled: int | None = field(default=None, init=False)

    # ── classification traits ─────────────────────────────────────────────────

    @property
    def is_neuron(self) -> bool:
        return True

    # ── neuron traits ─────────────────────────────────────────────────────────

    @property
    def state_size(self) -> int:
        return self.size

    @property
    def state_names(self) -> list[str]:
        return ["voltage"]

    @property
    def state_size_define(self) -> str:
        return f"L{self.name}_LIF_SIZE"

    @property
    def d_scale(self) -> int:
        return _D_SCALE

    def output_element_type(self, quantize: bool) -> str:
        return "i8" if quantize else "f32"

    # ── quantization ──────────────────────────────────────────────────────────

    def quantize(self) -> None:
        self.decay_scaled = round(self.decay * (1 << _D_SCALE))
        self.threshold_scaled = round(self.threshold * (1 << _D_SCALE))
        self.v_reset_scaled = round(self.v_reset * (1 << _D_SCALE))

    # ── MLIR function arg contributions ──────────────────────────────────────

    def state_func_args(self, quantize: bool) -> list[tuple[str, str]]:
        t = "i32" if quantize else "f32"
        return [(f"%voltage_{self.name}", f"memref<{self.size}x{t}>")]

    # ── MLIR body emission ────────────────────────────────────────────────────

    def emit_mlir(
        self,
        input_var: str,
        is_last: bool,
        quantize: bool,
    ) -> tuple[list[str], str]:
        out_var = "%output" if is_last else f"%spikes_{self.name}"
        alloca = not is_last
        if quantize:
            lines = _emit_lif_int(self, input_var, f"%voltage_{self.name}", out_var, alloca)
        else:
            lines = _emit_lif_float(self, input_var, f"%voltage_{self.name}", out_var, alloca)
        return lines, out_var


def parse_lif(node: nir.LIF, name: str) -> LIFInfo:
    if not np.allclose(node.v_leak, 0.0):
        raise ValueError("v_leak must be 0 for LIF")
    if np.unique(node.tau).size != 1:
        raise ValueError("tau must be uniform across all LIF neurons")
    if np.unique(node.r).size != 1:
        raise ValueError("r must be uniform across all LIF neurons")
    if np.unique(node.v_threshold).size != 1:
        raise ValueError("v_threshold must be uniform across all LIF neurons")
    if np.unique(node.v_reset).size != 1:
        raise ValueError("v_reset must be uniform across all LIF neurons")

    dt = float(node.tau[0] / node.r[0])
    decay = float(1 - (dt / node.tau[0]))
    size = int(node.input_type["input"][0])

    return LIFInfo(
        name=name,
        size=size,
        decay=decay,
        threshold=float(node.v_threshold[0]),
        v_reset=float(node.v_reset[0]),
    )


# ── MLIR emission helpers ──────────────────────────────────────────────────────


def _emit_lif_float(
    info: LIFInfo,
    input_var: str,
    voltage_var: str,
    output_var: str,
    alloca_output: bool,
) -> list[str]:
    n = info.size
    lines = ["", f"    // --- LIF {info.name}: ({n}) neurons ---"]
    if alloca_output:
        lines.append(f"    {output_var} = memref.alloca() : memref<{n}xf32>")
    lines.append(
        f"    snn.lif ins({input_var}) state({voltage_var}) out({output_var})"
        f" {{decay_float = {info.decay:.10e} : f64,"
        f" threshold_float = {info.threshold:.10e} : f64,"
        f" v_reset_float = {info.v_reset:.10e} : f64}}"
        f" : memref<{n}xf32>, memref<{n}xf32>"
        f" -> memref<{n}xf32>",
    )
    return lines


def _emit_lif_int(
    info: LIFInfo,
    input_var: str,
    voltage_var: str,
    output_var: str,
    alloca_output: bool,
) -> list[str]:
    n = info.size
    lines = ["", f"    // --- LIF {info.name}: ({n}) neurons, Q{_D_SCALE} ---"]
    if alloca_output:
        lines.append(f"    {output_var} = memref.alloca() : memref<{n}xi8>")
    lines.append(
        f"    snn.lif ins({input_var}) state({voltage_var}) out({output_var})"
        f" {{d_scale = {_D_SCALE} : i64,"
        f" decay_int = {info.decay_scaled} : i64,"
        f" threshold_int = {info.threshold_scaled} : i64,"
        f" v_reset_int = {info.v_reset_scaled} : i64}}"
        f" : memref<{n}xi32>, memref<{n}xi32>"
        f" -> memref<{n}xi8>",
    )
    return lines
