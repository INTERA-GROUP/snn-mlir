# Copyright 2026 N Vision Systems And Technologies SL
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from dataclasses import dataclass, field

import nir
import numpy as np

from ._base import NodeInfo

__all__ = ["CubaLIFInfo", "parse_cubalif"]

_D_SCALE = 12


@dataclass
class CubaLIFInfo(NodeInfo):
    name: str
    size: int
    cur_decay: float
    vol_decay: float
    threshold: float
    cur_decay_scaled: int | None = field(default=None, init=False)
    vol_decay_scaled: int | None = field(default=None, init=False)
    threshold_scaled: int | None = field(default=None, init=False)

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
        return ["current", "voltage"]

    @property
    def state_size_define(self) -> str:
        return f"L{self.name}_CUBALIF_SIZE"

    @property
    def d_scale(self) -> int:
        return _D_SCALE

    def output_element_type(self, quantize: bool) -> str:
        return "i8" if quantize else "f32"

    # ── quantization ──────────────────────────────────────────────────────────

    def quantize(self) -> None:
        self.cur_decay_scaled = round(self.cur_decay * (1 << _D_SCALE))
        self.vol_decay_scaled = round(self.vol_decay * (1 << _D_SCALE))
        self.threshold_scaled = round(self.threshold * (1 << _D_SCALE))

    # ── MLIR function arg contributions ──────────────────────────────────────

    def state_func_args(self, quantize: bool) -> list[tuple[str, str]]:
        t = "i32" if quantize else "f32"
        return [
            (f"%current_{self.name}", f"memref<{self.size}x{t}>"),
            (f"%voltage_{self.name}", f"memref<{self.size}x{t}>"),
        ]

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
            lines = _emit_cubalif_int(
                self,
                input_var,
                f"%current_{self.name}",
                f"%voltage_{self.name}",
                out_var,
                alloca,
            )
        else:
            lines = _emit_cubalif_float(
                self,
                input_var,
                f"%current_{self.name}",
                f"%voltage_{self.name}",
                out_var,
                alloca,
            )
        return lines, out_var


def parse_cubalif(node: nir.CubaLIF, name: str) -> CubaLIFInfo:
    dt = float(node.tau_mem[0] / node.r[0])
    cur_decay = float(1 - (dt / node.tau_syn[0]))
    vol_decay = float(1 - (dt / node.tau_mem[0]))
    threshold = float(node.v_threshold[0])
    size = int(node.input_type["input"][0])

    if not np.allclose(node.v_leak, 0):
        raise ValueError("v_leak must be 0 for CubaLIF")
    if not np.allclose(node.v_reset, 0):
        raise ValueError("CubaLIF v_reset != 0 not supported yet")
    if np.unique(node.tau_syn).size != 1:
        raise ValueError("cur_decay must be uniform across all neurons")
    if np.unique(node.tau_mem).size != 1:
        raise ValueError("vol_decay must be uniform across all neurons")
    if np.unique(node.v_threshold).size != 1:
        raise ValueError("v_threshold must be uniform across all neurons")

    return CubaLIFInfo(
        name=name,
        size=size,
        cur_decay=cur_decay,
        vol_decay=vol_decay,
        threshold=threshold,
    )


# ── MLIR emission helpers ──────────────────────────────────────────────────────


def _emit_cubalif_float(
    info: CubaLIFInfo,
    input_var: str,
    current_var: str,
    voltage_var: str,
    output_var: str,
    alloca_output: bool,
) -> list[str]:
    n = info.size
    lines = ["", f"    // --- CubaLIF {info.name}: ({n}) neurons ---"]
    if alloca_output:
        lines.append(f"    {output_var} = memref.alloca() : memref<{n}xf32>")
    lines.append(
        f"    snn.cubalif ins({input_var}) state({current_var}, {voltage_var})"
        f" out({output_var})"
        f" {{cur_decay_float = {info.cur_decay:.10e} : f64,"
        f" vol_decay_float = {info.vol_decay:.10e} : f64,"
        f" threshold_float = {info.threshold:.10e} : f64}}"
        f" : memref<{n}xf32>, memref<{n}xf32>, memref<{n}xf32>"
        f" -> memref<{n}xf32>",
    )
    return lines


def _emit_cubalif_int(
    info: CubaLIFInfo,
    input_var: str,
    current_var: str,
    voltage_var: str,
    output_var: str,
    alloca_output: bool,
) -> list[str]:
    n = info.size
    lines = ["", f"    // --- CubaLIF {info.name}: ({n}) neurons, Q{_D_SCALE} ---"]
    if alloca_output:
        lines.append(f"    {output_var} = memref.alloca() : memref<{n}xi8>")
    lines.append(
        f"    snn.cubalif ins({input_var}) state({current_var}, {voltage_var})"
        f" out({output_var})"
        f" {{d_scale = {_D_SCALE} : i64,"
        f" cur_decay_int = {info.cur_decay_scaled} : i64,"
        f" vol_decay_int = {info.vol_decay_scaled} : i64,"
        f" threshold_int = {info.threshold_scaled} : i64}}"
        f" : memref<{n}xi32>, memref<{n}xi32>, memref<{n}xi32>"
        f" -> memref<{n}xi8>",
    )
    return lines
