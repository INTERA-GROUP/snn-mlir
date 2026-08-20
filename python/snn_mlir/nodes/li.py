# Copyright 2026 N Vision Systems And Technologies SL
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import math
from dataclasses import dataclass, field

import nir
import numpy as np

from ._base import NodeInfo, nir_shape

__all__ = ["LIInfo", "parse_i", "parse_li"]

_D_SCALE = 12


@dataclass
class LIInfo(NodeInfo):
    name: str
    shape: tuple[int, ...]
    decay: float
    decay_scaled: int | None = field(default=None, init=False)

    # ── classification traits ─────────────────────────────────────────────────

    @property
    def is_neuron(self) -> bool:
        return True

    # ── shape traits ──────────────────────────────────────────────────────────

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
        """A point neuron is shape-preserving: it wears whatever it is fed."""
        self.shape = shape

    # ── neuron traits ─────────────────────────────────────────────────────────

    @property
    def state_size(self) -> int:
        return self.size

    @property
    def state_names(self) -> list[str]:
        return ["voltage"]

    @property
    def state_size_define(self) -> str:
        return f"L{self.c_name}_LI_SIZE"

    @property
    def d_scale(self) -> int:
        return _D_SCALE

    def output_element_type(self, quantize: bool) -> str:
        return "i32" if quantize else "f32"

    # ── quantization ──────────────────────────────────────────────────────────

    def quantize(self) -> None:
        self.decay_scaled = round(self.decay * (1 << _D_SCALE))

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
        out_var = "%output" if is_last else f"%voltage_out_{self.name}"
        alloca = not is_last
        if quantize:
            lines = _emit_li_int(self, input_var, f"%voltage_{self.name}", out_var, alloca)
        else:
            lines = _emit_li_float(self, input_var, f"%voltage_{self.name}", out_var, alloca)
        return lines, out_var


def parse_li(node: nir.LI, name: str) -> LIInfo:
    # LI is a LIF neuron without the firing/reset terminal: same leaky-integrator
    # dynamics, so the decay is derived from tau/r exactly as in parse_lif.
    if not np.allclose(node.v_leak, 0.0):
        raise ValueError("v_leak must be 0 for LI")
    if np.unique(node.tau).size != 1:
        raise ValueError("tau must be uniform across all LI neurons")
    if np.unique(node.r).size != 1:
        raise ValueError("r must be uniform across all LI neurons")

    dt = float(node.tau[0] / node.r[0])
    decay = float(1 - (dt / node.tau[0]))
    return LIInfo(
        name=name,
        shape=nir_shape(node.input_type, "input", node=name),
        decay=decay,
    )


def parse_i(node: nir.I, name: str) -> LIInfo:
    """``nir.I`` — the non-leaky integrator — as an ``snn.li``.

    ``I`` is to ``LI`` exactly what ``IF`` is to ``LIF``: no ``tau``, no
    ``v_leak``, so ``decay = 1`` by definition rather than by derivation, and
    ``r`` is a bare input gain that must be 1 for the emitted ``voltage +=
    input`` to be the trained dynamics. See ``parse_if`` for the full argument.

    ``nir.I`` is the smallest NIR neuron there is — ``r`` is its only
    parameter — so that guard is the whole parser.
    """
    if np.unique(node.r).size != 1:
        raise ValueError("r must be uniform across all I neurons")
    if not np.allclose(node.r, 1.0):
        raise ValueError(
            f"I '{name}': input gain r = {float(np.max(node.r)):.6g} != 1. Unlike LI, "
            "an I node has no tau for r to carry the timestep in, so r is a bare gain "
            "on the input and the emitted update (voltage += input) would compute "
            "different dynamics than were trained."
        )

    return LIInfo(
        name=name,
        shape=nir_shape(node.input_type, "input", node=name),
        decay=1.0,
    )


# ── MLIR emission helpers ──────────────────────────────────────────────────────


def _emit_li_float(
    info: LIInfo,
    input_var: str,
    voltage_var: str,
    output_var: str,
    alloca_output: bool,
) -> list[str]:
    n = info.size
    lines = ["", f"    // --- LI {info.name}: ({n}) neurons ---"]
    if alloca_output:
        lines.append(f"    {output_var} = memref.alloca() : memref<{n}xf32>")
    lines.append(
        f"    snn.li ins({input_var}) state({voltage_var}) out({output_var})"
        f" {{decay_float = {info.decay:.10e} : f64}}"
        f" : memref<{n}xf32>, memref<{n}xf32>"
        f" -> memref<{n}xf32>",
    )
    return lines


def _emit_li_int(
    info: LIInfo,
    input_var: str,
    voltage_var: str,
    output_var: str,
    alloca_output: bool,
) -> list[str]:
    n = info.size
    lines = ["", f"    // --- LI {info.name}: ({n}) neurons, Q{_D_SCALE} ---"]
    if alloca_output:
        lines.append(f"    {output_var} = memref.alloca() : memref<{n}xi32>")
    lines.append(
        f"    snn.li ins({input_var}) state({voltage_var}) out({output_var})"
        f" {{d_scale = {_D_SCALE} : i64,"
        f" decay_int = {info.decay_scaled} : i64}}"
        f" : memref<{n}xi32>, memref<{n}xi32>"
        f" -> memref<{n}xi32>",
    )
    return lines
