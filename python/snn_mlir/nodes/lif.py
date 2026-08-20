# Copyright 2026 N Vision Systems And Technologies SL
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import math
from dataclasses import dataclass, field

import nir
import numpy as np

from ._base import NodeInfo, memref_type, nir_shape

__all__ = ["LIFInfo", "parse_if", "parse_lif"]

_D_SCALE = 12


@dataclass
class LIFInfo(NodeInfo):
    name: str
    shape: tuple[int, ...]
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
        return f"L{self.c_name}_LIF_SIZE"

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
    if not np.allclose(node.v_reset, 0.0):
        raise ValueError("LIF v_reset != 0 not supported yet")

    dt = float(node.tau[0] / node.r[0])
    decay = float(1 - (dt / node.tau[0]))
    shape = nir_shape(node.input_type, "input", node=name)

    return LIFInfo(
        name=name,
        shape=shape,
        decay=decay,
        threshold=float(node.v_threshold[0]),
        v_reset=float(node.v_reset[0]),
    )


def parse_if(node: nir.IF, name: str) -> LIFInfo:
    """``nir.IF`` — the non-leaky integrate-and-fire neuron — as an ``snn.lif``.

    IF is NOT ``parse_lif`` with the decay forced to 1. ``nir.IF`` carries no
    ``tau`` and no ``v_leak`` at all, so there is nothing to derive a decay
    from: ``decay = 1`` is the *definition* of the node, not a computed result.
    Sharing a parser would simply raise ``AttributeError`` on ``node.tau``.

    The one field that needs a guard is ``r``, and it needs the opposite
    treatment from the one it gets on LIF. In ``tau*dv/dt = (v_leak - v) + r*i``
    the exporter convention is ``r = tau/dt``, which makes ``decay = 1 - 1/r``
    and leaves the input gain identically 1 for any ``r`` — the ``r`` in the
    equation and the ``r`` carrying ``dt`` cancel, which is why ``parse_lif``
    must NOT reject ``r != 1``: there ``r`` encodes the leak. IF's equation is
    ``dv/dt = r*i``: no ``tau``, so nothing carries ``dt`` and nothing cancels.
    ``r`` is left as a bare input gain, and the update this emits is
    ``voltage += input``, which assumes it is 1. Same class of guard as the
    ``k != 1`` check in ``parse_cubalif``.
    """
    if np.unique(node.r).size != 1:
        raise ValueError("r must be uniform across all IF neurons")
    if np.unique(node.v_threshold).size != 1:
        raise ValueError("v_threshold must be uniform across all IF neurons")
    if np.unique(node.v_reset).size != 1:
        raise ValueError("v_reset must be uniform across all IF neurons")
    if not np.allclose(node.r, 1.0):
        raise ValueError(
            f"IF '{name}': input gain r = {float(np.max(node.r)):.6g} != 1. Unlike LIF, "
            "an IF node has no tau for r to carry the timestep in, so r is a bare gain "
            "on the input and the emitted update (voltage += input) would compute "
            "different dynamics than were trained."
        )
    if not np.allclose(node.v_reset, 0.0):
        raise ValueError("IF v_reset != 0 not supported yet")

    return LIFInfo(
        name=name,
        shape=nir_shape(node.input_type, "input", node=name),
        decay=1.0,
        threshold=float(node.v_threshold.flat[0]),
        v_reset=float(node.v_reset.flat[0]),
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
    memref_t = memref_type(info.shape, "f32")
    lines = ["", f"    // --- LIF {info.name}: ({n}) neurons ---"]
    if alloca_output:
        lines.append(f"    {output_var} = memref.alloca() : {memref_t}")
    lines.append(
        f"    snn.lif ins({input_var}) state({voltage_var}) out({output_var})"
        f" {{decay_float = {info.decay:.10e} : f64,"
        f" threshold_float = {info.threshold:.10e} : f64,"
        f" v_reset_float = {info.v_reset:.10e} : f64}}"
        f" : {memref_t}, {memref_t}"
        f" -> {memref_t}",
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
    state_t = memref_type(info.shape, "i32")
    spike_t = memref_type(info.shape, "i8")
    lines = ["", f"    // --- LIF {info.name}: ({n}) neurons, Q{_D_SCALE} ---"]
    if alloca_output:
        lines.append(f"    {output_var} = memref.alloca() : {spike_t}")
    lines.append(
        f"    snn.lif ins({input_var}) state({voltage_var}) out({output_var})"
        f" {{d_scale = {_D_SCALE} : i64,"
        f" decay_int = {info.decay_scaled} : i64,"
        f" threshold_int = {info.threshold_scaled} : i64,"
        f" v_reset_int = {info.v_reset_scaled} : i64}}"
        f" : {state_t}, {state_t}"
        f" -> {spike_t}",
    )
    return lines
