# Copyright 2026 N Vision Systems And Technologies SL
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import math
from dataclasses import dataclass

import nir

from ._base import NodeInfo, memref_type

__all__ = ["FlattenInfo", "parse_flatten"]


@dataclass
class FlattenInfo(NodeInfo):
    """A shape-collapsing layer: ``memref.collapse_shape``.

    Flatten is the rank boundary between the convolutional body (rank-N feature
    maps) and the dense tail (rank-1 vectors). It carries no weights, no state
    and does no arithmetic — it reinterprets one contiguous buffer at a lower
    rank — so it lowers to a plain ``memref.collapse_shape`` rather than an snn
    op, and is element-type agnostic (it needs no float/quant guard).

    ``start_dim``/``end_dim`` name the (inclusive) run of input dims folded into
    one; dims outside that run pass through. With ``start_dim=0, end_dim=-1``
    the whole feature map collapses to a rank-1 vector.
    """

    start_dim: int
    end_dim: int
    in_shape_: tuple[int, ...] = ()  # refined by adopt_in_shape

    # ── shape traits ──────────────────────────────────────────────────────────

    def _bounds(self) -> tuple[int, int]:
        """``(start, end)`` normalized against the current input rank."""
        rank = len(self.in_shape_)
        start = self.start_dim if self.start_dim >= 0 else self.start_dim + rank
        end = self.end_dim if self.end_dim >= 0 else self.end_dim + rank
        return start, end

    @property
    def in_shape(self) -> tuple[int, ...] | None:
        # None (unconstrained) until the graph walk hands over the predecessor's
        # shape — a Flatten declares no geometry of its own.
        return self.in_shape_ or None

    @property
    def out_shape(self) -> tuple[int, ...] | None:
        if not self.in_shape_:
            return None
        start, end = self._bounds()
        folded = math.prod(self.in_shape_[start : end + 1])
        return (*self.in_shape_[:start], folded, *self.in_shape_[end + 1 :])

    def adopt_in_shape(self, shape: tuple[int, ...]) -> None:
        # A pure reshape: it takes whatever its predecessor produced.
        self.in_shape_ = shape

    # ── MLIR body emission ────────────────────────────────────────────────────

    def emit_mlir(
        self,
        input_var: str,
        is_last: bool,
        quantize: bool,
    ) -> tuple[list[str], str]:
        scalar_t = "i8" if quantize else "f32"
        out_var = f"%flat_{self.name}"
        in_ty = memref_type(self.in_shape, scalar_t)
        out_ty = memref_type(self.out_shape, scalar_t)

        # Reassociation: the folded run [start..end] is one group; every other
        # input dim is a singleton group, so the groups tile all input dims.
        start, end = self._bounds()
        rank = len(self.in_shape_)
        groups = (
            [[d] for d in range(start)]
            + [list(range(start, end + 1))]
            + [[d] for d in range(end + 1, rank)]
        )
        reassoc = ", ".join("[" + ", ".join(str(d) for d in g) + "]" for g in groups)

        lines = [
            "",
            f"    // --- Flatten {self.name}: {self.in_shape} -> {self.out_shape} ---",
            f"    {out_var} = memref.collapse_shape {input_var} [{reassoc}]"
            f" : {in_ty} into {out_ty}",
        ]
        return lines, out_var


def parse_flatten(node: nir.Flatten, name: str) -> FlattenInfo:
    """``nir.Flatten`` → :class:`FlattenInfo`.

    The input shape is left for ``_propagate_shapes`` to fill via
    ``adopt_in_shape`` — a Flatten's geometry is entirely its predecessor's.
    """
    return FlattenInfo(
        name=name,
        start_dim=int(node.start_dim),
        end_dim=int(node.end_dim),
    )
