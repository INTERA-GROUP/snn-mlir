# Copyright 2026 N Vision Systems And Technologies SL
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from abc import ABC, abstractmethod

import numpy as np

from .._cname import c_identifier


class NodeInfo(ABC):
    """Base class for parsed NIR nodes.

    Trait properties default to False / None so graph-level logic can branch on
    ``is_synapse`` / ``is_neuron`` without isinstance checks, keeping the graph
    walker independent of concrete node types.
    """

    # ── naming ────────────────────────────────────────────────────────────────

    @property
    def c_name(self) -> str:
        """The node name as a valid C identifier (see ``snn_mlir._cname``).

        MLIR emission uses ``name`` verbatim (MLIR identifiers allow dots);
        every C generator derives its variable/macro names from this instead.
        """
        return c_identifier(self.name)  # type: ignore[attr-defined]

    # ── classification traits ─────────────────────────────────────────────────

    @property
    def is_synapse(self) -> bool:
        """True for weight-carrying layers: Linear, Affine, Conv (future)."""
        return False

    @property
    def is_neuron(self) -> bool:
        """True for state-carrying neuron layers: CubaLIF, LIF, CubaLI, LI."""
        return False

    # ── weight traits (synapse layers) ────────────────────────────────────────

    @property
    def weight_shape(self) -> tuple[int, int] | None:
        return None

    @property
    def float_weights(self) -> np.ndarray | None:
        return None

    @property
    def float_bias(self) -> np.ndarray | None:
        return None

    @property
    def int8_weights(self) -> np.ndarray | None:
        return None

    @property
    def int32_bias(self) -> np.ndarray | None:
        return None

    @property
    def w_scale(self) -> int | None:
        return None

    # ── neuron traits (neuron layers) ─────────────────────────────────────────

    @property
    def state_size(self) -> int | None:
        return None

    @property
    def state_names(self) -> list[str]:
        return []

    @property
    def state_size_define(self) -> str | None:
        return None

    @property
    def d_scale(self) -> int | None:
        return None

    def output_element_type(self, quantize: bool) -> str:
        raise NotImplementedError(
            f"{type(self).__name__}.output_element_type not implemented",
        )

    # ── quantization ──────────────────────────────────────────────────────────

    def quantize(self) -> None:  # noqa: B027  (optional hook, intentional no-op)
        """Compute and store this layer's quantization parameters in place.

        Default is a no-op; synapse and neuron layers that carry quantizable
        data override it. Called once per layer by ``graph.quantize_layers()``.
        """

    # ── MLIR module-level constant contributions ──────────────────────────────

    def weight_globals(self, quantize: bool) -> list[str]:
        """Module-level ``memref.global`` constant declarations for this layer."""
        return []

    # ── MLIR function argument contributions ──────────────────────────────────

    def state_func_args(self, quantize: bool) -> list[tuple[str, str]]:
        return []

    # ── MLIR body emission ────────────────────────────────────────────────────

    @abstractmethod
    def emit_mlir(
        self,
        input_var: str,
        is_last: bool,
        quantize: bool,
    ) -> tuple[list[str], str]:
        """Return (lines, output_var). output_var becomes input_var for next node."""
