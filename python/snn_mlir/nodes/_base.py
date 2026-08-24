# Copyright 2026 N Vision Systems And Technologies SL
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .._cname import c_identifier


def nir_shape(types: dict | None, key: str, *, node: str) -> tuple[int, ...]:
    """One NIR ``input_type``/``output_type`` entry as a tuple of ints.

    NIR stores these as ``{"input": np.array([...])}``. Reading element ``[0]``
    truncates anything past rank 1 (a ``(16, 16, 16)`` feature map would read
    back as 16), so the whole entry is taken and ``NodeInfo.size`` derives the
    flat count from it. Raises when the entry is missing rather than yielding a
    silently-zero shape: NIR nodes built in Python can carry no shape at all.
    """
    entry = (types or {}).get(key)
    if entry is None:
        raise ValueError(
            f"Node '{node}': NIR declares no {key} shape, so the layer's size is "
            f"unknown. Re-export the model with shapes, or call "
            f"nir.NIRGraph.infer_types() on the graph before converting it."
        )
    return tuple(int(d) for d in np.atleast_1d(entry))


def memref_type(shape: tuple[int, ...], elem: str) -> str:
    """The MLIR memref type for a tensor of `shape` holding `elem` scalars.

    ``(200,), "f32"`` → ``memref<200xf32>``; ``(16, 16, 16), "i8"`` →
    ``memref<16x16x16xi8>``. The one place the ``x``-separated spelling lives.
    """
    if not shape:
        raise ValueError(
            "memref_type needs at least one dimension; a rank-0 memref carries "
            "no layer shape.",
        )
    dims = "x".join(str(int(d)) for d in shape)
    return f"memref<{dims}x{elem}>"


@dataclass
class NodeInfo(ABC):
    """Base class for parsed NIR nodes.

    Trait properties default to False / None so graph-level logic can branch on
    ``is_synapse`` / ``is_neuron`` without isinstance checks, keeping the graph
    walker independent of concrete node types. ``NeuronInfo`` and ``SynapseInfo``
    specialise the two roles.
    """

    name: str

    # ── naming ────────────────────────────────────────────────────────────────

    @property
    def c_name(self) -> str:
        """The node name as a valid C identifier (see ``snn_mlir._cname``).

        MLIR emission uses ``name`` verbatim (MLIR identifiers allow dots);
        every C generator derives its variable/macro names from this instead.
        """
        return c_identifier(self.name)

    # ── classification traits ─────────────────────────────────────────────────

    @property
    def is_synapse(self) -> bool:
        """True for weight-carrying layers: Linear, Affine, Conv (future)."""
        return False

    @property
    def is_neuron(self) -> bool:
        """True for state-carrying neuron layers: CubaLIF, LIF, CubaLI, LI."""
        return False

    # ── shape traits ──────────────────────────────────────────────────────────
    #
    # Every layer reports the shape it reads and the one it writes; the graph
    # walk propagates them generically so a rank-changing node (Conv2d, pooling,
    # Flatten) drops in without the walk learning what it does. ``None`` means
    # "unconstrained" and propagation passes through unchanged.

    @property
    def in_shape(self) -> tuple[int, ...] | None:
        return None

    @property
    def out_shape(self) -> tuple[int, ...] | None:
        return None

    def adopt_in_shape(self, shape: tuple[int, ...]) -> None:  # noqa: B027
        """Take the shape propagated from this layer's predecessor.

        Shape-preserving layers record it; layers whose output shape is fixed by
        their own parameters (a synapse's weight matrix) leave this a no-op.
        """

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
        """``(name, memref type)`` for each state buffer this layer needs.

        These are function arguments, so their types are half of the positional
        C ABI and must match the descriptors ``_codegen`` generates. Spelled
        from the layer's ``shape``, so a conv-fed neuron's state is a rank-N
        feature map — the same rank its op body and the C descriptor use.
        """
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


@dataclass
class NeuronInfo(NodeInfo):
    """A point neuron: state-carrying, elementwise, shape-preserving.

    Base for CubaLIF, LIF, CubaLI and LI. Owns the shape it carries; concrete
    neurons add their decays, threshold and reset, and the neuron/MLIR traits.
    """

    shape: tuple[int, ...]

    @property
    def is_neuron(self) -> bool:
        return True

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
        # Shape-preserving: the neuron takes whatever its predecessor produced.
        self.shape = shape


@dataclass
class SynapseInfo(NodeInfo):
    """A weight-carrying layer: the shape is decided here, not carried through.

    Base for Linear/Affine (and Conv, later). The weight matrix fixes both ends,
    so ``adopt_in_shape`` stays the base no-op; concrete synapses supply their
    own ``in_shape``/``out_shape`` and the weight traits.
    """

    @property
    def is_synapse(self) -> bool:
        return True
