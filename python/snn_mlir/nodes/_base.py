# Copyright 2026 N Vision Systems And Technologies SL
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from abc import ABC, abstractmethod

import numpy as np

from .._cname import c_identifier


def nir_shape(types: dict | None, key: str, *, node: str) -> tuple[int, ...]:
    """One NIR ``input_type``/``output_type`` entry as a plain tuple of ints.

    NIR stores these as ``{"input": np.array([...])}``. Reading element ``[0]``
    of that array — which every neuron parser used to do — silently truncates
    anything past rank 1: a ``(16, 16, 16)`` feature map reads back as 16
    neurons instead of 4096. Taking the whole entry is the only correct answer,
    and ``NodeInfo.size`` derives the flat count from it.

    Raises when the entry is missing: NIR nodes constructed in Python (rather
    than read from a file) can carry no shape at all — ``SumPool2d`` and
    ``AvgPool2d`` null theirs unconditionally in ``__post_init__`` — and a
    silently-zero shape would be far worse than a refusal.
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
    ``memref<16x16x16xi8>``. Emitters build type strings by hand in a dozen
    places, and every one of them used to interpolate a single flat count, which
    is only correct while every layer is rank 1. Routing them through one helper
    means a rank-changing layer needs no new formatting code — and, more to the
    point, that there is exactly one place where the ``x``-separated MLIR spelling
    is known.
    """
    if not shape:
        raise ValueError(
            "memref_type needs at least one dimension; a rank-0 memref carries "
            "no layer shape.",
        )
    dims = "x".join(str(int(d)) for d in shape)
    return f"memref<{dims}x{elem}>"


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

    # ── shape traits ──────────────────────────────────────────────────────────
    #
    # Every layer knows the shape of the tensor it reads and the one it writes.
    # For the fully-connected set both are rank-1, but the graph walk propagates
    # them generically so a rank-changing node (Conv2d, pooling, Flatten) drops
    # in without the walk learning what it does. ``None`` means "this layer does
    # not constrain the shape" and propagation passes through it unchanged.

    @property
    def in_shape(self) -> tuple[int, ...] | None:
        """Shape of the tensor this layer reads, or None if unconstrained."""
        return None

    @property
    def out_shape(self) -> tuple[int, ...] | None:
        """Shape of the tensor this layer writes, or None if unconstrained."""
        return None

    def adopt_in_shape(self, shape: tuple[int, ...]) -> None:  # noqa: B027
        """Take the shape propagated from this layer's predecessor.

        Shape-preserving layers (neurons, rescale) override this to record the
        shape they were handed; layers whose output shape is fixed by their own
        parameters (a synapse's weight matrix) leave it a no-op. Called by
        ``_graph.parse_graph`` in execution order, so a layer is always handed
        the shape its predecessor actually produces rather than the one NIR
        declared for it (NIR's own inference is not always right — see
        docs/python/nir-mapping.md).
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
        C ABI: the generated descriptors in ``_codegen`` must agree with them
        exactly. Implementations still spell them from the flat ``size``, which
        pins them to rank 1 — deliberately, because widening them is not a
        dialect change but an ABI change, and it has to move together with the
        C side and with ``%input``/``%output`` in ``_emit``. The op bodies have
        already moved to ``memref_type(shape, …)``; at rank 1 the two spellings
        are the same string, so nothing disagrees today.
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
