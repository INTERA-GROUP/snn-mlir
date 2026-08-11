# Copyright 2026 N Vision Systems And Technologies SL
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Emission tests for recurrent graphs: state buffers, fan-in merge, goldens.

The emitted module is a positional C ABI (see _emit's module docstring), so the
argument order and the state-copy placement are locked byte-for-byte by two
golden files. The goldens use a deterministic model — regenerate them by
running this file's ``_golden_graph()`` through ``snn_mlir.to_mlir`` after any
*intended* emission change.
"""

import subprocess
from pathlib import Path

import nir
import numpy as np
import pytest
import snn_mlir
from snn_mlir._run import _resolve_optional

GOLDENS = Path(__file__).parent / "goldens"


def _golden_graph() -> nir.NIRGraph:
    """input(4) → fc1 → lif1(5) ⇄ w_rec → fc2 → lif2(3), fixed weights.

    The CubaLIF parameters follow the discrete-exporter convention with
    ``dt = 0.1``: ``r = tau_mem/dt``, ``w_in = tau_syn/dt`` (so k = 1), giving
    cur_decay 0.8 and vol_decay 0.5 — representative values, unlike the
    degenerate cur_decay = 0 that w_in = 1 forces.
    """

    def cubalif(size: int) -> nir.CubaLIF:
        return nir.CubaLIF(
            tau_syn=np.full(size, 0.5),
            tau_mem=np.full(size, 0.2),
            r=np.full(size, 2.0),
            w_in=np.full(size, 5.0),
            v_leak=np.zeros(size),
            v_threshold=np.ones(size),
            v_reset=np.zeros(size),
            input_type={"input": np.array([size])},
        )

    def weights(out_size: int, in_size: int) -> np.ndarray:
        # Deterministic, sign-varied, max|w| = 0.5 (natural w_scale 7).
        n = out_size * in_size
        w = np.linspace(-0.5, 0.5, n, dtype=np.float32) * np.where(np.arange(n) % 3 == 0, -1.0, 1.0)
        return w.astype(np.float32).reshape(out_size, in_size)

    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([4])}),
            "fc1": nir.Linear(weight=weights(5, 4)),
            "lif1": cubalif(5),
            "w_rec": nir.Linear(weight=weights(5, 5)),
            "fc2": nir.Linear(weight=weights(3, 5)),
            "lif2": cubalif(3),
            "output": nir.Output(output_type={"output": np.array([3])}),
        },
        edges=[
            ("input", "fc1"),
            ("fc1", "lif1"),
            ("lif1", "w_rec"),
            ("w_rec", "lif1"),
            ("lif1", "fc2"),
            ("fc2", "lif2"),
            ("lif2", "output"),
        ],
    )


# ── structure ─────────────────────────────────────────────────────────────────


def test_prev_spikes_arg_follows_the_neurons_own_state():
    """The positional ABI: %prev_spikes_<n> comes right after <n>'s state
    buffers, before any later layer's state."""
    mlir = snn_mlir.to_mlir(_golden_graph(), quantize=True)
    args = mlir.split("attributes")[0]
    order = [
        args.index("%current_lif1"),
        args.index("%voltage_lif1"),
        args.index("%prev_spikes_lif1 : memref<5xi8>"),
        args.index("%current_lif2"),
    ]
    assert order == sorted(order)


def test_prev_spikes_is_float_typed_in_float_mode():
    mlir = snn_mlir.to_mlir(_golden_graph(), quantize=False)
    assert "%prev_spikes_lif1 : memref<5xf32>" in mlir


def test_recurrent_synapse_reads_the_state_buffer():
    mlir = snn_mlir.to_mlir(_golden_graph(), quantize=True)
    assert "snn.linear ins(%prev_spikes_lif1, %w_w_rec)" in mlir


def test_merge_adds_the_rescaled_branches_before_the_neuron():
    mlir = snn_mlir.to_mlir(_golden_graph(), quantize=True)
    assert "ins(%rescaled_fc1, %rescaled_w_rec" in mlir
    assert "arith.addi" in mlir
    assert "snn.cubalif ins(%merged_lif1)" in mlir
    # The merge must come after both rescales and before the neuron.
    assert (
        mlir.index("%rescaled_w_rec = ")
        < mlir.index("%rescaled_fc1 = ")
        < mlir.index("%merged_lif1 = ")
        < mlir.index("snn.cubalif ins(%merged_lif1)")
    )


def test_float_merge_uses_addf_on_synapse_outputs():
    mlir = snn_mlir.to_mlir(_golden_graph(), quantize=False)
    assert "ins(%synapse_fc1, %synapse_w_rec" in mlir
    assert "arith.addf" in mlir
    assert "snn.rescale" not in mlir


def test_spikes_are_copied_into_the_state_buffer_at_timestep_end():
    for quantize, spike_t in ((True, "i8"), (False, "f32")):
        mlir = snn_mlir.to_mlir(_golden_graph(), quantize=quantize)
        copy = (
            f"memref.copy %spikes_lif1, %prev_spikes_lif1"
            f" : memref<5x{spike_t}> to memref<5x{spike_t}>"
        )
        assert copy in mlir
        # After every layer, right before the return.
        last_neuron = (
            "snn.cubalif ins(%rescaled_fc2)" if quantize else "snn.cubalif ins(%synapse_fc2)"
        )
        assert mlir.index(copy) > mlir.index(last_neuron)


def test_linear_chain_emits_no_recurrent_constructs(nir_linear_cubalif):
    for quantize in (False, True):
        mlir = snn_mlir.to_mlir(nir_linear_cubalif, quantize=quantize)
        assert "prev_spikes" not in mlir
        assert "memref.copy" not in mlir
        assert "linalg.generic" not in mlir


# ── goldens (the byte-level ABI lock) ─────────────────────────────────────────


@pytest.mark.parametrize(
    "quantize,name", [(False, "recurrent_float"), (True, "recurrent_quantized")]
)
def test_recurrent_emission_matches_golden(quantize, name):
    mlir = snn_mlir.to_mlir(_golden_graph(), quantize=quantize)
    golden = (GOLDENS / f"{name}.mlir").read_text()
    assert mlir == golden


# ── validity (needs a built snn-opt) ──────────────────────────────────────────


@pytest.mark.skipif(
    _resolve_optional()["snn_opt"] is None,
    reason="snn-opt not resolvable (build it, add to PATH, or set SNN_OPT)",
)
@pytest.mark.parametrize("quantize", [False, True])
def test_recurrent_emission_is_valid_and_lowers(tmp_path, quantize):
    snn_opt = _resolve_optional()["snn_opt"]
    f = tmp_path / "rec.mlir"
    f.write_text(snn_mlir.to_mlir(_golden_graph(), quantize=quantize))
    for extra in ([], ["--convert-snn-to-linalg"]):
        subprocess.run(
            [str(snn_opt), str(f), *extra, "-o", "/dev/null"],
            check=True,
            capture_output=True,
        )


# ── the C harness does not lie ────────────────────────────────────────────────


def test_codegen_folder_refuses_recurrent_models(tmp_path):
    """codegen_folder's main.c cannot pass the prev-spikes buffers yet; it must
    refuse rather than generate a harness that calls the kernel with too few
    arguments."""
    nir.write(str(tmp_path / "model.nir"), _golden_graph())
    (tmp_path / "input.csv").write_text("0,1,0,1\n1,0,1,0\n")
    with pytest.raises(NotImplementedError, match="recurrent"):
        snn_mlir.codegen_folder(tmp_path)
