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


# ── the C harness mirrors the ABI ─────────────────────────────────────────────


def _write_model_folder(tmp_path, n_steps=8, seed=7):
    """Golden graph + a Bernoulli spike input.csv; returns the input array."""
    rng = np.random.default_rng(seed)
    inputs = (rng.random((n_steps, 4)) < 0.5).astype(np.int8)
    nir.write(str(tmp_path / "model.nir"), _golden_graph())
    (tmp_path / "input.csv").write_text(
        "".join(",".join(str(int(v)) for v in row) + "\n" for row in inputs),
    )
    return inputs


def test_codegen_folder_harness_passes_prev_spikes(tmp_path):
    """main.c must own a zero-initialized prev-spikes buffer per recurrent
    neuron and pass it per the _emit positional ABI: right after that neuron's
    own state descriptors, before any later layer's."""
    _write_model_folder(tmp_path)
    build = snn_mlir.codegen_folder(tmp_path, quantize=True)
    main_c = (build / "main.c").read_text()

    assert "int8_t prev_spikes_lif1[Llif1_CUBALIF_SIZE] = {0};" in main_c
    call = main_c.split("_mlir_ciface_snn_forward_step(\n            ")[1].split(");")[0]
    assert call == (
        "&in_desc, &clif1_desc, &vlif1_desc, &prev_lif1_desc, &clif2_desc, &vlif2_desc, &out_desc"
    )


# ── host execution matches the numpy golden ───────────────────────────────────


def _q12_reference(graph, inputs: np.ndarray) -> np.ndarray:
    """Bit-exact numpy transcription of the quantized (Q12) kernel.

    Integer arithmetic end to end, so — unlike a float reference — accumulation
    order cannot perturb the result: the compiled binary must match exactly.
    Reads every quantization parameter off the already-quantized ``graph``
    (post rescale-insertion, so synapse w_scales are the shared ones).
    """
    fc1, lif1, w_rec, fc2, lif2 = (graph.nodes[n] for n in ("fc1", "lif1", "w_rec", "fc2", "lif2"))
    w = {s.name: s.int8_weights.astype(np.int64) for s in (fc1, w_rec, fc2)}
    shift = {s.name: 12 - s.w_scale for s in (fc1, w_rec, fc2)}

    prev = np.zeros(lif1.size, dtype=np.int64)
    c1 = v1 = np.zeros(lif1.size, dtype=np.int64)
    c2 = v2 = np.zeros(lif2.size, dtype=np.int64)
    out = []
    for x in inputs.astype(np.int64):

        def neuron(n, s, c, v):
            c = ((n.cur_decay_scaled * c) >> 12) + s
            v = ((n.vol_decay_scaled * v) >> 12) + c
            spike = (v > n.threshold_scaled).astype(np.int64)
            return c, np.where(spike > 0, 0, v), spike

        merged = (w["fc1"] @ x << shift["fc1"]) + (w["w_rec"] @ prev << shift["w_rec"])
        c1, v1, spk1 = neuron(lif1, merged, c1, v1)
        c2, v2, spk2 = neuron(lif2, w["fc2"] @ spk1 << shift["fc2"], c2, v2)
        prev = spk1
        out.append(spk2)
    return np.array(out, dtype=np.int64)


@pytest.mark.skipif(
    not snn_mlir.toolchain_available(),
    reason="host toolchain not resolvable (build snn-opt, or set SNN_OPT/MLIR_DIR/CC)",
)
def test_run_recurrent_quantized_matches_numpy(tmp_path):
    inputs = _write_model_folder(tmp_path)
    results = snn_mlir.run_folder(tmp_path, quantize=True)
    actual = np.loadtxt(results, delimiter=",", dtype=np.int64, ndmin=2)

    graph = snn_mlir.parse_graph(tmp_path / "model.nir")
    snn_mlir.quantize_layers(graph)
    from snn_mlir._graph import insert_rescale_nodes

    insert_rescale_nodes(graph)  # applies the shared fan-in w_scale in place
    expected = _q12_reference(graph, inputs)
    assert (actual == expected).all(), "quantized host run diverged from the Q12 reference"
    assert expected.any(), "degenerate golden: the reference never spikes"


@pytest.mark.skipif(
    not snn_mlir.toolchain_available(),
    reason="host toolchain not resolvable (build snn-opt, or set SNN_OPT/MLIR_DIR/CC)",
)
def test_run_recurrent_float_executes(tmp_path):
    """The float path compiles, runs, and emits one spike row per timestep.

    Bit-exactness against numpy is only asserted for the quantized path — float
    accumulation order is toolchain-defined, and a reference that must match it
    bit-for-bit would be flaky by construction.
    """
    inputs = _write_model_folder(tmp_path)
    results = snn_mlir.run_folder(tmp_path, quantize=False)
    actual = np.loadtxt(results, delimiter=",", dtype=np.int64, ndmin=2)
    assert actual.shape == (inputs.shape[0], 3)
    assert set(np.unique(actual)) <= {0, 1}
