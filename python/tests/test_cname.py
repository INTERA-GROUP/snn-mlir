# Copyright 2026 N Vision Systems And Technologies SL
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""C-identifier mangling: dotted NIR node names must yield compilable C.

The ABI is positional so names never cross it — but main.c's own variables and
snn_data.h's macros are built from node names, and NIR allows characters C does
not (``lif1.lif``). MLIR keeps the original names; only the C side mangles.
"""

import nir
import numpy as np
import pytest
import snn_mlir
from snn_mlir import c_identifier, ensure_unique_c_names


def test_c_identifier_mangling():
    assert c_identifier("lif1.lif") == "lif1_lif"
    assert c_identifier("fc1") == "fc1"
    assert c_identifier("a-b c.d") == "a_b_c_d"
    assert c_identifier("0layer") == "_0layer"
    assert c_identifier("") == "_"


def test_ensure_unique_c_names_raises_on_collision():
    class _Fake:
        def __init__(self, name):
            self.name = name
            self.c_name = c_identifier(name)

    ensure_unique_c_names([_Fake("a.b"), _Fake("c")])
    with pytest.raises(ValueError, match="collide"):
        ensure_unique_c_names([_Fake("a.b"), _Fake("a_b")])


def _dotted_graph() -> nir.NIRGraph:
    """Linear chain whose node names carry dots (the submodule-path pattern)."""
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([4])}),
            "block.fc": nir.Linear(weight=np.eye(4, dtype=np.float32) * 0.5),
            "block.lif": nir.LIF(
                tau=np.full(4, 0.2),
                r=np.full(4, 2.0),
                v_leak=np.zeros(4),
                v_threshold=np.ones(4),
                v_reset=np.zeros(4),
            ),
            "output": nir.Output(output_type={"output": np.array([4])}),
        },
        edges=[("input", "block.fc"), ("block.fc", "block.lif"), ("block.lif", "output")],
    )


def test_codegen_mangles_dotted_names(tmp_path):
    nir.write(str(tmp_path / "model.nir"), _dotted_graph())
    (tmp_path / "input.csv").write_text("0,1,0,1\n")
    build = snn_mlir.codegen_folder(tmp_path, quantize=True)

    main_c = (build / "main.c").read_text()
    header = (build / "snn_data.h").read_text()
    mlir = (build / "network.mlir").read_text()

    assert "voltage_block_lif" in main_c
    assert "block.lif" not in main_c
    assert "#define Lblock_lif_LIF_SIZE 4" in header
    # MLIR keeps the NIR names verbatim — dots are legal MLIR identifiers.
    assert "%voltage_block.lif" in mlir
