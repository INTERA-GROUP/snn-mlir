# Copyright 2026 N Vision Systems And Technologies SL
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""CLI surface tests: version, the `export` verb, and the not-yet-wired stubs."""

import nir
import numpy as np
import pytest
from snn_mlir._cli import main


@pytest.fixture
def nir_file(tmp_path, nir_linear_cubalif):
    """Write the conftest Linear->CubaLIF graph to a .nir on disk."""
    path = tmp_path / "network.nir"
    nir.write(path, nir_linear_cubalif)
    return path


@pytest.fixture
def model_folder(tmp_path, nir_linear_cubalif):
    """A codegen-ready folder: one network.nir (Linear 8->16) + a 3x8 input.csv."""
    nir.write(tmp_path / "network.nir", nir_linear_cubalif)
    rng = np.random.default_rng(0)
    np.savetxt(tmp_path / "input.csv", rng.integers(0, 2, (3, 8)), fmt="%d", delimiter=",")
    return tmp_path


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "snn-mlir" in capsys.readouterr().out


def test_no_command_errors(capsys):
    # A required subcommand is missing -> argparse exits 2.
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_export_writes_beside_input(nir_file):
    assert main(["export", str(nir_file)]) == 0
    out = nir_file.with_suffix(".mlir")
    assert out.is_file()
    assert "module" in out.read_text()


def test_export_output_flag(nir_file, tmp_path):
    out = tmp_path / "custom.mlir"
    assert main(["export", str(nir_file), "-o", str(out)]) == 0
    assert out.is_file()
    # default location must NOT be written when -o is given
    assert not nir_file.with_suffix(".mlir").exists()


def test_export_quantize(nir_file):
    assert main(["export", str(nir_file), "-q"]) == 0
    text = nir_file.with_suffix(".mlir").read_text()
    # quantized modules carry int8 weights
    assert "i8" in text


def test_export_missing_file(capsys):
    assert main(["export", "does_not_exist.nir"]) == 1
    assert "not found" in capsys.readouterr().err


def test_codegen_writes_build(model_folder):
    assert main(["codegen", str(model_folder)]) == 0
    build = model_folder / "build"
    for name in ("network.mlir", "snn_data.h", "main.c", "input.h"):
        assert (build / name).is_file(), name
    # n_steps comes from the 3 CSV rows; INPUT_SIZE from the Linear(8->16)
    header = (build / "snn_data.h").read_text()
    assert "#define N_STEPS      3" in header
    assert "#define INPUT_SIZE   8" in header
    assert "int8_t L0_input[3][8]" in (build / "input.h").read_text()


def test_codegen_quantize(model_folder):
    assert main(["codegen", str(model_folder), "-q"]) == 0
    # quantized driver feeds the kernel an int8 input descriptor
    assert "mk1d_i8" in (model_folder / "build" / "main.c").read_text()


def test_codegen_missing_input_csv(tmp_path, nir_linear_cubalif, capsys):
    nir.write(tmp_path / "network.nir", nir_linear_cubalif)
    assert main(["codegen", str(tmp_path)]) == 1
    assert "input.csv not found" in capsys.readouterr().err


def test_codegen_no_nir(tmp_path, capsys):
    (tmp_path / "input.csv").write_text("0,0\n")
    assert main(["codegen", str(tmp_path)]) == 1
    assert "no .nir file found" in capsys.readouterr().err


def test_codegen_width_mismatch(tmp_path, nir_linear_cubalif, capsys):
    nir.write(tmp_path / "network.nir", nir_linear_cubalif)
    # network expects 8 inputs; give it 5 columns
    np.savetxt(tmp_path / "input.csv", np.zeros((2, 5), dtype=int), fmt="%d", delimiter=",")
    assert main(["codegen", str(tmp_path)]) == 1
    assert "expects 8 inputs" in capsys.readouterr().err


def test_run_not_implemented(capsys):
    assert main(["run", "some_folder"]) == 1
    assert "not available yet" in capsys.readouterr().err
