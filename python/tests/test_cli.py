# Copyright 2026 N Vision Systems And Technologies SL
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""CLI surface tests: version, the `export` verb, and the not-yet-wired stubs."""

import nir
import pytest
from snn_mlir._cli import main


@pytest.fixture
def nir_file(tmp_path, nir_linear_cubalif):
    """Write the conftest Linear->CubaLIF graph to a .nir on disk."""
    path = tmp_path / "network.nir"
    nir.write(path, nir_linear_cubalif)
    return path


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


def test_codegen_not_implemented(capsys):
    assert main(["codegen", "some_folder"]) == 1
    assert "not available yet" in capsys.readouterr().err


def test_run_not_implemented(capsys):
    assert main(["run", "some_folder"]) == 1
    assert "not available yet" in capsys.readouterr().err
