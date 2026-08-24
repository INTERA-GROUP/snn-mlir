# Copyright 2026 N Vision Systems And Technologies SL
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""``nir.Conv2d`` → ``snn.conv2d``.

The first N-D synapse: rank-3 ``(C, H, W)`` activations, rank-4 ``(OC, C, Kh,
Kw)`` weights. Unlike a dense layer, its output geometry depends on the input
spatial size, so the shape it produces is computed, not read off the weights.
"""

import nir
import numpy as np
import pytest
from snn_mlir.nodes import NODE_PARSERS
from snn_mlir.nodes.conv import ConvInfo, parse_conv2d


def _conv_node(C=2, OC=4, K=3, H=6, W=6, stride=1, padding=1, dilation=1, groups=1, bias=True):
    node = nir.Conv2d(
        input_shape=np.array([H, W]),
        weight=np.random.randn(OC, C, K, K).astype(np.float32),
        bias=np.zeros(OC, dtype=np.float32),
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    node.input_type = {"input": np.array([C, H, W])}
    if not bias:
        node.bias = None
    return node


# ── shape algebra ─────────────────────────────────────────────────────────────


def test_parse_reads_the_four_weight_dims():
    info = parse_conv2d(_conv_node(C=2, OC=16, K=5), "c0")
    assert isinstance(info, ConvInfo)
    assert info.in_channels == 2
    assert info.out_channels == 16
    assert info.kernel == (5, 5)
    assert info.weight_shape == (16, 2, 5, 5)


def test_in_and_out_shape_follow_the_conv_formula():
    # (34+2-5)/2 + 1 = 16, the nmnistcnn first conv.
    info = parse_conv2d(_conv_node(C=2, OC=16, K=5, H=34, W=34, stride=2, padding=1), "c0")
    assert info.in_shape == (2, 34, 34)
    assert info.out_shape == (16, 16, 16)


def test_unit_stride_same_padding_preserves_spatial():
    info = parse_conv2d(_conv_node(C=16, OC=8, K=3, H=16, W=16, stride=1, padding=1), "c0")
    assert info.out_shape == (8, 16, 16)


def test_adopt_in_shape_reflows_output_geometry():
    """A conv's output spatial size tracks whatever its predecessor produces."""
    info = parse_conv2d(_conv_node(C=2, OC=4, K=3, H=6, W=6, stride=1, padding=1), "c0")
    info.adopt_in_shape((2, 10, 12))
    assert info.in_shape == (2, 10, 12)
    assert info.out_shape == (4, 10, 12)


def test_non_square_stride_and_padding():
    node = _conv_node(C=1, OC=1, K=3, H=8, W=8)
    node.stride = np.array([2, 1])
    node.padding = np.array([1, 0])
    info = parse_conv2d(node, "c0")
    # H: (8+2-3)/2+1 = 4 ; W: (8+0-3)/1+1 = 6
    assert info.out_shape == (1, 4, 6)


# ── the loud guards (dilation / groups), same class as CubaLIF k != 1 ─────────


@pytest.mark.parametrize("dilation", [2, np.array([2, 2]), np.array([1, 2])])
def test_rejects_dilation(dilation):
    node = _conv_node()
    node.dilation = dilation
    with pytest.raises(ValueError, match="dilation"):
        parse_conv2d(node, "c0")


def test_rejects_groups():
    node = _conv_node()
    node.groups = 2
    with pytest.raises(ValueError, match="groups"):
        parse_conv2d(node, "c0")


# ── emission ──────────────────────────────────────────────────────────────────


def test_registered():
    assert NODE_PARSERS[nir.Conv2d] is parse_conv2d


def test_emits_snn_conv2d_with_stride_and_padding_attrs():
    info = parse_conv2d(_conv_node(C=2, OC=4, K=3, H=6, W=6, stride=2, padding=1), "c0")
    lines, out_var = info.emit_mlir("%input", is_last=False, quantize=False)
    text = "\n".join(lines)
    assert "snn.conv2d" in text
    assert "stride = array<i64: 2, 2>" in text
    assert "padding = array<i64: 1, 1>" in text
    # (6+2-3)/2 + 1 = 3
    assert "memref<2x6x6xf32>, memref<4x2x3x3xf32> -> memref<4x3x3xf32>" in text
    assert out_var == "%synapse_c0"


def test_weight_globals_are_rank4_and_bias_rank1():
    info = parse_conv2d(_conv_node(C=2, OC=4, K=3), "c0")
    globs = "\n".join(info.weight_globals(quantize=False))
    assert "@w_c0 : memref<4x2x3x3xf32>" in globs
    assert "@b_c0 : memref<4xf32>" in globs


def test_no_bias_emits_no_bias_operand():
    info = parse_conv2d(_conv_node(bias=False), "c0")
    assert info.bias is None
    lines, _ = info.emit_mlir("%input", is_last=False, quantize=False)
    assert "bias(" not in "\n".join(lines)


def test_quantized_path_is_not_implemented_yet():
    info = parse_conv2d(_conv_node(), "c0")
    with pytest.raises(NotImplementedError):
        info.emit_mlir("%input", is_last=False, quantize=True)


# ── the N-D C ABI: a conv entry makes the whole kernel signature rank-3 ────────


def _conv_if_graph(C=2, OC=4, K=3, H=6, W=6):
    conv = _conv_node(C=C, OC=OC, K=K, H=H, W=W, stride=1, padding=1)
    neuron = nir.IF(
        r=np.ones((OC, H, W)),
        v_threshold=np.ones((OC, H, W)),
        v_reset=np.zeros((OC, H, W)),
        input_type={"input": np.array([OC, H, W])},
    )
    g = nir.NIRGraph(
        nodes={
            "input": nir.Input(np.array([C, H, W])),
            "conv": conv,
            "if": neuron,
            "output": nir.Output(np.array([OC, H, W])),
        },
        edges=[("input", "conv"), ("conv", "if"), ("if", "output")],
    )
    g.infer_types()
    return g


def test_graph_input_shape_is_rank3_for_a_conv_entry():
    from snn_mlir import parse_graph

    graph = parse_graph(_conv_if_graph())
    assert graph.input_shape == (2, 6, 6)
    assert graph.input_size == 72  # product, back-compatible with dense consumers


def test_emitted_kernel_carries_rank3_state_and_io():
    from snn_mlir import parse_graph
    from snn_mlir._emit import generate_mlir

    mlir = generate_mlir(parse_graph(_conv_if_graph()), quantize=False)
    assert "%input : memref<2x6x6xf32>" in mlir
    assert "%voltage_if : memref<4x6x6xf32>" in mlir  # state is a feature map, not flat
    assert "%output : memref<4x6x6xf32>" in mlir


def test_main_c_descriptors_are_rank3(tmp_path):
    import snn_mlir

    nir.write(tmp_path / "model.nir", _conv_if_graph())
    np.savetxt(tmp_path / "input.csv", np.zeros((2, 72), dtype=int), fmt="%d", delimiter=",")
    build = snn_mlir.codegen_folder(tmp_path, quantize=False)
    main_c = (build / "main.c").read_text()
    assert "Memref3Df32" in main_c
    assert "mk3d_f32(input_buf, 2, 6, 6)" in main_c
    assert "mk3d_f32(voltage_if, 4, 6, 6)" in main_c
    assert "{d1*d2, d2, 1}" in main_c  # row-major strides
