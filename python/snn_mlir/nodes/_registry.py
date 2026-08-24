# Copyright 2026 N Vision Systems And Technologies SL
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from collections.abc import Callable

import nir

from ._base import NodeInfo
from .conv import parse_conv2d
from .conv1d import parse_conv1d
from .cubali import parse_cubali
from .cubalif import parse_cubalif
from .li import parse_i, parse_li
from .lif import parse_if, parse_lif
from .linear import parse_affine, parse_linear

# Maps NIR node type → parser(node, name) → NodeInfo.
# Add an entry here to support a new node type (e.g. Conv2d). Quantization lives
# on the NodeInfo subclass itself (NodeInfo.quantize), so this is the single
# registry needed to wire in a new node.
NODE_PARSERS: dict[type, Callable[..., NodeInfo]] = {
    nir.Linear: parse_linear,
    nir.Affine: parse_affine,
    nir.Conv2d: parse_conv2d,
    nir.Conv1d: parse_conv1d,
    nir.CubaLIF: parse_cubalif,
    nir.CubaLI: parse_cubali,
    nir.LIF: parse_lif,
    nir.LI: parse_li,
    # The non-leaky pair. They map onto the SAME two ops as their leaky
    # counterparts — a separate PARSER, not a separate op, because the only
    # difference is which fields NIR gives them (no tau, no v_leak) and the
    # dialect cannot see that difference. Exactly the parse_linear/parse_affine
    # pattern, where two NIR nodes share snn.linear.
    nir.IF: parse_if,
    nir.I: parse_i,
}
