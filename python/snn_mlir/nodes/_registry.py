# Copyright 2026 Sensing & Control Systems, S.L.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from collections.abc import Callable

import nir

from ._base import NodeInfo
from .cubali import parse_cubali
from .cubalif import parse_cubalif
from .li import parse_li
from .lif import parse_lif
from .linear import parse_affine, parse_linear

# Maps NIR node type → parser(node, name) → NodeInfo.
# Add an entry here to support a new node type (e.g. Conv2d). Quantization lives
# on the NodeInfo subclass itself (NodeInfo.quantize), so this is the single
# registry needed to wire in a new node.
NODE_PARSERS: dict[type, Callable[..., NodeInfo]] = {
    nir.Linear: parse_linear,
    nir.Affine: parse_affine,
    nir.CubaLIF: parse_cubalif,
    nir.CubaLI: parse_cubali,
    nir.LIF: parse_lif,
    nir.LI: parse_li,
}
