# Copyright 2026 N Vision Systems And Technologies SL
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""C identifiers for NIR node names.

NIR allows characters in node names that C identifiers do not — dotted names
like ``lif1.lif`` are common in real exports (a submodule path). MLIR
identifiers accept them, so MLIR emission uses the node name as-is; every C
code generator instead derives its variable and macro names from
:attr:`~snn_mlir.nodes.NodeInfo.c_name`, which is ``c_identifier(name)``.

Mangling is not injective (``a.b`` and ``a_b`` both become ``a_b``), so any
generator producing one C translation unit from a graph must call
:func:`ensure_unique_c_names` first and let a collision fail loudly rather
than emit two variables with the same name.
"""

import re


def c_identifier(name: str) -> str:
    """Mangle ``name`` into a valid C identifier (deterministic).

    Every character outside ``[A-Za-z0-9_]`` becomes ``_``; a leading digit
    (or an empty name) gets a ``_`` prefix.
    """
    mangled = re.sub(r"[^0-9A-Za-z_]", "_", name)
    if not mangled or mangled[0].isdigit():
        mangled = "_" + mangled
    return mangled


def ensure_unique_c_names(layers) -> None:
    """Raise ``ValueError`` if two layers' names mangle to the same C name.

    ``layers`` is any iterable of :class:`~snn_mlir.nodes.NodeInfo` (a
    ``GraphInfo`` works — iterating yields the forward-path layers).
    """
    by_c_name: dict[str, list[str]] = {}
    for layer in layers:
        by_c_name.setdefault(layer.c_name, []).append(layer.name)
    collisions = {c: names for c, names in by_c_name.items() if len(names) > 1}
    if collisions:
        detail = "; ".join(
            f"{', '.join(repr(n) for n in names)} -> '{c}'" for c, names in collisions.items()
        )
        raise ValueError(
            f"node names collide after C-identifier mangling ({detail}); "
            "rename the nodes so their C names are distinct"
        )
