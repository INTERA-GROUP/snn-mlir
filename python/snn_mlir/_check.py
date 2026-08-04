# Copyright 2026 N Vision Systems And Technologies SL
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Non-throwing suitability check for a NIR graph.

``parse_graph`` answers "convert this model" and raises on the first thing it
cannot handle. That is the right behaviour for a compiler front-end and the
wrong one for a reviewer: it reports one problem, loses every other, and gives
the caller a string instead of the node the string is about.

``check`` answers the other question — *is this model supported, and if not,
where exactly?* — by running the same rules over every node independently and
collecting the results. Three properties make it usable as a library:

* **Total.** It does not raise for a graph, whatever the graph contains, and it
  terminates on cyclic edge sets (which the ``parse_graph`` walk does not).
* **Complete.** Every node is reported, not just the first failing one.
* **Anchored.** Findings name the node or edge they belong to, so a caller can
  paint them onto a rendering of the graph rather than printing a sentence.

The node-level rules are not reimplemented here: each node is handed to its real
parser from ``NODE_PARSERS`` and the exception it raises *is* the finding. There
is therefore no second copy of "v_reset must be 0" to drift out of step with the
parser that enforces it. Only the topology walk is restated, because it is
structural (successor counts) rather than semantic.
"""

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import nir

from .nodes import NODE_PARSERS

ERROR = "error"
WARNING = "warning"

#: NIR graph terminals. They have no parser and never will — ``parse_graph``
#: starts at ``input`` and stops at ``output`` without parsing either — so they
#: must not be reported as unsupported node types.
TERMINAL_TYPES = (nir.Input, nir.Output)


@dataclass(frozen=True)
class Finding:
    """One thing the front-end cannot handle, anchored to where it applies."""

    kind: str
    """Machine-readable slug: ``unsupported_type``, ``unsupported_parameter``,
    ``nonlinear_topology``, ``dead_end``, ``cycle``, ``missing_terminal``,
    ``unreachable``, ``parser_error``."""

    message: str
    """Human-readable sentence. For node-level findings this is verbatim the
    message the parser raised, so it matches what a failed conversion prints."""

    severity: str = ERROR
    """``error`` (will not compile) or ``warning`` (compiles, worth knowing)."""

    node: str | None = None
    """Name of the node this applies to, or ``None`` for whole-graph findings."""


@dataclass(frozen=True)
class NodeReport:
    """Per-node verdict."""

    name: str
    type: str
    """NIR node class name, e.g. ``CubaLIF``."""

    role: str
    """``terminal``, ``synapse``, ``neuron``, or ``unsupported``."""

    ok: bool
    findings: tuple[Finding, ...] = ()


@dataclass(frozen=True)
class Report:
    """The result of :func:`check`."""

    ok: bool
    """True when nothing of severity ``error`` was found."""

    nodes: tuple[NodeReport, ...] = ()
    """One entry per graph node, in the graph's own node order."""

    graph: tuple[Finding, ...] = field(default=())
    """Whole-graph findings (topology, terminals, reachability)."""

    order: tuple[str, ...] = ()
    """Node names along the ``input`` -> ``output`` path, in walk order. Empty
    when the walk could not complete. ``nodes`` is in the graph's own node order,
    which need not be the order data flows through it; this is that order, for
    callers laying the graph out or printing it."""

    @property
    def findings(self) -> list[Finding]:
        """Every finding, node-level first then whole-graph."""
        return [f for n in self.nodes for f in n.findings] + list(self.graph)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == ERROR]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == WARNING]

    def as_dict(self) -> dict:
        """A plain JSON-serializable dict, for tooling that is not Python."""
        return dataclasses.asdict(self) | {"ok": self.ok}


def _role(node: object, parsed: object | None) -> str:
    if isinstance(node, TERMINAL_TYPES):
        return "terminal"
    if parsed is None:
        return "unsupported"
    if getattr(parsed, "is_synapse", False):
        return "synapse"
    if getattr(parsed, "is_neuron", False):
        return "neuron"
    return "unknown"


def _check_node(name: str, node: object) -> NodeReport:
    """Run the node's own parser and turn whatever it raises into a finding."""
    if isinstance(node, TERMINAL_TYPES):
        return NodeReport(name=name, type=type(node).__name__, role="terminal", ok=True)

    parser = NODE_PARSERS.get(type(node))
    if parser is None:
        return NodeReport(
            name=name,
            type=type(node).__name__,
            role="unsupported",
            ok=False,
            findings=(
                Finding(
                    kind="unsupported_type",
                    message=f"Node '{name}' has unsupported type: {type(node).__name__}",
                    node=name,
                ),
            ),
        )

    try:
        parsed = parser(node, name)
    except (ValueError, NotImplementedError) as exc:
        # A deliberate rejection: the parser looked at the parameters and said no.
        return NodeReport(
            name=name,
            type=type(node).__name__,
            role="unknown",
            ok=False,
            findings=(Finding(kind="unsupported_parameter", message=str(exc), node=name),),
        )
    except Exception as exc:
        # Anything else is a defect in the parser (a missing NIR attribute, say),
        # not a statement about the model. Reported rather than raised — totality
        # is this function's contract — but under a distinct kind, so a package
        # bug is never presented to a user as "your model is unsupported".
        return NodeReport(
            name=name,
            type=type(node).__name__,
            role="unknown",
            ok=False,
            findings=(
                Finding(
                    kind="parser_error",
                    message=f"Node '{name}' ({type(node).__name__}) could not be parsed: "
                    f"{type(exc).__name__}: {exc}",
                    node=name,
                ),
            ),
        )

    return NodeReport(name=name, type=type(node).__name__, role=_role(node, parsed), ok=True)


def _check_topology(graph: nir.NIRGraph) -> tuple[tuple[Finding, ...], tuple[str, ...]]:
    """Restate ``parse_graph``'s walk, reporting instead of raising.

    Returns the findings and the walk order, which is a by-product of the same
    traversal and the order callers actually want to display the graph in.

    Adds a cycle guard the walk itself lacks: ``parse_graph`` would spin forever
    on a closed loop of single-successor edges, and a checker that hangs is worse
    than one that is wrong.
    """
    adj: dict[str, list[str]] = {}
    for src, dst in graph.edges:
        adj.setdefault(src, []).append(dst)

    findings: list[Finding] = []
    for terminal in ("input", "output"):
        if terminal not in graph.nodes:
            findings.append(
                Finding(
                    kind="missing_terminal",
                    message=f"Graph has no '{terminal}' node; the walk from 'input' to "
                    "'output' cannot be made.",
                ),
            )
    if findings:
        return tuple(findings), ()

    visited: list[str] = []
    seen: set[str] = set()
    current = "input"
    while current != "output":
        seen.add(current)
        visited.append(current)
        nexts = adj.get(current, [])
        if len(nexts) > 1:
            # Worded exactly as parse_graph words it, so the checker and a failed
            # conversion say the same sentence about the same graph.
            findings.append(
                Finding(
                    kind="nonlinear_topology",
                    message=f"Node '{current}' has {len(nexts)} successors — "
                    "only linear-chain graphs are supported.",
                    node=current,
                ),
            )
            return tuple(findings), ()
        if not nexts:
            findings.append(
                Finding(
                    kind="dead_end",
                    message=f"Node '{current}' has no successors — the graph never "
                    "reaches 'output'.",
                    node=current,
                ),
            )
            return tuple(findings), ()
        current = nexts[0]
        if current in seen:
            findings.append(
                Finding(
                    kind="cycle",
                    message=f"Edges form a cycle back to '{current}' — "
                    "only linear-chain graphs are supported.",
                    node=current,
                ),
            )
            return tuple(findings), ()

    # The walk completed, so "not visited" is now a meaningful statement. These
    # nodes are silently ignored by parse_graph: the model converts, but not all
    # of it does, which is worth saying out loud.
    #
    # Terminals are skipped, and not only the 'output' the walk stops at: NIRGraph
    # synthesizes an Input/Output pair around any disconnected component, so a
    # single stray node otherwise reports as three unreachable ones — two of which
    # the user never wrote.
    for name, node in graph.nodes.items():
        if name not in seen and not isinstance(node, TERMINAL_TYPES):
            findings.append(
                Finding(
                    kind="unreachable",
                    message=f"Node '{name}' is not on the path from 'input' to 'output' "
                    "and will be ignored.",
                    severity=WARNING,
                    node=name,
                ),
            )
    return tuple(findings), (*visited, "output")


def check(source: "nir.NIRGraph | str | Path") -> Report:
    """Report whether a NIR graph can be converted, and what blocks it.

    Every node is checked independently against its real parser and the topology
    is checked separately, so a graph whose nodes are all individually fine can
    still be reported as unsupported — a recurrent connection is exactly that
    case.

    Args:
        source: A nir.NIRGraph object, or a path to a .nir file.

    Returns:
        A :class:`Report`. ``report.ok`` is True when the model can be converted.

    Note:
        Never raises for a graph. Passing a *path* reads it with ``nir.read``,
        which may raise as usual if the file is missing or malformed.

    Example::

        import snn_mlir

        report = snn_mlir.check("network.nir")
        if not report.ok:
            for f in report.errors:
                print(f.node or "graph", f.message)

    """
    if isinstance(source, (str, Path)):
        source = nir.read(str(source))

    nodes = tuple(_check_node(name, node) for name, node in source.nodes.items())
    graph, order = _check_topology(source)
    ok = all(n.ok for n in nodes) and not any(f.severity == ERROR for f in graph)
    return Report(ok=ok, nodes=nodes, graph=graph, order=order)
