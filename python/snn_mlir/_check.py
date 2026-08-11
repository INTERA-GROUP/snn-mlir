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

* **Total.** It does not raise for a graph, whatever the graph contains,
  cyclic edge sets included.
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

from . import _graph
from .nodes import NODE_PARSERS

ERROR = "error"
WARNING = "warning"
INFO = "info"

#: NIR graph terminals. They have no parser and never will — ``parse_graph``
#: starts at ``input`` and stops at ``output`` without parsing either — so they
#: must not be reported as unsupported node types.
TERMINAL_TYPES = (nir.Input, nir.Output)


@dataclass(frozen=True)
class Finding:
    """One thing the front-end cannot handle, anchored to where it applies."""

    kind: str
    """Machine-readable slug: ``unsupported_type``, ``unsupported_parameter``,
    ``nonlinear_topology``, ``unsupported_fan_in``, ``dead_end``, ``cycle``,
    ``recurrent_edge``, ``missing_terminal``, ``unreachable``, ``parser_error``."""

    message: str
    """Human-readable sentence. For node-level findings this is verbatim the
    message the parser raised, so it matches what a failed conversion prints."""

    severity: str = ERROR
    """``error`` (will not compile), ``warning`` (compiles, worth knowing), or
    ``info`` (a structural fact the conversion handles — a recurrent edge)."""

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


def _check_topology(
    graph: nir.NIRGraph,
    roles: dict[str, str],
) -> tuple[tuple[Finding, ...], tuple[str, ...]]:
    """Run ``parse_graph``'s own topology analysis, reporting instead of raising.

    Returns the findings and the forward execution order, which is a by-product
    of the same analysis and the order callers actually want to display the
    graph in. A canonical recurrence is not an error: the analysis breaks the
    neuron→synapse edge at the timestep boundary and reports it as an
    ``info``-severity ``recurrent_edge`` finding.

    Delegating to :func:`_graph.analyze_topology` (rather than restating the
    walk) is what keeps the two from drifting: the sentence this reports is the
    sentence a failed conversion raises. The analysis also terminates on cyclic
    edge sets the old walk would spin on.
    """
    topo = _graph.analyze_topology(list(graph.edges), roles)

    findings = [
        Finding(kind=i.kind, message=i.message, severity=i.severity, node=i.node)
        for i in topo.issues
    ]
    findings += [
        Finding(
            kind="recurrent_edge",
            message=f"Recurrent edge '{neuron}' -> '{synapse}': '{synapse}' reads "
            f"'{neuron}' spikes from the previous timestep.",
            severity=INFO,
            node=synapse,
        )
        for neuron, synapse in topo.recurrent_edges
    ]
    if any(f.severity == ERROR for f in findings):
        return tuple(findings), ()
    return tuple(findings), ("input", *topo.order, "output")


def check(source: "nir.NIRGraph | str | Path") -> Report:
    """Report whether a NIR graph can be converted, and what blocks it.

    Every node is checked independently against its real parser and the topology
    is checked separately, so a graph whose nodes are all individually fine can
    still be reported as unsupported — an unbreakable cycle is exactly that
    case. A canonical self-recurrence (neuron ⇄ recurrent synapse) is supported
    and reported as an ``info``-severity ``recurrent_edge`` finding, not an
    error.

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
    roles = {n.name: n.role for n in nodes}
    graph, order = _check_topology(source, roles)
    ok = all(n.ok for n in nodes) and not any(f.severity == ERROR for f in graph)
    return Report(ok=ok, nodes=nodes, graph=graph, order=order)
