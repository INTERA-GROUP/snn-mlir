# Copyright 2026 N Vision Systems And Technologies SL
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""NIR graph walk: parse, order, and structure a graph for MLIR emission.

The walk accepts linear chains and the canonical SNN self-recurrence — a
neuron whose spikes feed a recurrent synapse that projects back onto the same
neuron. The recurrent neuron→synapse edge is broken at the timestep boundary
(the synapse reads the previous timestep's spikes from a state buffer), which
turns the graph into a DAG; anything still cyclic after that, and any other
branching, is rejected loudly.

``parse_graph`` raises on the first problem; ``_check.check`` reuses
``analyze_topology`` to report the same problems without raising.
"""

import math
import warnings
from dataclasses import dataclass, field

import nir

from .nodes import NODE_PARSERS, NodeInfo, RescaleInfo

#: The two by-name graph terminals. The walk starts at ``input`` and stops at
#: ``output``; neither is parsed into a layer.
TERMINALS = ("input", "output")


# ── topology analysis (shared with _check) ────────────────────────────────────


@dataclass(frozen=True)
class TopologyIssue:
    """One structural problem, in the same shape ``_check.Finding`` uses."""

    kind: str
    message: str
    node: str | None = None
    severity: str = "error"


@dataclass
class Topology:
    """Result of :func:`analyze_topology`."""

    order: list[str]
    """Non-terminal node names in forward (topological) execution order."""

    edges: list[tuple[str, str]]
    """Forward edges after cycle breaking, terminals included."""

    recurrent_edges: list[tuple[str, str]]
    """The broken neuron→synapse edges, in detection order."""

    issues: list[TopologyIssue] = field(default_factory=list)


def analyze_topology(
    edges: list[tuple[str, str]],
    roles: dict[str, str],
) -> Topology:
    """Order a NIR edge set, breaking canonical recurrences, reporting problems.

    Args:
        edges: The graph's ``(src, dst)`` edge list.
        roles: Node name → ``"terminal" | "synapse" | "neuron" | ...`` for every
               node in the graph. Cycle breaking needs to know which nodes are
               neurons and synapses; nodes with any other role make a cycle
               unbreakable.

    The analysis stops at the first error-severity issue (matching what a
    conversion would raise); warnings (unreachable nodes) do not stop it.
    """
    issues: list[TopologyIssue] = []
    for terminal in TERMINALS:
        if terminal not in roles:
            issues.append(
                TopologyIssue(
                    kind="missing_terminal",
                    message=f"Graph has no '{terminal}' node; the walk from 'input' to "
                    "'output' cannot be made.",
                ),
            )
    if issues:
        return Topology([], [], [], issues)

    adj: dict[str, list[str]] = {}
    for src, dst in edges:
        adj.setdefault(src, []).append(dst)

    # Reachability is decided on the original graph, before any edge is broken:
    # a recurrent synapse is reached through the very edge that is later broken.
    reach: set[str] = set()
    stack = ["input"]
    while stack:
        node = stack.pop()
        if node in reach:
            continue
        reach.add(node)
        stack.extend(adj.get(node, []))

    # ── cycle breaking ────────────────────────────────────────────────────────
    # The one supported cycle is the canonical SNN self-recurrence: a two-node
    # loop neuron ⇄ synapse. Its neuron→synapse edge is broken (the synapse
    # reads the previous timestep's spikes), never the synapse→neuron edge —
    # that direction keeps the state buffer spike-typed. Anything else — self
    # loops, longer cycles, cycles through unparseable nodes — is rejected.
    recurrent: list[tuple[str, str]] = []
    while True:
        cycle = _find_cycle(adj, reach)
        if cycle is None:
            break
        breakable = [
            (u, v) for u, v in cycle if roles.get(u) == "neuron" and roles.get(v) == "synapse"
        ]
        if len(cycle) == 2 and len(breakable) == 1:
            u, v = breakable[0]
            adj[u].remove(v)
            recurrent.append((u, v))
        else:
            path = " -> ".join([cycle[0][0]] + [v for _, v in cycle])
            issues.append(
                TopologyIssue(
                    kind="cycle",
                    message=f"Edges form a cycle ({path}) that is not a neuron->synapse "
                    "self-recurrence — unsupported topology.",
                    node=cycle[0][0],
                ),
            )
            return Topology([], [], recurrent, issues)

    # ── successor / predecessor rules (on the broken graph) ───────────────────
    preds: dict[str, list[str]] = {}
    for src, succs in adj.items():
        if src not in reach:
            continue
        for dst in succs:
            preds.setdefault(dst, []).append(src)

    for name in roles:
        if name == "output" or name not in reach:
            continue
        succs = adj.get(name, [])
        if len(succs) > 1:
            issues.append(
                TopologyIssue(
                    kind="nonlinear_topology",
                    message=f"Node '{name}' has {len(succs)} successors — "
                    "only linear-chain graphs are supported.",
                    node=name,
                ),
            )
            return Topology([], [], recurrent, issues)
        if not succs:
            issues.append(
                TopologyIssue(
                    kind="dead_end",
                    message=f"Node '{name}' has no successors — the graph never reaches 'output'.",
                    node=name,
                ),
            )
            return Topology([], [], recurrent, issues)

    # Fan-in is the merge form: it is only supported at a neuron, and every
    # branch feeding it must be a synapse (each contributes one weighted term
    # to the elementwise sum in front of the neuron).
    for name, pred_list in preds.items():
        if name not in reach or len(pred_list) < 2:
            continue
        if roles.get(name) != "neuron" or any(roles.get(p) != "synapse" for p in pred_list):
            issues.append(
                TopologyIssue(
                    kind="unsupported_fan_in",
                    message=f"Node '{name}' has {len(pred_list)} predecessors — "
                    "fan-in is only supported at a neuron fed by synapse outputs.",
                    node=name,
                ),
            )
            return Topology([], [], recurrent, issues)

    # A recurrent synapse's one input is the state buffer (the broken edge);
    # a forward predecessor on top of that would be silently dropped by the
    # emitter, so it is rejected here instead.
    for _, synapse in recurrent:
        if preds.get(synapse):
            issues.append(
                TopologyIssue(
                    kind="unsupported_fan_in",
                    message=f"Recurrent synapse '{synapse}' also has forward "
                    "predecessors — unsupported topology.",
                    node=synapse,
                ),
            )
            return Topology([], [], recurrent, issues)

    # ── Kahn topological sort ─────────────────────────────────────────────────
    # Deterministic tie-break: recurrent synapses first (they read state and
    # start the timestep), then the graph's own node order. Linear chains never
    # tie, so their order is exactly the old walk order.
    nonterm = [n for n in roles if n in reach and n not in TERMINALS]
    position = {n: i for i, n in enumerate(nonterm)}
    rec_targets = {v for _, v in recurrent}
    indegree = {n: sum(1 for p in preds.get(n, []) if p not in TERMINALS) for n in nonterm}
    ready = [n for n in nonterm if indegree[n] == 0]
    order: list[str] = []
    while ready:
        ready.sort(key=lambda n: (0 if n in rec_targets else 1, position[n]))
        node = ready.pop(0)
        order.append(node)
        for succ in adj.get(node, []):
            if succ in TERMINALS:
                continue
            indegree[succ] -= 1
            if indegree[succ] == 0:
                ready.append(succ)
    if len(order) != len(nonterm):  # pragma: no cover — cycles were all broken
        issues.append(
            TopologyIssue(
                kind="cycle",
                message="Edges form a cycle — unsupported topology.",
            ),
        )
        return Topology([], [], recurrent, issues)

    # The walk completed, so "not reached" is now a meaningful statement: these
    # nodes are silently ignored by parse_graph — the model converts, but not
    # all of it does. Terminals are skipped (NIRGraph synthesizes an
    # Input/Output pair around any disconnected component).
    for name, role in roles.items():
        if name not in reach and role != "terminal":
            issues.append(
                TopologyIssue(
                    kind="unreachable",
                    message=f"Node '{name}' is not on the path from 'input' to 'output' "
                    "and will be ignored.",
                    node=name,
                    severity="warning",
                ),
            )

    recurrent_set = set(recurrent)
    kept_edges = [e for e in edges if e[0] in reach and e not in recurrent_set]
    return Topology(order, kept_edges, recurrent, issues)


def _find_cycle(
    adj: dict[str, list[str]],
    reach: set[str],
) -> list[tuple[str, str]] | None:
    """Return one cycle (as its edge list) reachable from 'input', or None."""
    GRAY, BLACK = 1, 2
    color: dict[str, int] = {}
    path: list[str] = []

    def dfs(u: str) -> list[tuple[str, str]] | None:
        color[u] = GRAY
        path.append(u)
        for v in adj.get(u, []):
            if v not in reach:
                continue
            c = color.get(v)
            if c == GRAY:
                nodes = path[path.index(v) :]
                return [(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)] + [(u, v)]
            if c is None:
                found = dfs(v)
                if found:
                    return found
        color[u] = BLACK
        path.pop()
        return None

    return dfs("input")


# ── the parsed graph ──────────────────────────────────────────────────────────


@dataclass
class GraphInfo:
    """A parsed NIR graph: layers plus the structure connecting them.

    Iterating (or indexing) a ``GraphInfo`` yields the forward-path layers in
    execution order, so code written against the old ``list[NodeInfo]`` return
    of ``parse_graph`` keeps working unchanged.
    """

    nodes: dict[str, NodeInfo]
    """Parsed layers, keyed by NIR node name (terminals excluded)."""

    order: list[str]
    """Node names in forward (topological) execution order."""

    edges: list[tuple[str, str]]
    """Forward edges after cycle breaking, including the ``input``/``output``
    terminals."""

    recurrent_edges: list[tuple[str, str]] = field(default_factory=list)
    """Broken neuron→synapse edges: for each ``(neuron, synapse)``, the synapse
    reads the neuron's previous-timestep spikes from a state buffer."""

    @property
    def layers(self) -> list[NodeInfo]:
        """The forward-path layers in execution order."""
        return [self.nodes[name] for name in self.order]

    @property
    def _entry_synapse(self) -> NodeInfo:
        """The synapse the network input feeds.

        The entry node (successor of ``input``) defines it — NOT the first
        layer in ``order``, which for a recurrent graph is the recurrent
        synapse. A leading non-synapse falls back to the first synapse in
        order, as the old list-based consumers did.
        """
        entry = next((dst for src, dst in self.edges if src == "input"), None)
        entry_layer = self.nodes.get(entry)
        if entry_layer is None or not entry_layer.is_synapse:
            entry_layer = next((layer for layer in self if layer.is_synapse), None)
        if entry_layer is None:
            raise ValueError("No synapse (weight) layer found in graph.")
        return entry_layer

    @property
    def input_shape(self) -> tuple[int, ...]:
        """The network's input shape — rank-1 ``(N,)`` for a dense entry,
        rank-3 ``(C, H, W)`` for a conv entry. The entry synapse reports it.
        """
        return self._entry_synapse.in_shape

    @property
    def input_size(self) -> int:
        """The network's flat input width (product of :attr:`input_shape`)."""
        return math.prod(self.input_shape)

    @property
    def output_size(self) -> int:
        """The network's output width: the last neuron's state size."""
        last_neuron = next((layer for layer in reversed(self.layers) if layer.is_neuron), None)
        if last_neuron is None:
            raise ValueError("No neuron layer found in graph.")
        return last_neuron.state_size

    def predecessors(self, name: str) -> list[str]:
        return [src for src, dst in self.edges if dst == name]

    def successors(self, name: str) -> list[str]:
        return [dst for src, dst in self.edges if src == name]

    # list-of-layers compatibility
    def __iter__(self):
        return iter(self.layers)

    def __len__(self) -> int:
        return len(self.order)

    def __getitem__(self, index):
        return self.layers[index]


def as_graph_info(layers: "list[NodeInfo] | GraphInfo") -> GraphInfo:
    """Wrap a plain layer list as a linear-chain :class:`GraphInfo`."""
    if isinstance(layers, GraphInfo):
        return layers
    nodes: dict[str, NodeInfo] = {}
    order: list[str] = []
    edges: list[tuple[str, str]] = []
    prev = "input"
    for i, layer in enumerate(layers):
        key = layer.name
        if key in nodes or key in TERMINALS:
            key = f"{key}#{i}"
        nodes[key] = layer
        order.append(key)
        edges.append((prev, key))
        prev = key
    edges.append((prev, "output"))
    return GraphInfo(nodes=nodes, order=order, edges=edges)


# ── parsing ───────────────────────────────────────────────────────────────────


def _declared_shape(node: object, key: str) -> tuple[int, ...] | None:
    """A NIR node's own declared ``input``/``output`` shape, or None if absent."""
    entry = (getattr(node, f"{key}_type", None) or {}).get(key)
    return None if entry is None else tuple(int(d) for d in entry)


def _propagate_shapes(
    graph: nir.NIRGraph,
    nodes: dict[str, NodeInfo],
    topo: Topology,
) -> None:
    """Flow shapes along the chain, warning wherever NIR disagrees with us.

    Every parser reads its layer's shape from that node's own NIR
    ``input_type``, which makes each layer self-consistent but says nothing
    about whether consecutive layers actually fit together. This walk answers
    that: it starts at the ``input`` terminal, hands each layer the shape its
    predecessor really produces, and lets the layer decide what it emits.

    **The propagated shape wins.** NIR's declared types are a cross-check, not
    the authority, because NIR's own inference is demonstrably not reliable —
    ``Conv2d.__post_init__`` uses the kernel *height* for both spatial
    dimensions, so a non-square kernel gets an output shape that is simply
    wrong. Trusting the file there would bake that bug into the emitted memref
    types. Disagreements are reported rather than raised: the shape we compute
    is the one the arithmetic requires, and a stale ``output_type`` in a file is
    the model's problem, not a reason to refuse it.
    """
    input_shape = _declared_shape(graph.nodes.get("input"), "output")
    neuron_of_recurrent = {syn: neuron for neuron, syn in topo.recurrent_edges}
    produced: dict[str, tuple[int, ...] | None] = {}

    for name in topo.order:
        layer = nodes[name]

        # Where this layer's input comes from. A recurrent synapse has no
        # forward predecessor — its input is the state buffer, which holds the
        # spikes of the neuron the broken edge came from.
        if name in neuron_of_recurrent:
            incoming = [nodes[neuron_of_recurrent[name]].out_shape]
        else:
            incoming = [
                input_shape if src == "input" else produced.get(src)
                for src, dst in topo.edges
                if dst == name and (src == "input" or src in produced)
            ]
        incoming = [shape for shape in incoming if shape is not None]

        if incoming:
            arriving = incoming[0]
            for other in incoming[1:]:
                if other != arriving:
                    warnings.warn(
                        f"Node '{name}': fan-in branches arrive with different shapes "
                        f"({arriving} and {other}); the elementwise merge in front of "
                        f"a neuron requires one shape.",
                        UserWarning,
                        stacklevel=3,
                    )
            declared_in = layer.in_shape
            if declared_in is not None and tuple(declared_in) != arriving:
                warnings.warn(
                    f"Node '{name}': its predecessor produces {arriving}, but the node "
                    f"declares an input shape of {tuple(declared_in)}. Using {arriving} "
                    f"— if the node cannot absorb it, the emitted MLIR will not verify.",
                    UserWarning,
                    stacklevel=3,
                )
            layer.adopt_in_shape(arriving)

        produced[name] = layer.out_shape

        declared_out = _declared_shape(graph.nodes.get(name), "output")
        if declared_out is not None and produced[name] is not None:
            if tuple(produced[name]) != declared_out:
                warnings.warn(
                    f"Node '{name}': NIR declares output_type {declared_out}, but the "
                    f"layer produces {tuple(produced[name])}. Using the computed shape "
                    f"— NIR's own shape inference is not authoritative here.",
                    UserWarning,
                    stacklevel=3,
                )


def parse_graph(graph: nir.NIRGraph) -> GraphInfo:
    """Parse a NIR graph into layers ordered for one forward timestep.

    Accepts linear chains and the canonical self-recurrence (see the module
    docstring); raises ``ValueError`` / ``NotImplementedError`` on the first
    unsupported node or structure.
    """
    adj: dict[str, list[str]] = {}
    for src, dst in graph.edges:
        adj.setdefault(src, []).append(dst)

    reach: set[str] = set()
    stack = ["input"]
    while stack:
        node = stack.pop()
        if node in reach:
            continue
        reach.add(node)
        stack.extend(adj.get(node, []))

    nodes: dict[str, NodeInfo] = {}
    roles: dict[str, str] = {}
    for name, node in graph.nodes.items():
        if name in TERMINALS:
            roles[name] = "terminal"
            continue
        if name not in reach:
            continue
        parser = NODE_PARSERS.get(type(node))
        if parser is None:
            raise NotImplementedError(
                f"Node '{name}' has unsupported type: {type(node).__name__}",
            )
        layer = parser(node, name)
        nodes[name] = layer
        roles[name] = "synapse" if layer.is_synapse else "neuron" if layer.is_neuron else "unknown"

    topo = analyze_topology(list(graph.edges), roles)
    for issue in topo.issues:
        if issue.severity == "error":
            raise ValueError(issue.message)

    _propagate_shapes(graph, nodes, topo)

    ordered_nodes = {name: nodes[name] for name in topo.order}
    return GraphInfo(
        nodes=ordered_nodes,
        order=topo.order,
        edges=topo.edges,
        recurrent_edges=topo.recurrent_edges,
    )


def quantize_layers(layers: "list[NodeInfo] | GraphInfo") -> None:
    """Compute each layer's quantization parameters in-place."""
    for layer in layers:
        layer.quantize()


# ── rescale insertion (quantized mode) ────────────────────────────────────────


def insert_rescale_nodes(layers: "list[NodeInfo] | GraphInfo") -> GraphInfo:
    """Insert a synthetic :class:`RescaleInfo` on every synapse→neuron edge.

    Edge-based: a neuron fed by two synapses (the recurrent merge) gets one
    rescale per incoming edge. Synapses that feed the same neuron are first
    clamped to a shared ``w_scale`` (the minimum of the candidates) so every
    branch of the merge arrives in the same Q-format — since the rescale is a
    left shift and left shifts distribute over addition, any backend may then
    fuse the merge without changing the result.

    Accepts the old plain layer list (wrapped as a linear chain) or a
    :class:`GraphInfo`; returns a new :class:`GraphInfo` with the rescale
    nodes spliced into ``nodes``/``order``/``edges``.
    """
    graph = as_graph_info(layers)
    _share_fan_in_w_scales(graph)

    nodes = dict(graph.nodes)
    order = list(graph.order)
    edges: list[tuple[str, str]] = []
    for src, dst in graph.edges:
        syn = graph.nodes.get(src)
        neuron = graph.nodes.get(dst)
        if syn is None or neuron is None or not syn.is_synapse or not neuron.is_neuron:
            edges.append((src, dst))
            continue
        rescale = RescaleInfo(
            name=syn.name,
            shape=syn.out_shape,
            _w_scale=syn.w_scale,
            _d_scale=neuron.d_scale,
        )
        key = f"{src}/rescale"
        while key in nodes:
            key += "'"
        nodes[key] = rescale
        order.insert(order.index(src) + 1, key)
        edges.extend([(src, key), (key, dst)])
    return GraphInfo(
        nodes=nodes,
        order=order,
        edges=edges,
        recurrent_edges=list(graph.recurrent_edges),
    )


def _share_fan_in_w_scales(graph: GraphInfo) -> None:
    """Clamp synapses feeding the same neuron to one shared ``w_scale``.

    The shared scale is the minimum of the candidates — the fan-in branches
    lose no representable range, only the finer-scaled branch is re-rounded.
    Requantization recomputes from the float weights, so this is idempotent.
    """
    for name, layer in graph.nodes.items():
        if not layer.is_neuron:
            continue
        synapses = [
            graph.nodes[p]
            for p in graph.predecessors(name)
            if p in graph.nodes and graph.nodes[p].is_synapse
        ]
        if len(synapses) < 2:
            continue
        scales = [s.w_scale for s in synapses]
        if any(scale is None for scale in scales):
            continue  # not quantized — nothing to share
        shared = min(scales)
        for synapse in synapses:
            if synapse.w_scale != shared:
                synapse.requantize(shared)
