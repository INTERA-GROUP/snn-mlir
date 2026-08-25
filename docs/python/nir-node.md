# Adding a NIR node type

## Why extend the parsers?

The frontend only knows the NIR node types it has parsers for — everything else raises a clear
"unsupported type" error. Extending coverage is how the project grows: new neuron models, new
connectivity primitives, or any node your framework emits. We **actively want** these
contributions, and the design keeps them small and self-contained.

The most valuable additions right now are the still-unmapped NIR primitives — `nir.Delay`,
`nir.Scale`, `nir.Threshold` — and any neuron variant your hardware needs. If you're unsure where to start,
[get in touch](../contributing.md) and we'll point you at the right node.

## How it's structured

`NODE_PARSERS` is the single registry mapping NIR node types to handler functions. All other
per-node behavior — quantization, MLIR emission, classification traits — lives on the
`NodeInfo` subclass itself, so adding a new NIR node type is three steps.

### 1. Create a `NodeInfo` subclass

Subclass the base that matches the node's role: `NeuronInfo` for a state-carrying
neuron, `SynapseInfo` for a weight-carrying layer, or `NodeInfo` directly for a
shape-only node (reshape, pooling) that is neither.

```python
from snn_mlir.nodes import NeuronInfo
from dataclasses import dataclass

@dataclass
class MyNeuronInfo(NeuronInfo):
    # NeuronInfo already carries `name` and `shape` and sets is_neuron = True;
    # add the node's own parameters and implement its abstract members.
    decay_int: int

    # Override quantize() if the node has quantizable parameters (no-op by
    # default). Called once per layer before MLIR emission in quantized mode.
    def quantize(self) -> None:
        ...

    def emit_mlir(self, input_var, is_last, quantize):
        # Return (list_of_mlir_lines, output_var_name)
        ...
```

The `is_synapse` / `is_neuron` traits are the classification vocabulary graph logic
branches on — but they are overridden *only* on the two role bases, so the way to set
them correctly is to extend the matching base rather than flip a flag by hand. The pure
topological walk (shape propagation) branches on the flags; consumers that then reach for
a role-specific member (a neuron's `state_size`, a synapse's `requantize`) narrow with an
`isinstance` check against the role base, which is why a neuron/synapse node must subclass
it. Get the base right and new node types are handled by existing machinery automatically.

### 2. Write a parser function

```python
import nir
def parse_mynode(node: nir.MyNode, name: str) -> MyNodeInfo:
    return MyNodeInfo(name=name, size=node.output_shape[0])
```

This is also where you **discretize** any continuous NIR parameters (see
[NIR mapping](nir-mapping.md)) and validate the dialect's assumptions.

### 3. Register it

```python
from snn_mlir.nodes import NODE_PARSERS
NODE_PARSERS[nir.MyNode] = parse_mynode
```

Add a roundtrip test in `test/Dialect/SNN/` for the new op and a Python unit test for the
parser, then open a PR — see [Contributing](../contributing.md).
