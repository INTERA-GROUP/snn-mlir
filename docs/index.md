# SNN-MLIR

[![CI](https://github.com/INTERA-GROUP/snn-mlir/actions/workflows/ci.yml/badge.svg)](https://github.com/INTERA-GROUP/snn-mlir/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/snn-mlir/badge/?version=latest)](https://snn-mlir.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/snn-mlir)](https://pypi.org/project/snn-mlir/)
[![arXiv](https://img.shields.io/badge/arXiv-2606.09213-b31b1b.svg)](https://arxiv.org/abs/2606.09213)
[![Collaboration Network](https://img.shields.io/badge/Collaboration_Network-Open_Neuromorphic-blue)](https://open-neuromorphic.org/)

![snn-mlir compilation flow: NIR → SNN dialect MLIR → LLVM IR → executable](assets/snn-mlir_flow.png)

An out-of-tree [MLIR](https://mlir.llvm.org/) dialect for Spiking Neural Networks (SNNs),
compatible with the [NIR (Neuromorphic Intermediate Representation)](https://neuroir.org/) standard.

The dialect provides type-polymorphic operations that work with both `f32` (float) and
quantized (`i8`/`i32`) types, enabling a single IR to target both simulation and
hardware-optimized deployments. A reference CPU lowering (`SNNToLinalg`) converts SNN ops to
standard `linalg`/`arith` operations that any MLIR-based backend can consume.

A companion Python package (`snn-mlir`, available on [PyPI](https://pypi.org/project/snn-mlir/))
reads any NIR file and emits SNN dialect MLIR text, together with the C sources of a reference
CPU runtime. Its `snn-mlir` command takes a trained network from `.nir` to a running binary
without writing a line of Python:

```bash
snn-mlir export  model.nir      # → model.mlir          (SNN dialect IR)
snn-mlir codegen my_model/      # → my_model/build/     (MLIR + C sources)
snn-mlir run     my_model/      # → build/results.csv   (compiled and executed)
```

See the [Quick start](getting-started/quickstart.md).

---

## Supported NIR nodes

`snn-mlir` consumes [NIR](https://neuroir.org/), so it is **framework-agnostic**: any simulator
that exports NIR can feed it. What matters is not the simulator but the **node types** the model
uses. The table below maps every [NIR primitive](https://neuroir.org/docs/primitives/) to its
support status. See [NIR mapping](python/nir-mapping.md) for the details of each.

| [NIR primitive](https://neuroir.org/docs/primitives/) | Supported | Maps to | Notes |
|---|:---:|---|---|
| `Input` / `Output` | ✅ | — | Graph entry/exit, handled implicitly |
| `Linear` | ✅ | `snn.linear` | No bias |
| `Affine` | ✅ | `snn.linear` | Bias as second operand |
| `Conv2d` | ✅ | `snn.conv2d` | Float + quantized |
| `Conv1d` | ✅ | `snn.conv1d` | Float + quantized |
| `SumPool2d` | ✅ | `snn.sumpool2d` | Float + quantized |
| `AvgPool2d` | ✅ | `snn.avgpool2d` | Float + quantized |
| `Flatten` | ✅ | — | Structural reshape |
| `LIF` | ✅ | `snn.lif` | Leaky integrate-and-fire |
| `LI` | ✅ | `snn.li` | Leaky integrator (no threshold) |
| `IF` | ✅ | `snn.lif` | Non-leaky (`decay = 1`, requires `r = 1`) |
| `I` | ✅ | `snn.li` | Non-leaky integrator |
| `CubaLIF` | ✅ | `snn.cubalif` | Current-based LIF (two states) |
| `CubaLI` | ✅ | `snn.cubali` | Current-based leaky integrator |
| `Delay` / `Scale` / `Threshold` | — | — | Not yet mapped |

!!! note "A node your model needs isn't mapped yet?"
    Adding one is a contained task — see [Adding a NIR node type](python/nir-node.md). If your
    framework writes a NIR graph `snn-mlir` doesn't yet handle, you are very welcome to **send us
    the NIR graph** and we'll take a look. See [Contributing](contributing.md).

### Example models, and where they come from

The shipped [examples](examples/snn-oxford.md) are trained in four different simulators — the
same models feed the same pipeline, which is exactly the point of consuming NIR:

| Example | Simulator | What it exercises |
|---|---|---|
| [SNN Oxford](examples/snn-oxford.md) | [LAVA-DL](https://github.com/lava-nc/lava-dl) | A plain feedforward chain of `Linear` synapses and `CubaLIF` neurons — the simplest end-to-end path |
| [SNNTorch](examples/snntorch.md) | [snnTorch](https://github.com/jeshraghian/snntorch/) | A mix of `Linear`/`Affine` (bias and no-bias) with both `LIF` and `CubaLIF` neurons in one graph |
| [BrailleNN](examples/brailernn.md) | [Norse](https://github.com/norse/norse) | Canonical SNN self-recurrence — a `CubaLIF` layer feeding a recurrent synapse back onto itself |
| [N-MNIST CNN](examples/nmnistcnn.md) | [Sinabs](https://sinabs.readthedocs.io/) | Convolutional spiking vision: `Conv2d`, `SumPool2d`, `Flatten` and non-leaky `IF` neurons |

---

## Which path is for you?

snn-mlir sits at the boundary between the neuromorphic and compiler worlds, so there are three
natural entry points — described in more detail in
[How it works](getting-started/how-it-works.md#three-ways-to-use-it):

- **Just want to run a model?** Start at the [Quick start](getting-started/quickstart.md). Point
  the CLI at a folder with your `.nir` and an input, and get compilable C — or a finished run —
  back out.
- **Coming from SNNs / neuromorphics?** You'll most likely care about the
  [Python package](python/nir-mapping.md) and the translation from **NIR to MLIR** — exporting
  your trained network and getting a portable, embedded-ready representation out.
- **Coming from compilers / embedded systems?** You'll most likely care about the
  [SNN MLIR dialect](dialect/overview.md) and [lowering it to your hardware](dialect/lowering-pass.md) —
  bringing your own backend (CPU, FPGA, ASIC) under a shared IR.

---

## Questions, doubts, or bugs?

We're happy to help. Reach out to the maintainers any time:

- **Alex G. Gener** — [alejandro.garcia@intera-group.com](mailto:alejandro.garcia@intera-group.com)
- **Alvaro Rollon** — [alvaro.rollon@intera-group.com](mailto:alvaro.rollon@intera-group.com)

## Citation

A companion paper describing snn-mlir is published on arXiv. If you use snn-mlir in your research, please cite the white paper directly:

```bibtex
@misc{gener2026snnmlirmlirdialectcompiling,
      title={SNN-MLIR: An MLIR Dialect for Compiling Neuromorphic SNNs from NIR to Bare-Metal C}, 
      author={Alejandro García Gener and Alvaro Rollón de Pinedo},
      year={2026},
      eprint={2606.09213},
      archivePrefix={arXiv},
      primaryClass={cs.PL},
      url={https://arxiv.org/abs/2606.09213}, 
}
```

## License

Apache License 2.0 WITH LLVM-exception — see the `LICENSE` file in the repository.
