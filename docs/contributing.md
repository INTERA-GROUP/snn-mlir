# Contributing

Thanks for considering a contribution — snn-mlir is young and deliberately extensible, so
there's a lot of high-impact, well-scoped work to pick up, whether you come from the
neuromorphic side or the compiler side. New NIR nodes, new hardware backends, more dimensions,
better metrics, docs fixes: all welcome.

## Don't have code to contribute? Send us a NIR graph

The fastest way to help us broaden coverage is to **share a NIR graph we can test**. If your
framework exports to NIR but isn't in our [supported simulators](index.md#supported-simulators)
table — or uses a node we don't handle yet — send us the `.nir` file (or a small script that
produces it) and we'll test it, fix what's needed, and add it to the suite. No PR required.

## Submitting changes

A few light guidelines to keep things smooth:

- Run `uv run pre-commit install` once after cloning — the hooks apply ruff lint + formatting
  on every commit, so you don't have to think about style.
- Run `uv run pytest` before opening a PR; all Python unit tests should pass.
- Keep ops **type-polymorphic** — float and quantized must work through the same op.
- New ops need an `assemblyFormat` for human-readable `.mlir`, plus a roundtrip test in
  `test/Dialect/SNN/`.
- New NIR node types go in `python/snn_mlir/nodes/` with a matching `NODE_PARSERS` entry; put
  quantization in the class's `quantize()` method. See [Adding a NIR node type](python/nir-node.md).
- Follow MLIR naming conventions (`add_mlir_dialect_library`, `add_mlir_conversion_library`,
  `MLIR`-prefixed CMake targets).

Don't worry about getting every detail perfect on the first try — open the PR and we'll help
you polish it.

## Questions? Talk to us

Stuck, unsure where something fits, or want to sanity-check an idea before writing code? We're
genuinely happy to help — reach out any time:

- **Alex G. Gener** — [alejandro.garcia@intera-group.com](mailto:alejandro.garcia@intera-group.com)
- **Alvaro Rollon** — [alvaro.rollon@intera-group.com](mailto:alvaro.rollon@intera-group.com)
