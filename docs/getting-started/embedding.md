# Using the dialect in your own project

If you already have a larger MLIR-based project — your own compiler, an accelerator backend,
or a research toolchain — you don't have to adopt snn-mlir wholesale. The dialect is packaged
as ordinary CMake targets, so you can pull it in as a subdirectory or git submodule and link
only what you need: the dialect alone, or the dialect plus the reference CPU lowering.

## Consume the CMake targets

Add this repo as a subdirectory (or git submodule) and link the targets:

```cmake
add_subdirectory(third_party/snn-mlir)

target_include_directories(MyPass PRIVATE
  ${CMAKE_SOURCE_DIR}/third_party/snn-mlir/include
  ${CMAKE_BINARY_DIR}/third_party/snn-mlir/include
)

target_link_libraries(MyPass
  MLIRSNN          # dialect library
  MLIRSNNToLinalg  # CPU lowering pass (optional)
)
```

In your pass source:

```cpp
#include "SNN/SNNOps.h"
#include "SNN/Conversion/SNNToLinalg.h"  // if using the CPU lowering
```

From here you can write your own passes against the `snn` ops, or add a new lowering for your
target — see [Implementing a lowering pass](../dialect/lowering-pass.md).
