// Copyright 2026 Sensing & Control Systems, S.L.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===------ SNNToLinalg.h - SNN to Linalg conversion --------===//
#ifndef SNN_CONVERSION_SNNTOLINALG_H
#define SNN_CONVERSION_SNNTOLINALG_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace snn {
  std::unique_ptr<mlir::Pass> createConvertSNNToLinalgPass();
  void registerConvertSNNToLinalgPass();
} // namespace mlir::snn

#endif // SNN_CONVERSION_SNNTOLINALG_H
