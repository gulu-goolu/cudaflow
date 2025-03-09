#pragma once

#include <memory>

#include "cudaflow/allocator.h"
#include "driver_types.h"

namespace cudaflow {
class ICudaflow : public std::enable_shared_from_this<ICudaflow> {
 public:
  ~ICudaflow() = default;

  virtual IAllocator* allocator() = 0;

  virtual void memcpy(cudaMemcpyKind kind, const void* src, void* dst) = 0;

  using ComputeFn = void (*)(cudaStream_t s, void* user_data);
  virtual void compute(ComputeFn fn, void* user_data) = 0;

  static std::shared_ptr<ICudaflow> create(int32_t device_ordinal);
};
}  // namespace cudaflow
