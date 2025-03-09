#include <cstdint>
#include <memory>

#include "absl/strings/str_format.h"
#include "cuda_runtime_api.h"
#include "cudaflow/allocator.h"

namespace cudaflow {
namespace {
class DeviceMemoryAllocator : public IAllocator {
 public:
  DeviceMemoryAllocator(int32_t device_ordinal) : device_ordinal_(device_ordinal) {
    cudaDeviceProp props = {};
    cudaGetDeviceProperties(&props, device_ordinal);
    memory_alignment_ = props.textureAlignment;
  }

  uint32_t memory_alignment() const override { return memory_alignment_; }

  absl::StatusOr<Memory> allocate(size_t size) override {
    void* ptr = nullptr;
    cudaError_t malloc_ret = cudaMalloc(&ptr, size);
    if (malloc_ret != cudaSuccess) {
      return absl::UnknownError(
          absl::StrFormat("fail to allocate memory, ret_code: %s", cudaGetErrorName(malloc_ret)));
    }

    return Memory{.addr = ptr};
  }

  absl::Status deallocate(Memory memory) override {
    cudaError_t ret = cudaSetDevice(device_ordinal_);
    if (ret != cudaSuccess) {
      return absl::UnknownError(absl::StrFormat("fail to set device, device: %d, err_msg: %s",
                                                device_ordinal_, cudaGetErrorName(ret)));
    }

    cudaError_t free_ret = cudaFree(memory.addr);
    if (free_ret != cudaSuccess) {
      return absl::UnknownError(
          absl::StrFormat("fail to free memory, ret_code: %s", cudaGetErrorName(free_ret)));
    }

    return absl::OkStatus();
  }

 private:
  const int32_t device_ordinal_;
  uint32_t memory_alignment_ = 0;
};
}  // namespace

std::shared_ptr<IAllocator> IAllocator::create_device_allocator(int32_t device_ordinal) {
  return std::make_shared<DeviceMemoryAllocator>(device_ordinal);
}
}  // namespace cudaflow
