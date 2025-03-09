#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "absl/status/statusor.h"

namespace cudaflow {

struct CachingAllocatorOptions {
  uint32_t small_size_limits = 2 * 1024 * 1024;
};

struct Memory {
  void* addr;
};

class IAllocator : public std::enable_shared_from_this<IAllocator> {
 public:
  virtual ~IAllocator() = default;

  virtual uint32_t memory_alignment() const = 0;
  virtual absl::StatusOr<Memory> allocate(size_t size) = 0;
  virtual absl::Status deallocate(Memory memory) = 0;

  /**
   * @brief Create cuda device memory allocator
   *
   * @param device_ordinal
   * @return std::shared_ptr<IAllocator>
   */
  static std::shared_ptr<IAllocator> create_device_allocator(int32_t device_ordinal);

  static std::shared_ptr<IAllocator> create_caching_allocator(
      const std::shared_ptr<IAllocator> base, const CachingAllocatorOptions& options);
};

}  // namespace cudaflow
