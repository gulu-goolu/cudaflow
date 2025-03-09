#include <cstddef>
#include <memory>

#include "concurrentqueue.h"
#include "cudaflow/allocator.h"

namespace cudaflow {
namespace {
struct MemoryBlock {};

struct AddrAndSize {
  void *addr;
  size_t size;
};

class CachingAllocator : public IAllocator {
 public:
  CachingAllocator(std::shared_ptr<IAllocator> base, const CachingAllocatorOptions &options)
      : base_(base), options_(options) {}

  uint32_t memory_alignment() const override { return base_->memory_alignment(); }

  absl::StatusOr<Memory> allocate(size_t size) override {
    if (size <= options_.small_size_limits) {
    } else {
    }
    return absl::OkStatus();
  }

  absl::Status deallocate(Memory memory) override { return absl::OkStatus(); }

 private:
  std::shared_ptr<IAllocator> base_;
  const CachingAllocatorOptions options_;

  moodycamel::ConcurrentQueue<AddrAndSize> deallocated_;
};
}  // namespace

std::shared_ptr<IAllocator> IAllocator::create_caching_allocator(
    const std::shared_ptr<IAllocator> base, const CachingAllocatorOptions &options) {
  return std::make_shared<CachingAllocator>(base, options);
}
}  // namespace cudaflow
