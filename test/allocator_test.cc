#include "cudaflow/allocator.h"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <thread>
#include <vector>

#include "gflags/gflags.h"

DEFINE_int32(num_thread, 4, "");
DEFINE_int32(repeats, 10000, "");

void thread_fn(cudaflow::IAllocator* allocator, int32_t repeats) {
  for (int32_t i = 0; i < repeats; ++i) {
    auto mem = allocator->allocate(1024L).value();

    allocator->deallocate(mem);
  }
}

void run_test(cudaflow::IAllocator* allocator, int32_t num_thread, int32_t repeats) {
  auto tp0 = std::chrono::steady_clock::now();

  std::vector<std::thread> threads;

  for (int32_t i = 0; i < num_thread; ++i) {
    threads.push_back(std::thread([allocator, repeats] { thread_fn(allocator, repeats); }));
  }

  for (auto& t : threads) {
    t.join();
  }

  auto tp1 = std::chrono::steady_clock::now();
  std::cout << "elapsed time: " << (tp1 - tp0).count() / 1.0e3
            << " us, avg: " << double((tp1 - tp0).count()) / repeats << " ns/op" << std::endl;
}

int main(int argc, char* argv[]) {
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);

  auto allocator = cudaflow::IAllocator::create_device_allocator(0);
  run_test(allocator.get(), FLAGS_num_thread, FLAGS_repeats);
}
