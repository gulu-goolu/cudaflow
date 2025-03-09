#include "concurrentqueue.h"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <thread>
#include <vector>

#include "gflags/gflags.h"

DEFINE_int32(num_thread, 10, "");
DEFINE_int32(repeat_times, 100000, "");

template <typename T>
using ConcurrentQueue = moodycamel::ConcurrentQueue<T, moodycamel::ConcurrentQueueDefaultTraits>;

void fn(moodycamel::ConcurrentQueue<void*>* q, int32_t times) {
  for (int32_t i = 0; i < times; ++i) {
    q->enqueue(nullptr);
  }
}

void run_test(int32_t num_thread, int32_t times) {
  std::vector<std::thread> threads;

  moodycamel::ConcurrentQueue<void*> q;

  auto tp0 = std::chrono::steady_clock::now();

  for (int32_t i = 0; i < num_thread; ++i) {
    threads.push_back(std::thread([&q, times] { fn(&q, times); }));
  }

  for (auto& t : threads) {
    t.join();
  }

  auto tp1 = std::chrono::steady_clock::now();
  std::cout << "elapsed time: " << (tp1 - tp0).count() / 1.e03
            << " us, avg: " << double((tp1 - tp0).count()) / (times) << " op/ns" << std::endl;
}

int main(int argc, char* argv[]) {
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);

  run_test(FLAGS_num_thread, FLAGS_repeat_times);
}
