#include <merge_sort/merge_sort.hpp>
#include <quick_sort/quick_sort_serial.hpp>
#include <quick_sort/quick_sort_parallel.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <omp.h>

namespace
{

  constexpr const char *kReset = "\033[0m";
  constexpr const char *kGreen = "\033[32m";
  constexpr const char *kRed = "\033[31m";
  constexpr const char *kDim = "\033[2m";

  [[noreturn]] void fail(const char *test_name, const char *message)
  {
    std::cerr << kRed << "✗ FAIL" << kReset << "  " << test_name << kDim << " — "
              << message << kReset << '\n';
    std::abort();
  }

#define CHECK(test_name, cond)                       \
  do                                                 \
  {                                                  \
    if (!(cond))                                     \
    {                                                \
      fail((test_name), "assertion failed: " #cond); \
    }                                                \
  } while (0)

  void pass(const char *test_name)
  {
    std::cout << kGreen << "✓ PASS" << kReset << "  " << test_name << '\n';
  }

  template <typename Fn>
  void run_test(const char *name, Fn fn)
  {
    fn();
    pass(name);
  }

  template <typename T>
  bool is_sorted_vec(const std::vector<T> &v)
  {
    return std::is_sorted(v.begin(), v.end());
  }

  template <typename SortFn>
  void test_10_elements(SortFn sort)
  {
    const char *name = "10 elements";
    std::vector<int> v = {9, 3, 7, 1, 8, 2, 6, 0, 5, 4};
    sort(v);
    CHECK(name, is_sorted_vec(v));
    CHECK(name, v == std::vector<int>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
  }

  template <typename SortFn>
  void test_20_elements(SortFn sort)
  {
    const char *name = "20 elements";
    std::vector<int> v = {19, 7, 15, 3, 11, 1, 18, 6, 14, 2, 10, 0, 17, 5, 13, 9, 16, 4, 12, 8};
    sort(v);
    CHECK(name, is_sorted_vec(v));
    for (int i = 0; i < 20; ++i)
    {
      CHECK(name, v[static_cast<std::size_t>(i)] == i);
    }
  }

  template <typename SortFn>
  void test_random(std::size_t n, std::uint32_t seed, SortFn sort)
  {
    const std::string name = std::to_string(n) + " random elements (seed " + std::to_string(seed) + ")";
    std::vector<int> v(n);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(-1'000'000, 1'000'000);
    for (auto &x : v)
    {
      x = dist(rng);
    }
    std::vector<int> expected = v;
    std::sort(expected.begin(), expected.end());
    sort(v);
    CHECK(name.c_str(), v == expected);
  }

  constexpr int kSpeedupThreads = 12;
  constexpr std::size_t kSpeedupSize = 100'000;
  constexpr double kMinSpeedup = 2.0;
  constexpr int kSpeedupWarmup = 2;
  constexpr int kSpeedupTimedRuns = 5;

  using Clock = std::chrono::steady_clock;

  double seconds(Clock::duration d)
  {
    return std::chrono::duration<double>(d).count();
  }

  std::vector<int> random_vector(std::size_t n, std::uint32_t seed)
  {
    std::vector<int> v(n);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(-1'000'000, 1'000'000);
    for (auto &x : v)
    {
      x = dist(rng);
    }
    return v;
  }

  template <typename SortFn>
  double bench_best(SortFn sort_fn, const std::vector<int> &input, int warmup, int runs)
  {
    for (int w = 0; w < warmup; ++w)
    {
      std::vector<int> copy = input;
      sort_fn(copy);
    }

    double best = std::numeric_limits<double>::infinity();
    for (int r = 0; r < runs; ++r)
    {
      std::vector<int> copy = input;
      const auto t0 = Clock::now();
      sort_fn(copy);
      const auto t1 = Clock::now();
      best = std::min(best, seconds(t1 - t0));
    }
    return best;
  }

  void test_rank_matches_merge_path_on_random_merges()
  {
    const char *name = "sort_by_rank matches std::sort (cross-check sizes)";
    for (std::size_t n : {10, 20, 1000})
    {
      std::vector<int> v(n);
      std::mt19937 rng(static_cast<std::uint32_t>(n) * 17u);
      std::uniform_int_distribution<int> dist(-500'000, 500'000);
      for (auto &x : v)
      {
        x = dist(rng);
      }
      std::vector<int> expected = v;
      std::sort(expected.begin(), expected.end());

      std::vector<int> by_rank = v;
      merge_sort::sort_by_rank(by_rank);
      CHECK(name, by_rank == expected);

      std::vector<int> by_path = v;
      merge_sort::sort(by_path);
      CHECK(name, by_path == expected);
      CHECK(name, by_rank == by_path);
    }
  }

  void test_speedup_100k_merge_path()
  {
    const char *name = "100k speedup merge-path (12 threads, >=2x vs serial)";

    const int prev_threads = omp_get_max_threads();
    omp_set_num_threads(kSpeedupThreads);
    omp_set_dynamic(0);

    const std::vector<int> input = random_vector(kSpeedupSize, 99'001);

    const double serial_s = bench_best(
        [](std::vector<int> &v)
        { merge_sort::sort_serial(v); }, input, kSpeedupWarmup,
        kSpeedupTimedRuns);

    const double parallel_s = bench_best(
        [](std::vector<int> &v)
        { merge_sort::sort(v); }, input, kSpeedupWarmup, kSpeedupTimedRuns);

    omp_set_num_threads(prev_threads);

    const double speedup = serial_s / parallel_s;
    std::cout << kDim << "  serial: " << serial_s << "s  merge-path(" << kSpeedupThreads
              << " threads): " << parallel_s << "s  speedup: " << speedup << "x" << kReset << '\n';

    CHECK(name, parallel_s > 0.0);
    CHECK(name, speedup >= kMinSpeedup);
  }

  void test_speedup_1m_rank()
  {
    const char *name = "1m speedup splitters-by-rank (12 threads, >=2x vs serial)";

    const int prev_threads = omp_get_max_threads();
    omp_set_num_threads(kSpeedupThreads);
    omp_set_dynamic(0);

    const std::vector<int> input = random_vector(1000000, 99'002);

    const double serial_s = bench_best(
        [](std::vector<int> &v)
        { merge_sort::sort_serial(v); }, input, kSpeedupWarmup,
        kSpeedupTimedRuns);

    const double parallel_s = bench_best(
        [](std::vector<int> &v)
        { merge_sort::sort_by_rank(v); }, input, kSpeedupWarmup,
        kSpeedupTimedRuns);

    omp_set_num_threads(prev_threads);

    const double speedup = serial_s / parallel_s;
    std::cout << kDim << "  serial: " << serial_s << "s  rank(" << kSpeedupThreads
              << " threads): " << parallel_s << "s  speedup: " << speedup << "x" << kReset << '\n';

    CHECK(name, parallel_s > 0.0);
    CHECK(name, speedup >= kMinSpeedup);
  }

  void test_speedup_1m_quick()
  {
    const char *name = "1m speedup quick sort (12 threads, >=2x vs serial)";

    const int prev_threads = omp_get_max_threads();
    omp_set_num_threads(kSpeedupThreads);
    omp_set_dynamic(0);

    const std::vector<int> input = random_vector(1000000, 99'002);

    const double serial_s = bench_best(
        [](std::vector<int> &v)
        { quick_sort::sort_serial(v); }, input, kSpeedupWarmup,
        kSpeedupTimedRuns);

    const double parallel_s = bench_best(
        [](std::vector<int> &v)
        { quick_sort_parallel::sort(v); }, input, kSpeedupWarmup,
        kSpeedupTimedRuns);

    omp_set_num_threads(prev_threads);

    const double speedup = serial_s / parallel_s;
    std::cout << kDim << "  serial: " << serial_s << "s  rank(" << kSpeedupThreads
              << " threads): " << parallel_s << "s  speedup: " << speedup << "x" << kReset << '\n';

    CHECK(name, parallel_s > 0.0);
    CHECK(name, speedup >= kMinSpeedup);
  }

} // namespace

int main()
{
  const auto sort_parallel = [](std::vector<int> &v)
  { merge_sort::sort(v); };
  const auto sort_serial = [](std::vector<int> &v)
  { merge_sort::sort_serial(v); };
  const auto sort_srial_quick = [](std::vector<int> &v)
  { quick_sort::sort_serial(v); };
  const auto sort_parallel_quick = [](std::vector<int> &v)
  { quick_sort_parallel::sort(v); };

  std::cout << kDim << "quick_sort tests (parallel)" << kReset << '\n';
  run_test("10 elements", [&]
           { test_10_elements(sort_parallel_quick); });
  run_test("20 elements", [&]
           { test_20_elements(sort_parallel_quick); });
  run_test("10000 random elements", [&]
           { test_random(10'000, 42, sort_parallel_quick); });
  run_test("100000 random elements", [&]
           { test_random(100'000, 12345, sort_parallel_quick); });

  std::cout << kDim << "merge_sort tests (parallel)" << kReset << '\n';
  run_test("10 elements", [&]
           { test_10_elements(sort_parallel); });
  run_test("20 elements", [&]
           { test_20_elements(sort_parallel); });
  run_test("10000 random elements", [&]
           { test_random(10'000, 42, sort_parallel); });
  run_test("100000 random elements", [&]
           { test_random(100'000, 12345, sort_parallel); });

  const auto sort_by_rank = [](std::vector<int> &v)
  { merge_sort::sort_by_rank(v); };

  std::cout << kDim << "merge_sort tests (parallel, splitters-by-rank)" << kReset << '\n';
  run_test("rank: 10 elements", [&]
           { test_10_elements(sort_by_rank); });
  run_test("rank: 20 elements", [&]
           { test_20_elements(sort_by_rank); });
  run_test("rank: 1000 random elements", [&]
           { test_random(1'000, 88, sort_by_rank); });
  run_test("rank: 10000 random elements", [&]
           { test_random(10'000, 89, sort_by_rank); });
  run_test("rank: 100000 random elements", [&]
           { test_random(100'000, 90, sort_by_rank); });
  run_test("rank matches merge-path and std::sort", test_rank_matches_merge_path_on_random_merges);

  std::cout << kDim << "merge_sort tests (serial)" << kReset << '\n';
  run_test("serial: 10 elements", [&]
           { test_10_elements(sort_serial); });
  run_test("serial: 20 elements", [&]
           { test_20_elements(sort_serial); });
  run_test("serial: 1000 random elements", [&]
           { test_random(1'000, 77, sort_serial); });

  std::cout << kDim << "quick_sort tests (serial)" << kReset << '\n';
  run_test("serial: 10 elements", [&]
           { test_10_elements(sort_srial_quick); });
  run_test("serial: 20 elements", [&]
           { test_20_elements(sort_srial_quick); });
  run_test("serial: 1000 random elements", [&]
           { test_random(1'000, 77, sort_srial_quick); });

  std::cout << kDim << "merge_sort tests (performance)" << kReset << '\n';
  run_test("100k speedup merge-path (12 threads, >=2x vs serial)", test_speedup_100k_merge_path);
  run_test("1m speedup splitters-by-rank (12 threads, >=2x vs serial)", test_speedup_1m_rank);
  run_test("1m speedu quick (12 threads, >=2x vs serial)", test_speedup_1m_quick);
  std::cout << kGreen << "All tests passed." << kReset << '\n';
  return 0;
}
