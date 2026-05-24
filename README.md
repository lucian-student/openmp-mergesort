# Parallel merge sort (merge-path + OpenMP)

Bottom-up merge sort with **merge-path** partitioning for parallel merges, built with CMake and OpenMP. Includes a serial baseline (`sort_serial`) for correctness checks and benchmarks.

## Quick start (`build`)

Default directory for everyday development, tests, and performance work:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/merge_sort_tests
```

Or via CTest (sets `OMP_NUM_THREADS=12` for the speedup test):

```bash
ctest --test-dir build
```

## `build` vs `build-tsan`

Both are **separate CMake binary directories**. They do not share object files or caches. You configure each once (or after CMake/toolchain changes) and build independently.

| | **`build`** | **`build-tsan`** |
|---|-------------|------------------|
| **Purpose** | Normal development, CI, benchmarks | Find data races and threading bugs |
| **Build type** | `Release` (recommended) | `Debug` |
| **Sanitizer** | None | [ThreadSanitizer](https://github.com/google/sanitizers/wiki/ThreadSanitizerCppManual) (`-fsanitize=thread`) |
| **Debug symbols** | Minimal (Release) | Full (`-g`) |
| **Speed** | Fast (optimized) | Much slower (instrumentation + no optimization) |
| **Typical use** | Run `merge_sort_tests`, measure speedup | Run tests/binary under TSan when changing OpenMP code |

### Configure `build` (Release)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Use this tree for:

- Correctness tests (10 / 20 / 10k / 100k elements)
- Serial merge-sort tests (10 / 20 / 1k)
- The 100k speedup tests for merge-path and splitters-by-rank (each expects ≥2× vs serial with 12 threads)

### Configure `build-tsan` (ThreadSanitizer)

```bash
cmake -B build-tsan \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=thread -g" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=thread"
cmake --build build-tsan
./build-tsan/merge_sort_tests
```

ThreadSanitizer adds runtime checks for unsynchronized memory access between threads. It is the right tool when debugging parallel merge logic, OpenMP regions, or merge-path tile boundaries.

**Notes:**

- Requires a toolchain with TSan support (GCC/Clang on Linux usually works).
- OpenMP + TSan can be finicky (extra overhead, occasional false positives or crashes). If TSan fails to run, try fewer threads: `OMP_NUM_THREADS=1 ./build-tsan/merge_sort_tests`.
- After refactoring sources, re-run the `cmake -B build-tsan ...` line so the TSan tree picks up new files (same as for `build`).
- Do **not** use `build-tsan` for the speedup benchmark; timings are meaningless there.

## Project layout

```
include/merge_sort/merge_sort.hpp   Public API (declarations only)
src/sort_serial.cpp                Serial bottom-up merge sort
src/sort_parallel.cpp              Parallel sort: merge-path tiles + OpenMP
src/sort_parallel_rank.cpp         Parallel sort: splitters-by-rank + OpenMP
tests/test_merge_sort.cpp          Tests (cassert + speedup check)
```

Implementations are separate translation units (no shared internal headers).

## API

- `merge_sort::sort(std::vector<T>&)` — parallel sort (merge-path partitioning)
- `merge_sort::sort_by_rank(std::vector<T>&)` — parallel sort (randomized splitters-by-rank, NI-PDP p-path idea)
- `merge_sort::sort_serial(std::vector<T>&)` — same bottom-up structure, serial merges only

Explicit instantiations are provided for `int` in the `.cpp` files. For other element types, add matching `template void sort(...)` / `sort_serial(...)` lines in the source files.

## Requirements

- CMake ≥ 3.16
- C++17 compiler with OpenMP
- For `build-tsan`: compiler with `-fsanitize=thread`
