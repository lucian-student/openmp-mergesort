// Parallel bottom-up merge sort.
//
// Parallelism appears at two levels:
//   1. bottom_up_sort — when many independent run-pairs exist in a pass, merge them
//      in parallel across threads (#pragma omp parallel for over merge index m).
//   2. parallel_merge — when a single merge is large but few merges remain, split that
//      merge along the merge path into tiles (#pragma omp parallel for over tile t).
//
// Early passes use (1); late passes rely more on (2). Small merges always fall back to
// serial_merge to avoid OpenMP overhead.

#include "merge_sort/merge_sort.hpp"

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include <omp.h>

namespace merge_sort
{
  namespace
  {

    // --- Merge-path helpers (serial; used to split work inside parallel_merge) ---

    // Given sorted arrays A[0..na) and B[0..nb), return how many elements from A appear in
    // the first k positions of the merged output (the "co-rank" on the merge path).
    // Binary search on the diagonal i + j = k; no parallelism.
    template <typename T>
    std::size_t mergepath_partition(const T *a, std::size_t na, const T *b, std::size_t nb,
                                    std::size_t k)
    {
      std::size_t lo = (k > nb) ? (k - nb) : 0;
      std::size_t hi = std::min(k, na);
      while (lo < hi)
      {
        const std::size_t mid = lo + (hi - lo) / 2;
        const std::size_t j = k - mid;
        if (mid < na && j > 0 && a[mid] <= b[j - 1])
        {
          lo = mid + 1;
        }
        else
        {
          hi = mid;
        }
      }
      return lo;
    }

    // Classic two-pointer merge of two sorted ranges into out[0 .. nleft+nright).
    // Serial only; used as fallback inside parallel_merge and for tiny merges.
    template <typename T>
    void serial_merge(const T *left, std::size_t nleft, const T *right, std::size_t nright, T *out)
    {
      std::size_t i = 0;
      std::size_t j = 0;
      std::size_t k = 0;
      while (i < nleft && j < nright)
      {
        if (left[i] <= right[j])
        {
          out[k++] = left[i++];
        }
        else
        {
          out[k++] = right[j++];
        }
      }
      while (i < nleft)
      {
        out[k++] = left[i++];
      }
      while (j < nright)
      {
        out[k++] = right[j++];
      }
    }

    // Merge two sorted inputs into one sorted output using the merge-path algorithm.
    //
    // Parallelism: when called from the top level (not already inside an OpenMP team) and
    // the output length is large enough, the merged range is split into one tile per
    // thread. Each thread:
    //   - finds its sub-range [beg, end) on the output via mergepath_partition,
    //   - derives the corresponding slices of left and right,
    //   - runs a short serial merge into out[beg .. end).
    // Tiles are disjoint, so no synchronization is needed between threads.
    //
    // Falls back to serial_merge when:
    //   - already inside a parallel region (e.g. called from merge_runs under parallel for),
    //   - or total output size < threads * 64 (overhead would dominate).
    template <typename T>
    void parallel_merge(const T *left, std::size_t nleft, const T *right, std::size_t nright, T *out)
    {
      const std::size_t total = nleft + nright;
      if (total == 0)
      {
        return;
      }

      if (omp_in_parallel())
      {
        serial_merge(left, nleft, right, nright, out);
        return;
      }

      const int threads = omp_get_max_threads();
      if (total < static_cast<std::size_t>(threads) * 64)
      {
        serial_merge(left, nleft, right, nright, out);
        return;
      }

      const std::size_t tile =
          (total + static_cast<std::size_t>(threads) - 1) / static_cast<std::size_t>(threads);

      // PARALLEL: each iteration t owns output indices [beg, end) of the merged array.
#pragma omp parallel for schedule(static)
      for (int t = 0; t < threads; ++t)
      {
        const std::size_t beg = static_cast<std::size_t>(t) * tile;
        if (beg >= total)
        {
          continue;
        }
        const std::size_t end = std::min(beg + tile, total);

        // Start/end positions on the merge path for this tile.
        const std::size_t ia0 = mergepath_partition(left, nleft, right, nright, beg);
        const std::size_t ia1 = mergepath_partition(left, nleft, right, nright, end);
        const std::size_t ib0 = beg - ia0;
        const std::size_t ib1 = end - ia1;

        // Serial merge of left[ia0..ia1) with right[ib0..ib1) into out[beg..end).
        std::size_t i = ia0;
        std::size_t j = ib0;
        std::size_t k = beg;
        while (i < ia1 && j < ib1)
        {
          if (left[i] <= right[j])
          {
            out[k++] = left[i++];
          }
          else
          {
            out[k++] = right[j++];
          }
        }
        while (i < ia1)
        {
          out[k++] = left[i++];
        }
        while (j < ib1)
        {
          out[k++] = right[j++];
        }
      }
    }

    // Merge one pair of adjacent sorted runs in the bottom-up tree:
    //   src[left_begin .. left_begin+width)  and  src[left_begin+width .. left_begin+2*width)
    // into dst at the same index range. Tail runs (only left or only right) are copied.
    // Calls parallel_merge for a full left+right pair (which may parallelize inside).
    template <typename T>
    void merge_runs(const T *src, T *dst, std::size_t n, std::size_t left_begin, std::size_t width)
    {
      const std::size_t mid = std::min(left_begin + width, n);
      const std::size_t right_end = std::min(left_begin + 2 * width, n);
      const std::size_t nleft = mid - left_begin;
      const std::size_t nright = right_end - mid;

      if (nright == 0)
      {
        std::copy(src + left_begin, src + mid, dst + left_begin);
        return;
      }
      if (nleft == 0)
      {
        std::copy(src + mid, src + right_end, dst + left_begin);
        return;
      }

      parallel_merge(src + left_begin, nleft, src + mid, nright, dst + left_begin);
    }

    // Bottom-up merge sort with ping-pong buffers src/dst (aliased as a/b).
    //
    // Passes: width = 1, 2, 4, ... until the whole array is one sorted run.
    //
    // Parallelism per pass:
    //   - If num_merges >= max_threads: PARALLEL across merges — each thread handles
    //     disjoint index ranges via merge_runs (may call parallel_merge internally).
    //   - Else: serial loop over merges; parallelism can still happen inside parallel_merge
    //     when individual merges are large (typical late passes with few, big merges).
    template <typename T>
    void bottom_up_sort(T *src, T *dst, std::size_t n)
    {
      if (n <= 1)
      {
        return;
      }

      T *a = src;
      T *b = dst;

      for (std::size_t width = 1; width < n; width *= 2)
      {
        const std::size_t num_merges = (n + 2 * width - 1) / (2 * width);
        const int max_threads = omp_get_max_threads();
        const bool parallel_across = num_merges >= static_cast<std::size_t>(max_threads);

        if (parallel_across)
        {
          // PARALLEL: one merge_runs per iteration; merges touch disjoint dst regions.
#pragma omp parallel for schedule(static)
          for (std::ptrdiff_t m = 0; m < static_cast<std::ptrdiff_t>(num_merges); ++m)
          {
            const std::size_t left_begin = static_cast<std::size_t>(m) * 2 * width;
            merge_runs(a, b, n, left_begin, width);
          }
        }
        else
        {
          for (std::size_t m = 0; m < num_merges; ++m)
          {
            const std::size_t left_begin = m * 2 * width;
            merge_runs(a, b, n, left_begin, width);
          }
        }

        std::swap(a, b);
      }

      if (a != src)
      {
        std::copy(a, a + n, src);
      }
    }

  } // namespace

  template <typename T>
  void sort(std::vector<T> &data)
  {
    if (data.size() <= 1)
    {
      return;
    }
    std::vector<T> buffer(data.size());
    bottom_up_sort(data.data(), buffer.data(), data.size());
  }

  template void sort(std::vector<int> &);

} // namespace merge_sort
