// Parallel bottom-up merge sort using splitters-by-rank (p-path / NI-PDP
// style).
//
// Lecture idea (Splitters_by_Rank, two sorted inputs as S[0] and S[1]):
//   L[i] = 0, R[i] = length of S[i]
//   repeat while some L[i] < R[i]:
//     pick random pivot v inside current bounds
//     m[i] = rank of v in S[i]   (binary search / lower_bound)
//     if m[0] + m[1] >= target_rank  then  R[i] = m[i]
//     else                             L[i] = m[i]
//   return L[0], L[1]  as splitters for that global rank
//
// Parallelism (same structure as merge-path sort in sort_parallel.cpp):
//   1. bottom_up_sort — many run-pairs in a pass: #pragma omp parallel for over
//   m.
//   2. parallel_merge_by_rank — one large merge: #pragma omp parallel for over
//   t;
//      thread t uses rank = beg and rank = end to find tile boundaries, then
//      serial merge.

#include "merge_sort/merge_sort.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#include <omp.h>

namespace merge_sort
{
  namespace
  {

    constexpr int kMaxSplitterRounds = 128;

    // Exact merge-path co-rank; fallback when randomized splitters stall on
    // duplicate keys. Serial; no parallelism.
    template <typename T>
    std::size_t mergepath_partition(const T *a, std::size_t na, const T *b,
                                    std::size_t nb, std::size_t k)//here is ranked passed
    {
      std::size_t lo = (k > nb) ? (k - nb) : 0;
      std::size_t hi = std::min(k, na);//
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

    // m[i] in the lecture: count of elements in s[0..n) strictly before v in merge
    // order.
    template <typename T>
    std::size_t rank_in(const T *s, std::size_t n, const T &v)
    {
      return static_cast<std::size_t>(std::lower_bound(s, s + n, v) - s);//uses binary search to find idnex, where it could be inserted
    }

    // Splitters_by_Rank for S[0]=left, S[1]=right (p = 2).
    //
    // Finds (out_left, out_right) with out_left + out_right == rank: how many
    // elements to take from each input before position `rank` in the merged output.
    //
    // Serial randomized search; each round narrows [L[i], R[i]) using a random
    // pivot. If iterations do not converge (many equal keys), falls back to
    // mergepath_partition.
    template <typename T>
    void splitters_by_rank(const T *left, std::size_t nleft, const T *right,
                           std::size_t nright, std::size_t rank,
                           std::size_t &out_left, std::size_t &out_right,
                           std::uint32_t seed)// rank = begin of tile
    {
      if (rank == 0)//for first thread
      {
        out_left = 0;
        out_right = 0;
        return;
      }
      const std::size_t total = nleft + nright;//total elements
      if (rank >= total)//check if rank isnt out of array i guess
      {
        out_left = nleft;
        out_right = nright;
        return;
      }

      const T *seqs[2] = {left, right};
      std::size_t L[2] = {0, 0};
      std::size_t R[2] = {nleft, nright};

      std::mt19937 rng(seed);

      for (int round = 0; round < kMaxSplitterRounds; ++round)//max 128 cyklů
      {
        if (L[0] >= R[0] && L[1] >= R[1])// kontrola jestli levá není větší než pravá
        {
          break;
        }

        int pick = static_cast<int>(rng() % 2);
        if (L[pick] >= R[pick])//check
        {
          pick = 1 - pick;//picks the other index
        }
        if (L[pick] >= R[pick])//check
        {
          break;
        }

        std::uniform_int_distribution<std::size_t> dist(L[pick], R[pick] - 1);//picks index from that range
        const T v = seqs[pick][dist(rng)];//picks random value from selected sequence

        const std::size_t m0 = rank_in(left, nleft, v);//returns index on which it could be inserted without sortihg
        const std::size_t m1 = rank_in(right, nright, v);//returns index on which it could be inserted without sorting
        const std::size_t global = m0 + m1;//total rank

        if (global >= rank)//if global rank is higher than rank
        {
          R[0] = m0;//move array end
          R[1] = m1;//move array end
        }
        else
        {
          L[0] = m0;//move array start
          L[1] = m1;//move array start
        }
      }

      out_left = L[0];
      out_right = L[1];

      if (out_left + out_right != rank)//(rank=0, it checks if it isnt at starting position)
      {
        out_left = mergepath_partition(left, nleft, right, nright, rank);
        out_right = rank - out_left;
      }
    }

    // Standard two-pointer merge. Serial only.
    template <typename T>
    void serial_merge(const T *left, std::size_t nleft, const T *right,
                      std::size_t nright, T *out)
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

    // Writes out[beg .. end) from left[ia0..ia1) and right[ib0..ib1). Serial.
    template <typename T>
    void merge_slice(const T *left, std::size_t nleft, const T *right,
                     std::size_t nright, T *out, std::size_t beg, std::size_t end,
                     std::size_t ia0, std::size_t ia1, std::size_t ib0,
                     std::size_t ib1)
    {
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
      (void)end;
    }

    // Merge two sorted runs using splitters-by-rank to partition work across
    // threads.
    //
    // Thread t owns output tile [beg, end) with beg = t * total / p.
    //   splitters_by_rank(..., beg, ...) -> (ia0, ib0)  start on merge path
    //   splitters_by_rank(..., end, ...) -> (ia1, ib1)  end on merge path
    // then merge_slice fills out[beg..end).
    //
    // Parallelism: #pragma omp parallel for over t when not nested and merge is
    // large enough. Falls back to serial_merge when already inside a team or total
    // < threads * 64.
    template <typename T>
    void parallel_merge_by_rank(const T *left, std::size_t nleft, const T *right,
                                std::size_t nright, T *out)
    {
      const std::size_t total = nleft + nright; // total elements in right and left array
      if (total == 0)
      {
        return;
      }

      if (omp_in_parallel())
      {
        serial_merge(left, nleft, right, nright, out); // standard code for merging
        return;
      }

      const int threads = omp_get_max_threads();
      if (total < static_cast<std::size_t>(threads) * 64)
      {
        serial_merge(left, nleft, right, nright, out);
        return;
      }

      const std::size_t tile = (total + static_cast<std::size_t>(threads) - 1) /
                               static_cast<std::size_t>(threads); //how many elements for each thread

      // PARALLEL: one tile per thread; splitters found independently (no sync).
#pragma omp parallel for schedule(static)
      for (int t = 0; t < threads; ++t)
      {
        const std::size_t beg = static_cast<std::size_t>(t) * tile;//maybe start of block used by the thread?
        if (beg >= total)
        {
          continue;
        }
        const std::size_t end = std::min(beg + tile, total);//end index of tile

        const std::uint32_t seed = static_cast<std::uint32_t>(
            beg ^ (end << 16) ^ (nleft << 1) ^ (nright << 2) ^
            (static_cast<std::uint32_t>(t) * 2654435761u));//looks like some random seed

        std::size_t ia0 = 0;//
        std::size_t ib0 = 0;//
        std::size_t ia1 = 0;//
        std::size_t ib1 = 0;//
        splitters_by_rank(left, nleft, right, nright, beg, ia0, ib0, seed);//args: begining of tile, , , seed
        splitters_by_rank(left, nleft, right, nright, end, ia1, ib1, seed + 1u);

        merge_slice(left, nleft, right, nright, out, beg, end, ia0, ia1, ib0, ib1);
      }
    }

    // One bottom-up merge step: two adjacent runs of width `width` -> one run in
    // dst.
    template <typename T>
    void merge_runs(const T *src, T *dst, std::size_t n, std::size_t left_begin,
                    std::size_t width)
    {
      const std::size_t mid = std::min(left_begin + width, n);           // midlle
      const std::size_t right_end = std::min(left_begin + 2 * width, n); // end of second array
      const std::size_t nleft = mid - left_begin;                        // size of left array
      const std::size_t nright = right_end - mid;                        // size of right array

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

      parallel_merge_by_rank(src + left_begin, nleft, src + mid, nright,
                             dst + left_begin);
    }

    // Bottom-up merge sort; ping-pong buffers a/b alternate each pass (width 1, 2,
    // 4, ...).
    //
    // Parallelism per pass:
    //   - num_merges >= max_threads: PARALLEL for over merge index m (disjoint dst
    //   regions).
    //   - else: serial for over m; large merges still parallelize inside
    //   parallel_merge_by_rank.
    template <typename T>
    void bottom_up_sort(T *src, T *dst, std::size_t n)
    {
      if (n <= 1)
      {
        return;
      }

      T *a = src;
      T *b = dst;

      /*
      This loop sets: width -
      */
      for (std::size_t width = 1; width < n; width *= 2)
      {
        const std::size_t num_merges = (n + 2 * width - 1) / (2 * width); // pro (width=1,num_merges=10 001/2 = 5000),(width=1,num_merges=10 003/4 = 2500),(width=2,num_merges=1250)
        const int max_threads = omp_get_max_threads();
        const bool parallel_across =
            num_merges >= static_cast<std::size_t>(max_threads); // kontrola jestli je víc mergů než threadů

        if (parallel_across)
        {
          // PARALLEL: independent merge_runs per iteration m.
#pragma omp parallel for schedule(static)
          for (std::ptrdiff_t m = 0; m < static_cast<std::ptrdiff_t>(num_merges);
               ++m)
          {
            const std::size_t left_begin = static_cast<std::size_t>(m) * 2 * width;
            merge_runs(a, b, n, left_begin, width);
          }
        }
        else
        {
          for (std::size_t m = 0; m < num_merges; ++m) // přes všechny merge
          {
            const std::size_t left_begin = m * 2 * width; //(width=1,m=0,left_begin=0),(width=1,m=1,left_begin=2)
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
  void sort_by_rank(std::vector<T> &data)
  {
    omp_set_max_active_levels(2);
    if (data.size() <= 1)
    {
      return;
    }
    std::vector<T> buffer(data.size());
    bottom_up_sort(data.data(), buffer.data(), data.size());
  }

  template void sort_by_rank(std::vector<int> &);

} // namespace merge_sort
