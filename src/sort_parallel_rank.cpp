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
    constexpr int kMaxSplitterRounds = 64; // TWEAK: Reduced from 128 to 64. Fewer rounds speed up convergence.
    constexpr std::size_t kPWay = 12;        // TWEAK: 4-way is the performance sweet spot for cache line utilization.

    // Ultra-fast stack-allocated node tracking for the multi-way serial merge loop
    template <typename T>
    struct FastHeapNode {
      T value;
      std::size_t seq_idx;
      std::size_t elem_idx;
    };

    template <typename T>
    std::size_t rank_in(const T *s, std::size_t n, const T &v)
    {
      return static_cast<std::size_t>(std::lower_bound(s, s + n, v) - s);
    }

    // TWEAK: Inlined duplicate resolver using clean index adjustments to eliminate vector resizing overheads.
    template <typename T>
    void resolve_duplicates_multi(const T* seqs[kPWay], const std::size_t sizes[kPWay], std::size_t p,
                                  std::size_t rank, std::size_t indices[kPWay])
    {
      std::size_t current_rank = 0;//total rank
      for (std::size_t i = 0; i < p; ++i) current_rank += indices[i];
      
      while (current_rank < rank) {
        int best_seq = -1;
        for (std::size_t i = 0; i < p; ++i) {
          if (indices[i] < sizes[i]) {
            if (best_seq == -1 || seqs[i][indices[i]] < seqs[best_seq][indices[best_seq]]) {
              best_seq = i;
            }
          }
        }
        if (best_seq == -1) break;
        indices[best_seq]++;//inkrementuju takovou sekvenci, která má nejmenší hodnotu
        current_rank++;
      }
    }

    // TWEAK: Completely removed std::vector. Uses raw stack arrays to avoid dynamic memory allocation overhead.
    template <typename T>
    void splitters_by_rank_multi(const T* seqs[kPWay], const std::size_t sizes[kPWay], std::size_t p,
                                 std::size_t rank, std::size_t out_indices[kPWay], std::uint32_t seed)
    {
      for(std::size_t i = 0; i < p; ++i) out_indices[i] = 0;
      if (rank == 0) return;

      std::size_t total = 0;//total size of arrays
      for (std::size_t i = 0; i < p; ++i) total += sizes[i];

      if (rank >= total) {
        for (std::size_t i = 0; i < p; ++i) out_indices[i] = sizes[i];
        return;
      }

      std::size_t L[kPWay] = {0};//left size of array
      std::size_t R[kPWay];//right size of array
      for (std::size_t i = 0; i < p; ++i) R[i] = sizes[i];
      
      std::mt19937 rng(seed);

      for (int round = 0; round < kMaxSplitterRounds; ++round) {
        bool active = false;
        for (std::size_t i = 0; i < p; ++i) {
          if (L[i] < R[i]) { active = true; break; }//if at least one of the array has lower L 
        }
        if (!active) break;

        std::size_t pick = rng() % p;
        while (L[pick] >= R[pick]) {//picks from the still valid arrays
          pick = (pick + 1) % p;
        }

        std::uniform_int_distribution<std::size_t> dist(L[pick], R[pick] - 1);
        const T v = seqs[pick][dist(rng)];//picks one element from the range in a valid array

        std::size_t global = 0;//total rank
        std::size_t m[kPWay];
        for (std::size_t i = 0; i < p; ++i) {
          m[i] = rank_in(seqs[i], sizes[i], v);
          global += m[i];
        }

        if (global >= rank) {
          for (std::size_t i = 0; i < p; ++i) R[i] = m[i];//if rank is higher than R[i] = m[i]
        } else {
          for (std::size_t i = 0; i < p; ++i) L[i] = m[i];//if rank is lower than L[i] = m[i]
        }
      }
      //so apparently it tries find such v L[i]=R[i]

      std::size_t current_sum = 0;
      for (std::size_t i = 0; i < p; ++i) {
        out_indices[i] = L[i];
        current_sum += L[i];
      }

      //problem we didnt hit target rank, to zaručí, že najdeme cílový rank
      if (current_sum != rank) {
        resolve_duplicates_multi(seqs, sizes, p, rank, out_indices);
      }
    }

    // TWEAK: Uses a small, lightning-fast customized insertion-sort heap pattern.
    // std::priority_queue is too slow for 4 items; manually shifting elements in cache is much faster.
    template <typename T>
    void serial_merge_multi(const T* seqs[kPWay], const std::size_t beg_indices[kPWay],
                            const std::size_t end_indices[kPWay], std::size_t p, T *out, std::size_t out_beg)
    {
      FastHeapNode<T> heap[kPWay];
      std::size_t heap_size = 0;

      for (std::size_t i = 0; i < p; ++i) {
        if (beg_indices[i] < end_indices[i]) {
          heap[heap_size++] = {seqs[i][beg_indices[i]], i, beg_indices[i]};
        }
      }

      // Initial sorting of our tiny mini-heap
      for (std::size_t i = 1; i < heap_size; ++i) {
        FastHeapNode<T> key = heap[i];
        std::ptrdiff_t j = i - 1;
        while (j >= 0 && heap[j].value > key.value) {
          heap[j + 1] = heap[j];
          j--;
        }
        heap[j + 1] = key;
      }

      std::size_t out_idx = out_beg;
      while (heap_size > 0) {
        // Root element is always minimum
        FastHeapNode<T> min_node = heap[0];
        out[out_idx++] = min_node.value;

        std::size_t next_elem = min_node.elem_idx + 1;
        if (next_elem < end_indices[min_node.seq_idx]) {
          // Re-insert step using customized array filtering shifts
          FastHeapNode<T> next_node = {seqs[min_node.seq_idx][next_elem], min_node.seq_idx, next_elem};
          std::size_t i = 1;
          while (i < heap_size && heap[i].value < next_node.value) {
            heap[i - 1] = heap[i];
            i++;
          }
          heap[i - 1] = next_node;
        } else {
          // Pull up elements to backfill holes
          for (std::size_t i = 1; i < heap_size; ++i) {
            heap[i - 1] = heap[i];
          }
          heap_size--;
        }
      }
    }

    template <typename T>
    void parallel_merge_multi_way(const T* seqs[kPWay], const std::size_t sizes[kPWay], std::size_t p, T *out)
    {
      std::size_t total = 0;
      for (std::size_t i = 0; i < p; ++i) total += sizes[i];
      if (total == 0) return;

      std::size_t zero_indices[kPWay] = {0};

      if (omp_in_parallel()) {
        serial_merge_multi(seqs, zero_indices, sizes, p, out, 0);
        return;
      }

      const int threads = omp_get_max_threads();
      // TWEAK: Increased grain size work threshold from 64 to 2048 to minimize parallel tasking costs.
      if (total < static_cast<std::size_t>(threads) * 2048) {
        serial_merge_multi(seqs, zero_indices, sizes, p, out, 0);
        return;
      }

      const std::size_t tile = (total + threads - 1) / threads;

#pragma omp parallel for schedule(static)
      for (int t = 0; t < threads; ++t) {
        const std::size_t beg = static_cast<std::size_t>(t) * tile;// thread_id * n/P
        if (beg >= total) continue;
        const std::size_t end = std::min(beg + tile, total);//(thread_id + 1) * n/P

        const std::uint32_t seed = static_cast<std::uint32_t>(beg ^ (end << 16) ^ t);

        std::size_t ia0[kPWay];
        std::size_t ia1[kPWay];

        splitters_by_rank_multi(seqs, sizes, p, beg, ia0, seed);
        splitters_by_rank_multi(seqs, sizes, p, end, ia1, seed + 1u);

        serial_merge_multi(seqs, ia0, ia1, p, out, beg);
      }
    }

    template <typename T>
    void bottom_up_sort_multi(T *src, T *dst, std::size_t n)
    {
      if (n <= 1) return;

      T *a = src;
      T *b = dst;

      for (std::size_t width = 1; width < n; width *= kPWay) {
        std::size_t step = width * kPWay;
        std::size_t num_merges = (n + step - 1) / step;
        const int max_threads = omp_get_max_threads();
        const bool parallel_across = num_merges >= static_cast<std::size_t>(max_threads);

        auto run_merge_step = [&](std::size_t m) {
          std::size_t block_begin = m * step;
          
          const T* seqs[kPWay];
          std::size_t sizes[kPWay];
          std::size_t p = 0;
          
          for (std::size_t w = 0; w < kPWay; ++w) {
            std::size_t chunk_start = block_begin + w * width;
            if (chunk_start >= n) break;
            
            std::size_t chunk_end = std::min(chunk_start + width, n);
            seqs[p] = a + chunk_start;
            sizes[p] = chunk_end - chunk_start;
            p++;
          }

          if (p == 0) return;
          if (p == 1) {
            std::size_t start = block_begin;
            std::size_t length = sizes[0];
            std::copy(a + start, a + start + length, b + start);
            return;
          }

          parallel_merge_multi_way(seqs, sizes, p, b + block_begin);
        };

        if (parallel_across) {
#pragma omp parallel for schedule(static)
          for (std::ptrdiff_t m = 0; m < static_cast<std::ptrdiff_t>(num_merges); ++m) {
            run_merge_step(m);
          }
        } else {
          for (std::size_t m = 0; m < num_merges; ++m) {
            run_merge_step(m);
          }
        }

        std::swap(a, b);
      }

      if (a != src) {
        std::copy(a, a + n, src);
      }
    }

  } // namespace

  template <typename T>
  void sort_by_rank(std::vector<T> &data)
  {
    omp_set_max_active_levels(2);
    if (data.size() <= 1) return;
    
    std::vector<T> buffer(data.size());
    bottom_up_sort_multi(data.data(), buffer.data(), data.size());
  }

  template void sort_by_rank(std::vector<int> &);

} // namespace merge_sort