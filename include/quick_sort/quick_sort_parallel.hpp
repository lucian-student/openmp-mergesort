#pragma once

#include <vector>
#include <algorithm>
#include <cstddef>
#include <utility>
#include <random>
#include <functional>
#include <iostream>
#include <omp.h>

namespace quick_sort_parallel
{
    // Tuning parameters for the parallel partition algorithm
    constexpr size_t BLOCK_SIZE = 10000;         // Increased to reduce atomic contention
    constexpr size_t PARALLEL_THRESHOLD = 60000; // Raised to avoid parallel overhead on mid-sized chunks
    constexpr size_t TASK_THRESHOLD = 30000;     // Raised to drop to fast sequential sort sooner

    template <typename T>
    size_t select_pivot(T *data, size_t l, size_t h)
    {
        size_t mid = l + (h - l) / 2;

        // Sort the low, middle, and high elements in place to find the median index
        if (data[mid] < data[l])
            std::swap(data[l], data[mid]);
        if (data[h] < data[l])
            std::swap(data[l], data[h]);
        if (data[h] < data[mid])
            std::swap(data[mid], data[h]);

        // Return the index containing the median value
        return mid;
    }

    // Purely sequential fallback functions to avoid execution overhead down the tree
    template <typename T>
    size_t partition_sequential(T *data, size_t l, size_t h)
    {
        auto pivot_index = select_pivot(data, l, h);
        auto pivot = data[pivot_index];
        std::swap(data[pivot_index], data[h]);
        size_t i = l;
        for (size_t j = l; j < h; ++j)
        {
            if (data[j] < pivot)
            {
                if (i != j)
                    std::swap(data[i], data[j]);
                i++;
            }
        }
        std::swap(data[i], data[h]);
        return i;
    }

    template <typename T>
    void quick_sequential(T *data, ssize_t l, ssize_t h)
    {
        if (l >= h)
            return;
        size_t mid = partition_sequential(data, l, h);
        quick_sequential(data, l, mid - 1);
        quick_sequential(data, mid + 1, h);
    }

    struct Interval
    {
        ssize_t start;
        ssize_t end;
    };

    /**
     * Checks if two intervals overlap.
     * This assumes inclusive boundaries [start, end].
     */
    bool do_intervals_overlap(Interval a, Interval b)
    {
        // Optional: Ensure the intervals are well-formed (start <= end)
        // If your data guarantees this, you can omit these two lines.
        if (a.start > a.end)
            std::swap(a.start, a.end);
        if (b.start > b.end)
            std::swap(b.start, b.end);

        // The core overlap condition
        return (a.start <= b.end) && (b.start <= a.end);
    }

    template <typename T>
    size_t partition_parallel(T *data, size_t l, size_t h)
    {
        /**
         * @brief Thread-Safe Block-Based Parallel Partitioning
         *
         * STRATEGY:
         * Split [l, h] into fixed-size blocks. Threads atomically grab a Left Block
         * (scanning for > pivot) and a Right Block (scanning for < pivot) to swap elements.
         *
         * 1. MIDDLE-BLOCK ANOMALIES (The Convergence Point):
         *    A) OVERLAPPED BLOCKS: Left/Right allocations cross into the same block.
         *       - Optimization: Dynamically shrink block boundaries to the intersection point.
         *         Process safe segments immediately; this anchors the true pivot boundary.
         *    B) UNPAIRED BLOCKS: A block is allocated on one side, but the opposite side is exhausted.
         *       - Resolution: Ignore and leave untouched. Defer to the Cleanup Phase.
         *
         * 2. CLEANUP PHASE STAGE 1: SECONDARY PARALLEL PAIRING (The Multi-Pass Reduce)
         * - At initial termination, up to P blocks remain "dirty" or "unpaired".
         * - Invariant: Left-dirty blocks contain elements >= pivot.
         *              Right-dirty blocks contain elements <= pivot.
         * - Execution: Initialize a new set of atomic counters targeting ONLY the collected dirty blocks.
         * - Action: Threads parallel-match remaining Left-dirty blocks with Right-dirty blocks.
         *   Mismatched elements are swapped across these blocks in parallel.
         * - Repeat/Continue until it is mathematically impossible to form pairs.
         *
         * 3. CLEANUP PHASE STAGE 2: SINGLE-SIDE MIDDLE SWAP (The Final Squeeze)
         * - Post-Stage 1, the remaining dirty blocks will be strictly homogenic:
         *   either ONLY Left-dirty blocks remain, or ONLY Right-dirty blocks remain.
         * - Pivot Placement: Locate the exact "middle" boundary established by the clean/shrunk blocks.
         * - Action: Sequentially swap the remaining elements from these one-sided dirty blocks
         *   with the elements adjacent to the true middle boundary line.
         */
        std::cout << "PARTITIONING:" << std::endl;
        size_t MAX_THREADS = 48;
        // used this in fixup: swap between halfs based on index of fixup iteration
        // either std::pair<ssize_t, ssize_t> *currentLeftDirty = leftDirty;
        // or std::pair<ssize_t, ssize_t> *currentLeftDirty = leftDirty + MAX_THREADS;
        std::pair<ssize_t, ssize_t> leftDirty[MAX_THREADS * 2];
        std::pair<ssize_t, ssize_t> rightDirty[MAX_THREADS * 2];
        std::pair<ssize_t, ssize_t> *currentLeftDirty = leftDirty;
        std::pair<ssize_t, ssize_t> *currentRightDirty = rightDirty;

        ssize_t leftBlocks = 0;
        ssize_t rightBlocks = 0;

        ssize_t nextLeftBlocks = 0;
        ssize_t nextRightBlocks = 0;

        auto pivot_index = select_pivot(data, l, h);
        auto pivot = data[pivot_index];
        std::swap(data[pivot_index], data[h]);

        ssize_t next_left = l;
        ssize_t next_right = h - 1;

        omp_lock_t local_lock;
        omp_init_lock(&local_lock);

        /*
        Problém:
        1. 
        */
#pragma omp parallel
        {
#pragma omp master
            {
                std::cout << omp_get_num_threads() << std::endl;
            }

            ssize_t total_right_start = h - 1;
            ssize_t total_left_end = l;
            ssize_t left_start = 0;
            ssize_t left_end = 0;
            ssize_t right_start = 0;
            ssize_t right_end = 0;
            bool has_left = false;
            bool has_right = false;
            while (true)
            {
                auto prev_left = has_left;
                auto prev_right = has_right;

                /*
                Podle mě tady musí být kritická sekce
                1. 
                */
                if (!has_left)
                {
#pragma omp atomic capture
                    {
                        left_start = next_left;
                        next_left += BLOCK_SIZE;
                    }
                    left_end = std::min(left_start + BLOCK_SIZE, h - 1);
                    has_left = true;
                }
                if (!has_right)
                {
#pragma omp atomic capture
                    {
                        right_end = next_right;
                        next_right -= BLOCK_SIZE;
                    }
                    right_start = std::max(right_end - BLOCK_SIZE, l);
                    has_right = true;
                }

                // UNPAIRED DECTION is completely wrong
                // this is still bad detection
                // because u dont differnetiate UNPAIRED with OVERCOMITTED
                if (right_start < left_start || left_start > left_end || right_start > right_end)
                {
                    has_left = prev_left;
                    has_right = prev_right;
                    break;
                }

                // MIDDLE FIX
                if (right_start < left_end)
                {
                    right_start = left_end;
                }

                if (left_end > total_left_end)
                    total_left_end = left_end;
                if (right_start < total_right_start)
                    total_right_start = right_start;

                // here swap the elements
                while (left_start <= left_end && right_start <= right_end)
                {
                    while (left_start <= left_end && data[left_start] < pivot)
                    {
                        left_start++;
                    }
                    while (right_start <= right_end && !(data[right_end] < pivot))
                    {
                        right_end--;
                    }

                    if (left_start <= left_end && right_start <= right_end)
                    {
                        std::swap(data[left_start], data[right_end]);
                        left_start++;
                        right_end--;
                    }
                }
                //

                // Check if this thread has finished processing either block
                if (left_start > left_end)
                    has_left = false;
                if (right_start > right_end)
                    has_right = false;
            }

            omp_set_lock(&local_lock);
            std::cout << "(" << total_left_end << "," << total_right_start << ")" << (total_left_end < total_right_start) << std::endl;
            omp_unset_lock(&local_lock);

#pragma omp barrier

            if (has_left && left_start <= left_end)
            {
                ssize_t idx;
#pragma omp atomic capture
                idx = leftBlocks++;
                currentLeftDirty[idx] = {left_start, left_end};
            }
            if (has_right && right_start <= right_end)
            {
                ssize_t idx;
#pragma omp atomic capture
                idx = rightBlocks++;
                currentRightDirty[idx] = {right_start, right_end};
            }
        }
        // TODO
        std::cout << leftBlocks << ", " << rightBlocks << std::endl;
        exit(1);
        return l;
    }

    template <typename T>
    void quick(T *data, ssize_t l, ssize_t h)
    {
        if (l >= h)
            return;

        // Base case fallback optimization
        if (static_cast<size_t>(h - l) < TASK_THRESHOLD)
        {
            quick_sequential(data, l, h);
            return;
        }

        size_t mid;
        // Determine whether segment size justifies heavy parallel processing allocation
        if (static_cast<size_t>(h - l) > PARALLEL_THRESHOLD)
        {
            mid = partition_parallel(data, l, h);
        }
        else
        {
            mid = partition_sequential(data, l, h);
        }

// Task Tree Phase
#pragma omp task shared(data) firstprivate(l, mid)
        {
            quick(data, l, mid - 1);
        }
        quick(data, mid + 1, h);
    }

    template <typename T>
    void sort(std::vector<T> &data)
    {
        if (data.empty())
            return;

        T *arr = data.data();
        size_t l = 0;
        size_t h = data.size() - 1;

        // Requirement 1: Allow Nested Parallelism (Level 2) Explicit Configuration
        omp_set_nested(1); // Enable nesting explicitly
        omp_set_max_active_levels(2);

#pragma omp parallel
        {
// Requirement 1 & 2: Single thread kick-off to activate task generation context
#pragma omp single nowait
            {
                quick(arr, l, h);
            }
        }
    }
} // namespace quick_sort_parallel