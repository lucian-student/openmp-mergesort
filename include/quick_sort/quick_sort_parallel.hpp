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
    constexpr size_t BLOCK_SIZE = 16384;         // Increased to reduce atomic contention
    constexpr size_t PARALLEL_THRESHOLD = 65536; // Raised to avoid parallel overhead on mid-sized chunks
    constexpr size_t TASK_THRESHOLD = 32768;     // Raised to drop to fast sequential sort sooner

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
        /*
        First question how do i deal with unpaired blocks.
        * lets say i have 13 blocks and 12 threads.
        * the problem is that: some of them will pair up and some of them wont.

        Detection of unpaired:
        1. if they have some intersection: they paired : left_start, right_start,left_end,righ_end
        2. if they ovelap more right_start,left_start,right_end,left_end
             * didnt pair up
             * how do i handle this unpaired problem? just ignore these blocks

        I will be assigning blocks using atomic capture.
        * in the end i can have only p dirty blocks

        Cleanup:
        At most there will be p dirty blocks.
        - left  bocks contain  larger elements then pivot
        - right blocks contain smaller elements then pivot

        I can pair these up:
        In the end there will be only left/right dirty blocks

        How do i find the middle? Where i can safely swap elements from dirty blocks?
        I can just remember last index of right/left block. If all right blocks are clean(middle will be determined by right blocks(for left viceversa).

        In that point i will just swap out all dirty values.
        */
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