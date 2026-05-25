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

    // --- High-Performance Parallel Partitioning Loop ---
    template <typename T>
    size_t partition_parallel(T *data, size_t l, size_t h)
    {
        auto pivot_index = select_pivot(data, l, h);
        auto pivot = data[pivot_index];

        // Hide the pivot safely away at the highest boundary index
        std::swap(data[pivot_index], data[h]);

        // Atomic block trackers (Atomic Capture Pointers)
        size_t next_left_block = l;
        size_t next_right_block = h; // Exclusive of pivot at index 'h'

        // Global arrays tracking chunks that were left partially processed ("dirty")
        // Size bounded by max thread potential allocation footprint
        constexpr size_t MAX_THREADS = 128;
        size_t dirty_left_blocks[MAX_THREADS];
        size_t dirty_right_blocks[MAX_THREADS];
        int num_dirty_left = 0;
        int num_dirty_right = 0;

// Nested parallel level 2 initialization configuration
#pragma omp parallel num_threads(omp_get_max_threads())
        {
            size_t my_l_curr = 0, my_l_end = 0;
            size_t my_r_curr = 0, my_r_end = 0;
            bool has_left = false, has_right = false;

            while (true)
            {
                // Requirement 3: Atomic capture allocation for the Left Pointers
                // Requirement 3: Atomic capture allocation for the Left Pointers
                if (!has_left)
                {
#pragma omp atomic capture
                    {
                        my_l_curr = next_left_block;
                        next_left_block += BLOCK_SIZE;
                    }
                    my_l_end = std::min(my_l_curr + BLOCK_SIZE, h);
                    if (my_l_curr >= next_right_block)
                    {
// Horizons crossed. Refund the block size we over-allocated.
#pragma omp atomic update
                        next_left_block -= BLOCK_SIZE;
                        break;
                    }
                    has_left = true;
                }

                // Requirement 3: Atomic capture allocation for the Right Pointers
                if (!has_right)
                {
#pragma omp atomic capture
                    {
                        my_r_end = next_right_block;
                        next_right_block = (next_right_block > BLOCK_SIZE) ? (next_right_block - BLOCK_SIZE) : l;
                    }
                    my_r_curr = (my_r_end > BLOCK_SIZE) ? std::max(my_r_end - BLOCK_SIZE, l) : l;
                    if (my_r_end <= next_left_block || my_r_curr >= my_r_end)
                    {
                        // Horizons crossed. Refund the precise amount we over-allocated from the right.
                        if (my_r_end > my_r_curr)
                        {
                            size_t allocated_size = my_r_end - my_r_curr;
#pragma omp atomic update
                            next_right_block += allocated_size;
                        }
                        break;
                    }
                    has_right = true;
                }

                // Requirement 4: Thread possesses two functional blocks. Cross-swap misaligned data items.
                while (my_l_curr < my_l_end && my_r_curr < my_r_end)
                {
                    while (my_l_curr < my_l_end && data[my_l_curr] < pivot)
                        my_l_curr++;
                    while (my_r_curr < my_r_end && data[my_r_end - 1] >= pivot)
                        my_r_end--;

                    if (my_l_curr < my_l_end && my_r_curr < my_r_end)
                    {
                        std::swap(data[my_l_curr], data[my_r_end - 1]);
                        my_l_curr++;
                        my_r_end--;
                    }
                }

                if (my_l_curr >= my_l_end)
                    has_left = false;
                if (my_r_curr >= my_r_end)
                    has_right = false;
            }

            // Requirement 4 & 5: Save unaligned "dirty" block remnants that ran out of cross-partners
            if (has_left && my_l_curr < my_l_end)
            {
                int idx;
#pragma omp atomic capture
                idx = num_dirty_left++;
                dirty_left_blocks[idx] = my_l_curr; // Mark start position of unprocessed elements
            }
            if (has_right && my_r_curr < my_r_end)
            {
                int idx;
#pragma omp atomic capture
                idx = num_dirty_right++;
                dirty_right_blocks[idx] = my_r_end; // Mark end position of unprocessed elements
            }
        } // Implicit synchronization barrier happens right here

        // Requirement 6: Sequential Cleanup phase for lingering dirty intersections
        size_t final_left_ptr = (next_left_block < h) ? next_left_block : l;
        size_t final_right_ptr = next_right_block;

        // Process residual isolated unaligned pieces from the dynamic tracking buffers
        for (int x = 0; x < num_dirty_left; ++x)
        {
            size_t curr = dirty_left_blocks[x];
            while (curr < final_right_ptr)
            {
                if (data[curr] >= pivot)
                {
                    while (final_right_ptr > curr && data[final_right_ptr - 1] >= pivot)
                    {
                        final_right_ptr--;
                    }
                    if (final_right_ptr > curr)
                    {
                        std::swap(data[curr], data[final_right_ptr - 1]);
                        final_right_ptr--;
                    }
                }
                curr++;
            }
        }

        // Handle structural center convergence using fallback partitioning mechanics
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

        // Restore the hidden central pivot back to its final structural split destination
        std::swap(data[i], data[h]);
        return i;
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